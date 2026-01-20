import os
from typing import List
import pandas as pd
import json

from agentlite.agents.RetroSumEvolving.judge import TrajectoryJudge
from agentlite.agents.RetroSumEvolving.extractor import MemoryExtractor
from agentlite.agents.RetroSumEvolving.retriever import EHRRetriever
from agentlite.agents.RetroSumEvolving.consolidator import MemoryConsolidator
from agentlite.agents.RetroSumEvolving.models import MemoryItem, MemoryEntry, TrajectoryResult, ReActStep

from agentlite.agent_prompts import ReSumPromptGen
from agentlite.agent_prompts.prompt_utils import DEFAULT_PROMPT
from agentlite.agents.agent_utils import *
from agentlite.commons import AgentAct, TaskPackage, EHRManager
from agentlite.llm.agent_llms import BaseLLM
from agentlite.logging import DefaultLogger
from agentlite.logging.terminal_logger import AgentLogger
from agentlite.agents.MCPAgent import MCPBaseAgent


class MCPRetroSumAgent(MCPBaseAgent):
    """MCP-based agent with ReSum (Recursive Summarization) capabilities.

    This agent extends MCPBaseAgent to support long-context conversations by
    recursively summarizing the conversation history when token limits are approached.

    :param name: the name of this agent
    :type name: str
    :param role: the role of this agent
    :type role: str
    :param llm: the language model for this agent
    :type llm: BaseLLM
    :param constraint: the constraints of this agent, defaults to DEFAULT_PROMPT["constraint"]
    :type constraint: str, optional
    :param instruction: the agent instruction, defaults to DEFAULT_PROMPT["agent_instruction"]
    :type instruction: str, optional
    :param logger: the logger for this agent, defaults to DefaultLogger
    :type logger: AgentLogger, optional
    :param mcp_url: the MCP server URL, defaults to None
    :type mcp_url: str, optional
    :param summary_iteration: number of rounds between summarization, defaults to 10
    :type summary_iteration: int, optional
    """

    SUMMARIZE_TOOLS = {
        "get_records_by_time",
        "get_event_counts_by_time",
        "get_latest_records",
        "get_records_by_keyword",
        "get_records_by_value",
        "run_sql_query"
    }

    def __init__(
        self,
        name: str,
        role: str,
        llm: BaseLLM,
        constraint: str = DEFAULT_PROMPT["constraint"],
        instruction: str = DEFAULT_PROMPT["agent_instruction"],
        logger: AgentLogger = DefaultLogger,
        mcp_url: str = None,
        **kwargs
    ):
        super().__init__(
            name=name,
            role=role,
            llm=llm,
            constraint=constraint,
            instruction=instruction,
            logger=logger,
            mcp_url=mcp_url,
            **kwargs
        )
        self.task = kwargs["task_name"]

        self.prompt_gen = ReSumPromptGen(
            agent_role=self.role,
            constraint=self.constraint,
            instruction=self.instruction,
        )
        
        # summary config
        self.max_context_tokens = kwargs.get("max_context_tokens", 40 * 1024 - 1000)
        self.summary_type = kwargs.get("summary_type", "origin")
        self.summary_mode = kwargs.get("summary_mode", "add")
        self.sum_per_step = kwargs.get("sum_per_step", 1)
        self.max_same_steps = kwargs["max_same_steps"]
        self.token_count = 0
        self.last_summary = None

        # evolving config
        self.resum_enabled = True
        self.extract_from_failures = True
        self.retrieve_info = kwargs.get("retrieve_info", "latest")
        self.top_k_retrieval = kwargs.get("top_k_retrieval", 1)
        self.memory_cache_path = kwargs.get("memory_cache_path", None)
        self.max_memory_items = kwargs.get("max_memory_items", 10)
        self.update_memory = kwargs.get("update_memory", False)
        self.memory_inject = kwargs.get("memory_inject", "both")
        self.retrospect_context = kwargs.get("retrospect_context", "both")

        self.scores = []

        # Initialize all components without config
        self.judge = TrajectoryJudge(
            use_historical_comparison=True,
            use_weighted_score=False
        )
        
        self.extractor = MemoryExtractor(
            llm=self.llm,
            max_items=self.max_memory_items
        )
        
        if self.memory_cache_path:
            self.retriever = EHRRetriever(
                llm=self.llm,
                top_k=self.top_k_retrieval,
                similarity_threshold=kwargs.get("similarity_threshold", 0.5),
                cache_path=os.path.join(self.memory_cache_path, "embedding_cache.json"),
                embedding_model_path=kwargs.get("retriever_path", None),
                embedding_device=kwargs.get("retriever_device", 0)
            )
        
            self.consolidator = MemoryConsolidator(
                memory_bank_path=os.path.join(self.memory_cache_path, "memory_bank.json"),
                enable_logging=True
            )

    def count_tokens(self, messages: List[dict]) -> int:
        return self.llm.token_count(messages)
    
    def check_same_obs(self, temp_obs):
        if len(self.logger.current_execution_steps) < 2:
            self.same_steps = 0
        
        else:
            prev_obs = None
            for act_obs in self.logger.current_execution_steps[:-1][::-1]:
                if act_obs["action"] != "Single Tool Summary":
                    prev_obs = act_obs["observation"]
                    break

            # prev_obs = self.logger.current_execution_steps[-2]["observation"]
            if temp_obs == prev_obs:
                self.same_steps += 1
            
            else:
                self.same_steps = 0

    async def __next_act__(self) -> AgentAct:
        # 获取所有 mcp 服务器 工具列表信息
        available_tools = await self.get_available_tools()
        available_tools += self.inner_tools

        actor_context = []
        if self.retrospect_context == "summarizer":
            for turn in self.messages[::-1]:
                actor_context.append(turn)
                if "tool_name" in turn and turn["tool_name"] == "summary":
                    break
            
            actor_context = actor_context[::-1]
            if self.messages[0] not in actor_context:
                actor_context = self.messages[:2] + actor_context
        
        else:
            actor_context = self.messages

        response = self.llm_layer(actor_context, available_tools)
        # print(f"response: {response}", flush=True)
        # breakpoint()
        return self.__action_parser__(response)

    async def react(self, task: TaskPackage):
        """Execute the task with automatic summarization support.
        
        :param task: the task which agent receives and solves
        :type task: TaskPackage
        """
        task.completion = "active"
        task.answer = None
        
        step_size = 0
        self.last_summary = None
        full_trajectory = []
        
        self.logger.execute_task(task=task, agent_name=self.name)
        
        while task.completion == "active" and step_size < self.max_exec_steps:         
            action = await self.__next_act__()
            self.logger.take_action(action, agent_name=self.name, step_idx=step_size+1)
            observation = await self.forward(task, action)
            self.logger.get_obs(obs=observation)
            self.__st_memorize__(task, action, observation)
            step_size += 1

            full_trajectory.append(f"Action: {{name: {action.name}, params: {action.params}}}\nObservation: {observation}")
            
            # self.token_count = self.count_tokens(self.messages)
            # print(f"token_count: {self.token_count}")
            should_summarize = (
                (self.resum_enabled and self.token_count >= self.max_context_tokens * 0.9) or
                ((step_size + 1) % self.sum_per_step == 0)
            ) and (step_size < self.max_exec_steps)
            
            if should_summarize and task.completion == "active":
                await self._perform_summarization(task, step_size)
        
        if task.completion == "active" and step_size == self.max_exec_steps:
            action = await self.__forced_termination__()
            self.logger.take_action(action, agent_name=self.name, step_idx=step_size+1)
            observation = await self.forward(task, action)
            self.logger.get_obs(obs=observation)
            step_size += 1
        
        self.logger.end_execute(task=task, agent_name=self.name)

        await self._perform_summarization(task, step_size)

        return full_trajectory, self.last_summary

    async def summarize_conversation(
        self, 
        task: TaskPackage,
        messages: List[dict],
        last_summary: str = None,
        step_size: int = 0
    ):
        summary_prompt = self.prompt_gen.resum_prompt(task=task, messages=messages, last_summary=last_summary)

        summary_messages = [{"role": "user", "content": summary_prompt}]
        
        try:
            # self.messages = self.llm._truncate_messages(self.messages)
            response = self.llm_layer(summary_messages, tool_choice="none")
            summary = response.choices[0].message.content.strip()

            self.logger.get_llm_output(summary)

            action = AgentAct(name="Summary", params={"response": summary})
            observation = "OK"
            self.logger.take_action(action, agent_name=self.name, step_idx=step_size+1)
            self.logger.get_obs(obs=observation)
            self.__st_memorize__(task, action, observation)

            return summary
        except Exception as e:
            print(f"Error: {e}")
            return None

    async def _perform_summarization(self, task: TaskPackage, step_size: int):
        if self.retrospect_context == "actor":
            recent_messages = []
            for turn in self.messages[::-1]:
                recent_messages.append(turn)
                if "tool_name" in turn and turn["tool_name"] == "summary":
                    break
            
            recent_messages = recent_messages[::-1]
            if self.messages[0] not in recent_messages:
                recent_messages = self.messages[:2] + recent_messages

        else:
            if self.last_summary:
                recent_messages = self.messages[4:].copy()
            else:
                recent_messages = self.messages[3:].copy()

        summary_response = await self.summarize_conversation(
            task,
            recent_messages,
            self.last_summary,
            step_size=step_size
        )
        
        if summary_response:  
            self.last_summary = summary_response
            new_observation = self.prompt_gen.build_summary_observation(task.instruction, summary_response)

            if self.summary_mode == "add":
                self.messages.append({"role": "user", "content": new_observation, "tool_name": "summary"})
            
            elif self.summary_mode == "add_drop":
                prev_len = len(self.messages)
                self.messages = [turn for turn in self.messages if "tool_name" not in turn or turn["tool_name"] != "summary"]
                assert len(self.messages) >= prev_len - 1

                self.messages.append({"role": "user", "content": new_observation, "tool_name": "summary"})
            
            else:
                raise NotImplementedError
        
        # self.token_count = self.count_tokens(self.messages)
    
    async def get_ehr_data(self, task):
        subject_id = task.subject_id
        timestamp = task.timestamp

        async with self.mcp_client:
            if self.retrieve_info == "all":
                # Read table list
                table_list_result = await self.mcp_client.call_tool(
                    "read_resource_data",
                    {"uri": f"cache://ehr/ehr_data/{subject_id}/table_list.json"}
                )
                table_name_list = json.loads(table_list_result.content[0].text)
                table_list = []
                for table_name in table_name_list:
                    table_data_result = await self.mcp_client.call_tool(
                        "read_resource_data",
                        {"uri": f"cache://ehr/ehr_data/{subject_id}/{table_name}.json"}
                    )
                    table_data = json.loads(table_data_result.content[0].text)
                    df = pd.DataFrame(table_data)
                    table_list.append({
                        "table_name": table_name,
                        "table_data": df
                    })
                ehr_data = {
                    "subject_id": subject_id,
                    "timestamp": timestamp,
                    "table_list": table_list,
                }
            
            elif self.retrieve_info == "latest":
                table_list_result = await self.mcp_client.call_tool(
                    "read_resource_data",
                    {"uri": f"cache://ehr/ehr_data/{subject_id}/table_list.json"}
                )
                table_name_list = json.loads(table_list_result.content[0].text)
                table_list = []
                for table_name in table_name_list:
                    table_data_result = await self.mcp_client.call_tool(
                        "get_latest_records",
                        {"subject_id": str(subject_id), "table_name": table_name}
                    )
                    table_data = table_data_result.content[0].text
                    table_list.append({
                        "table_name": table_name,
                        "table_data": table_data
                    })
                ehr_data = {
                    "subject_id": subject_id,
                    "timestamp": timestamp,
                    "table_list": table_list,
                }

            else:
                raise NotImplementedError
        
        return ehr_data

    async def execute(self, task: TaskPackage):
        """
        Execute task with complete MCP ReasoningBank integration.

        Full cycle:
        0. Manually load EHR data
        1. Retrieve relevant memories
        2. Execute task with memory-augmented prompt (react)
        3. Judge success/failure
        4. Extract new memories
        5. Consolidate into memory bank

        Args:
            task: The task which agent receives and solves
        """ 
        query = task.instruction
        subject_id = task.subject_id
        timestamp = task.timestamp

        # Step 0: Manually call load_ehr tool before the main execution
        ehr_data = await self.get_ehr_data(task)

        # Step 1: Retrieve relevant memories
        retrieved_memories = []
        if self.memory_cache_path:
            memory_bank = self.consolidator.get_all_entries()
            retrieved_memories = self.retriever.retrieve(
                ehr_data, memory_bank, k=self.top_k_retrieval
            )
        
        # Inject memory into system prompt if available
        if retrieved_memories:
            if self.memory_inject == "both":
                reasoning_memory_augmented_prompt = self._build_memory_section(retrieved_memories, "reasoning_memory_items")
                # Update the first user message with memory context
                self.messages[0]["content"] += reasoning_memory_augmented_prompt

                summary_memory_augmented_prompt = self._build_memory_section(retrieved_memories, "summary_memory_items")
                self.prompt_gen.experience = summary_memory_augmented_prompt
            
            elif self.memory_inject == "actor":
                reasoning_memory_augmented_prompt = self._build_memory_section(retrieved_memories, "reasoning_memory_items")
                # Update the first user message with memory context
                self.messages[0]["content"] += reasoning_memory_augmented_prompt
            
            elif self.memory_inject == "summarizer":
                summary_memory_augmented_prompt = self._build_memory_section(retrieved_memories, "summary_memory_items")
                self.prompt_gen.experience = summary_memory_augmented_prompt
            
            else:
                raise NotImplementedError

        # Step 2: React (execute task with memory-augmented prompt)
        full_trajectory, summarized_trajectory = await self.react(task)

        # Build full trajectory string
        score, predictions = self.f1_score(task.answer, task.ground_truth)
        ground_truth = task.ground_truth
        model_output = task.answer

        if self.update_memory and self.memory_cache_path:
            # Step 3: Judge success/failure
            success = self.judge.judge_trajectory_success(score, self.scores)
            self.scores.append(score)

            # Step 4: Extract memories (respecting extract_from_failures setting)
            reasoning_memory_items, summary_memory_items = [], []
            should_extract = success or self.extract_from_failures

            if should_extract:
                reasoning_memory_items, summary_memory_items = self.extractor.extract_memories(
                    query, full_trajectory, summarized_trajectory, ground_truth, model_output, success
                )

            # Step 5: Consolidate
            self.consolidator.add_from_trajectory(
                subject_id,
                trajectory=summarized_trajectory,
                ground_truth=ground_truth,
                model_output=model_output,
                success=success,
                reasoning_memory_items=reasoning_memory_items,
                summary_memory_items=summary_memory_items,
                steps_taken=len(full_trajectory),
                timestamp=timestamp # Pass the EHR timestamp
            )
    
    def _build_memory_section(self, memories: List[tuple[float, MemoryEntry]], memory_type) -> str:
        """
        Build memory section to inject into prompt.

        Args:
            memories: Retrieved memories (list of (score, entry) tuples)

        Returns:
            str: Memory section text
        """
        memory_section = "## Relevant Past Experience\n\n"
        memory_section += "Here are relevant strategies from similar tasks:\n\n"

        for i, (score, mem) in enumerate(memories, 1):
            memory_section += f"### Memory {i} (Similarity: {score:.2f})\n"
            
            # Access memory items from object or dict
            memory_items = getattr(mem, memory_type, [])

            for item in memory_items:
                title = getattr(item, 'title', None) or (item.get('title') if isinstance(item, dict) else 'No Title')
                description = getattr(item, 'description', None) or (item.get('description') if isinstance(item, dict) else '')
                content = getattr(item, 'content', None) or (item.get('content') if isinstance(item, dict) else '')
                
                memory_section += f"- **{title}**: {description}\n"
                memory_section += f"  *Content*: {content}\n"
            memory_section += "\n"

        return memory_section
    
    def f1_score(self, output, standard_answer):
        predictions = set()
        try:
            if isinstance(output, str):
                pattern = r"```json\s*(.*?)\s*```"
                match = re.search(pattern, output, re.DOTALL)
                if match:
                    output = match.group(1).strip()
                output = eval(output)
            
            if isinstance(output, list):
                for item in output:
                    predictions.add(item)
            else:
                raise NotImplementedError
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Could not parse JSON: {e}")
        
        ground_truth = set()
        for answer in standard_answer:
            if answer['name'] is not None and not pd.isna(answer['name']) and isinstance(answer['name'], str):
                ground_truth.add(answer['name'])

        if len(predictions) == 0:
            return {'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0}, list(predictions)
        predictions_lower = {pred.lower() for pred in predictions}
        ground_truth_lower = {gt.lower() for gt in ground_truth}
        
        intersection = predictions_lower.intersection(ground_truth_lower)
        precision = len(intersection) / len(predictions_lower)
        recall = len(intersection) / len(ground_truth_lower) if len(ground_truth_lower) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        score = {
            'f1_score': f1_score,
            'precision': precision,
            'recall': recall
        }
        return score, list(predictions)