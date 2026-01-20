import copy
from pprint import pprint
from typing import List

from agentlite.agent_prompts import ReflecToolPromptGen
from agentlite.agent_prompts.prompt_utils import DEFAULT_PROMPT
from agentlite.agents.agent_utils import *
from agentlite.commons import AgentAct, TaskPackage, EHRManager
from agentlite.commons.AgentAct import ActObsChainType
from agentlite.llm.agent_llms import BaseLLM
from agentlite.logging import DefaultLogger
from agentlite.logging.terminal_logger import AgentLogger
from agentlite.memory.AgentSTMemory import AgentSTMemory, MultipleTrialSTMemory

from agentlite.agents.MCPAgent import MCPBaseAgent


class MCPReflecToolAgent(MCPBaseAgent):
    """the base agent class for multi-turn action calling. Subclass from ABCAgent

    :param name: the name of this agent
    :type name: str
    :param role: the role of this agent
    :type role: str
    :param llm: the language model for this agent
    :type llm: BaseLLM
    :param actions: the action space that the agent can choose from, defaults to []
    :type actions: List[BaseAction], optional
    :param constraint: the constraints of this agent , defaults to "You generation should be simple and clear."
    :type constraint: str, optional
    :param instruction: the agent instruction, defaults to "You are an intelligent agent.\
        You should follow your {PROMPT_TOKENS["role"]['begin']}, {PROMPT_TOKENS["action"]['begin']} to take actions.\
            Your generation should follow the example format. Finish the task as best as you can.". 
            PROMPT_TOKENS is defined in agentlite/agent_prompts/prompt_utils.py
    :type instruction: str, optional
    :param reasoning_type: the reasoning type of this agent, defaults to "react". See BaseAgent.__add_inner_actions__ for more details.
    :type reasoning_type: str, optional
    :param logger: the logger for this agent, defaults to DefaultLogger
    :type logger: AgentLogger, optional

    Methods:
        - __call__(task: TaskPackage) -> str
    """

    def __init__(
        self,
        name: str,
        role: str,
        llm: BaseLLM,
        constraint: str = DEFAULT_PROMPT["constraint"],
        instruction: str = DEFAULT_PROMPT["agent_instruction"],
        logger: AgentLogger = DefaultLogger,
        mcp_url: str = None,
        **kwargs,
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

        self.search_strategy = kwargs["search_strategy"]
        self.search_size = kwargs["search_size"]
        self.ckpt_path = kwargs["ckpt_path"]

        self.prompt_gen = ReflecToolPromptGen(
            agent_role=self.role,
            constraint=self.constraint,
            instruction=self.instruction,
            ckpt_path=self.ckpt_path
        )

    def __add_st_memory__(self, short_term_memory: AgentSTMemory = None):
        """adding short-term memory to agent

        :param short_term_memory: the short-term memory, defaults to None
        :type short_term_memory: AgentSTMemory, optional
        """ """
        """
        if short_term_memory:
            self.short_term_memory = short_term_memory
        else:
            self.short_term_memory = MultipleTrialSTMemory(agent_id=self.id)
    
    async def execute(self, task: TaskPackage):
        """multi-step execution of actions. Generate the actions for a task until reach the done

        :param task: the task which agent receives and solves
        :type task: TaskPackage
        """
        step_size = 0
        same_steps = 0
        self.logger.execute_task(task=task, agent_name=self.name)
        while task.completion == "active" and step_size < self.max_exec_steps:
            action = await self.__next_act__(task)
            if action is None:
                step_size = self.max_exec_steps
                break
            
            self.logger.take_action(action, agent_name=self.name, step_idx=step_size+1)
            observation = await self.forward(task, action)
            self.logger.get_obs(obs=observation)
            if self.same_obs(observation):
                same_steps += 1
            else:
                same_steps = 0
            
            if same_steps >= self.max_same_steps:
                step_size = self.max_exec_steps
                break

            step_size += 1

        if task.completion == "active" and step_size >= self.max_exec_steps:
            action = await self.__forced_termination__()
            self.logger.take_action(action, agent_name=self.name, step_idx=step_size+1)
            observation = await self.forward(task, action)
            self.logger.get_obs(obs=observation)
            step_size += 1
        
        self.logger.end_execute(task=task, agent_name=self.name)
    
    async def __next_act__(
        self, task: TaskPackage
    ) -> AgentAct:
        """one-step action generation

        :param task: the task which agent receives and solves
        :type task: TaskPackage
        :param action_chain: history actions and observation of this task from memory
        :type action_chain: ActObsChainType
        :return: the action for agent to execute
        :rtype: AgentAct
        """
        # 获取所有 mcp 服务器 工具列表信息
        async with self.mcp_client: 
            response = await self.mcp_client.list_tools()
        # 生成 function call 的描述信息
        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
        } for tool in response]

        available_tools += self.inner_tools

        if self.search_strategy == "refine":
            response = await self.refine(task, available_tools)

        elif self.search_strategy == "select":
            response = await self.select(task, available_tools)

        else:
            raise NotImplementedError #, f"{self.search_strategy} should be in ['refine', 'select']"

        # self.logger.get_llm_output(response)
        return self.__action_parser__(response)

    async def refine(self, task, available_tools):
        raw_action = self.llm_layer(self.messages, available_tools)
        
        for _ in range(self.search_size - 1):
            previous_raw_action = raw_action
            current_action = self.__action_parser__(raw_action, add_action_messages=False, log_thinking=False)
            current_observation = await self.forward(task, current_action)
            act_messages = [
                raw_action.choices[0].message.model_dump(),
                {"role": "user", "content": current_observation}
            ]
            act_text = self.llm.tokenizer.apply_chat_template(act_messages, tokenize=False)

            refine_prompt = self.prompt_gen.refine_mcp_action_prompt(
                task,
                current_action=current_action,
                action_text=act_text,
            )

            self.messages.append(
                {
                    "role": "user",
                    "content": refine_prompt
                }
            )
            raw_action = self.llm_layer(self.messages, available_tools)
            self.messages = self.messages[:-1]

            # if previous_raw_action.choices[0].finish_reason == "tool_calls":
            #     if previous_raw_action.choices[0].message.tool_calls[0] == raw_action.choices[0].message.tool_calls[0]:
            #         break
            
            # else:
            #     if previous_raw_action.choices[0].message.content == raw_action.choices[0].message.content:
            #         break
        
        return raw_action

    async def select(self, task, available_tools):
        if self.search_size == 1:
            raw_action = self.llm_layer(self.messages, available_tools)
        else:
            raw_actions = self.llm_layer(self.messages, available_tools, n=self.search_size)
            
            candidate_actions = []
            candidate_action_texts = []
            # for choice in raw_actions.choices:
            #     raw_action = copy.deepcopy(raw_actions)
            #     raw_action.choices = [choice]
            #     agent_act = self.__action_parser__(raw_action, add_action_messages=False)
            #     observation = await self.forward(task, agent_act)
            #     candidate_actions.append((agent_act, observation))
            pprint(raw_actions)
            for choice in raw_actions.choices:
                raw_action = copy.deepcopy(raw_actions)
                raw_action.choices = [choice]
                agent_act = self.__action_parser__(raw_action, add_action_messages=False, log_thinking=False)
                observation = await self.forward(task, agent_act)
                candidate_actions.append((agent_act, observation))

                act_messages = [
                    choice.message.model_dump(),
                    {"role": "user", "content": observation}
                ]
                candidate_action_texts.append(self.llm.tokenizer.apply_chat_template(act_messages, tokenize=False))

            select_prompt = self.prompt_gen.select_mcp_action_prompt(
                task,
                candidate_actions=candidate_actions,
                candidate_action_texts=candidate_action_texts
            )
            
            self.messages.append(
                {
                    "role": "user",
                    "content": select_prompt
                }
            )
            
            raw_action = self.llm_layer(self.messages, available_tools)
            self.messages = self.messages[:-1]

        return raw_action
    
    async def optimization(self, task: TaskPackage) -> str:
        self.logger.receive_task(task=task, agent_name=self.name)
        self.assign(task)
        self.messages = []
        self.messages.append({
            "role": "user",
            "content": self.prompt_gen.action_mcp_prompt(task=task),
        })

        _iter = 0
        prediction_list = []
        while _iter < 2:
            self.short_term_memory.add_task_new_trial(task)
            await self.execute_one_trial(task, trial_iter=_iter)
            prediction_list.append(task.answer)
            _iter += 1
        
        return {
            "task": task,
            "action_chain_new": self.short_term_memory.get_action_chain(task)[1:],
            "action_chain_old": self.short_term_memory.get_prev_action_chain(task),
            "answer_new": prediction_list[-1],
            "answer_old": prediction_list[-2]
        }
        
        
    async def reflection(self, task: TaskPackage):
        """Generate a reflection based on previous action chains

        :param task: the task which agent receives and solves
        :type task: TaskPackage
        """
        prev_action_chain = self.short_term_memory.get_prev_action_chain(task)

        if prev_action_chain:
            # Create reflection prompt
            reflection_prompt = self.prompt_gen.reflection_prompt(
                task=task,
                prev_action_chain=prev_action_chain,
            )

            self.logger.get_prompt(reflection_prompt)

            self.messages.append({
                "role": "user",
                "content": reflection_prompt,
            })

            reflextion = self.llm_layer(self.messages)
            reflection_text = reflextion.choices[0].message.content.strip()
            self.messages.append({
                "role": "assistant", 
                "content": reflection_text,
            })

            sys_message = self.messages[0]
            ref_message = self.messages[-1]
            self.messages = [sys_message, ref_message]

            self.logger.get_llm_output(reflection_text)

            action = AgentAct(name="Reflection", params={"response": reflection_text})
            observation = "OK"
            self.logger.take_action(action, agent_name=self.name, step_idx=0)
            self.logger.get_obs(obs=observation)
            self.__st_memorize__(task, action, observation)
    
    async def execute_one_trial(self, task: TaskPackage, trial_iter: int = 0):
        """Execute one trial of the task with MCP

        :param task: the task which agent receives and solves
        :type task: TaskPackage
        :param trial_iter: the current trial iteration, defaults to 0
        :type trial_iter: int, optional
        """
        task.completion = "active"
        task.answer = None

        step_size = 0
        self.logger.execute_task(task=task, agent_name=self.name)
        if trial_iter > 0:
            await self.reflection(task)
        
        while task.completion == "active" and step_size < self.max_exec_steps:
            action = await self.__next_act__(task)
            self.logger.take_action(action, agent_name=self.name, step_idx=step_size+1)
            observation = await self.forward(task, action)
            self.logger.get_obs(obs=observation)
            self.__st_memorize__(task, action, observation)
            step_size += 1
        
        if task.completion == "active" and step_size == self.max_exec_steps:
            action = await self.__forced_termination__()
            self.logger.take_action(action, agent_name=self.name, step_idx=step_size+1)
            observation = await self.forward(task, action)
            self.logger.get_obs(obs=observation)
            step_size += 1
            
        self.logger.end_execute(task=task, agent_name=self.name)
        response = self.respond(task)
        return response