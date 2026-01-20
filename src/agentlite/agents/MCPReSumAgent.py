import os
from typing import List
import json
from contextlib import AsyncExitStack

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from agentlite.agent_prompts import ReSumPromptGen
from agentlite.agent_prompts.prompt_utils import DEFAULT_PROMPT
from agentlite.agents.agent_utils import *
from agentlite.commons import AgentAct, TaskPackage, EHRManager
from agentlite.commons.AgentAct import ActObsChainType
from agentlite.llm.agent_llms import BaseLLM
from agentlite.logging import DefaultLogger
from agentlite.logging.terminal_logger import AgentLogger
from agentlite.memory.AgentSTMemory import AgentSTMemory, DictAgentSTMemory
from agentlite.agents.MCPAgent import MCPBaseAgent


class MCPReSumAgent(MCPBaseAgent):
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
    :param sum_per_step: number of rounds between summarization, defaults to 10
    :type sum_per_step: int, optional
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

        self.prompt_gen = ReSumPromptGen(
            agent_role=self.role,
            constraint=self.constraint,
            instruction=self.instruction,
        )
        
        self.sum_per_step = kwargs.get("sum_per_step", 10)
        self.max_context_tokens = kwargs.get("max_context_tokens", 40 * 1024 - 1000)
        self.summary_mode = kwargs.get("summary_mode", "add")
        self.resum_enabled = True
        self.token_count = 0
        self.last_summary = None

    def count_tokens(self, messages: List[dict]) -> int:
        return self.llm.token_count(messages)

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
            self.messages = self.llm._truncate_messages(self.messages)
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
            return None

    async def execute(self, task: TaskPackage):
        """Execute the task with automatic summarization support.
        
        :param task: the task which agent receives and solves
        :type task: TaskPackage
        """
        task.completion = "active"
        task.answer = None
        
        step_size = 0
        self.last_summary = None
        
        self.logger.execute_task(task=task, agent_name=self.name)
        
        while task.completion == "active" and step_size < self.max_exec_steps:         
            action = await self.__next_act__()
            self.logger.take_action(action, agent_name=self.name, step_idx=step_size+1)
            observation = await self.forward(task, action)
            self.logger.get_obs(obs=observation)
            self.__st_memorize__(task, action, observation)
            step_size += 1
            
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

    async def _perform_summarization(self, task: TaskPackage, step_size: int):
        if self.last_summary:
            recent_messages = self.messages[4:].copy()
        else:
            recent_messages = self.messages[3:].copy()

        #recent_messages = self.messages[1:].copy()
        #print(f"length of recent_messages: {len(recent_messages)}")
        #print(f"recent_messages: {recent_messages[0]}")

        summary_response = await self.summarize_conversation(
            task,
            recent_messages,
            self.last_summary,
            step_size=step_size
        )
        
        if summary_response:  
            self.last_summary = summary_response
            new_observation = self.prompt_gen.build_summary_observation(task.instruction, summary_response)

            # system_msg = self.messages[0:2]
            # self.messages = system_msg
            # if self.summary_mode == "add":
            #     self.messages.append({"role": "user", "content": new_observation, "tool_name": "summary"})
            
            # elif self.summary_mode == "add_drop":
            #     prev_len = len(self.messages)
            #     self.messages = [turn for turn in self.messages if "tool_name" not in turn or turn["tool_name"] != "summary"]
            #     assert len(self.messages) >= prev_len - 1

            #     self.messages.append({"role": "user", "content": new_observation, "tool_name": "summary"})
            
            # else:
            #     raise NotImplementedError

            system_msg = self.messages[0:3]
            self.messages = system_msg
            self.messages.append({"role": "assistant", "content": new_observation})

            # self.token_count = self.count_tokens(self.messages)
            # print(f"token_count: {self.token_count}")
