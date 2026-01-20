import re
from typing import List
import json
from contextlib import AsyncExitStack

from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from agentlite.agent_prompts import ReflexionPromptGen
from agentlite.agent_prompts.prompt_utils import DEFAULT_PROMPT
from agentlite.agents.agent_utils import *
from agentlite.commons import AgentAct, TaskPackage, EHRManager
from agentlite.commons.AgentAct import ActObsChainType
from agentlite.llm.agent_llms import BaseLLM
from agentlite.logging import DefaultLogger
from agentlite.logging.terminal_logger import AgentLogger
from agentlite.memory.AgentSTMemory import AgentSTMemory, MultipleTrialSTMemory
from agentlite.agents.MCPAgent import MCPBaseAgent


class MCPReflexionAgent(MCPBaseAgent):
    """MCP-based reflexion agent for multi-turn action calling with reflection capabilities.

    :param name: the name of this agent
    :type name: str
    :param role: the role of this agent
    :type role: str
    :param llm: the language model for this agent
    :type llm: BaseLLM
    :param constraint: the constraints of this agent , defaults to DEFAULT_PROMPT["constraint"]
    :type constraint: str, optional
    :param instruction: the agent instruction, defaults to DEFAULT_PROMPT["agent_instruction"]
    :type instruction: str, optional
    :param logger: the logger for this agent, defaults to DefaultLogger
    :type logger: AgentLogger, optional
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

        self.prompt_gen = ReflexionPromptGen(
            agent_role=self.role,
            constraint=self.constraint,
            instruction=self.instruction,
        )

        self.reflect_iter = kwargs["reflect_iter"]

    def __add_st_memory__(self, short_term_memory: AgentSTMemory = None):
        """adding short-term memory to agent

        :param short_term_memory: the short-term memory, defaults to None
        :type short_term_memory: AgentSTMemory, optional
        """
        if short_term_memory:
            self.short_term_memory = short_term_memory
        else:
            self.short_term_memory = MultipleTrialSTMemory(agent_id=self.id)

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
            self.messages = self.llm._truncate_messages(self.messages)
            reflextion = self.llm_layer(self.messages, tool_choice="none")
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

    async def execute(self, task: TaskPackage):
        iter = 0
        while iter <= self.reflect_iter:
            self.short_term_memory.add_task_new_trial(task)
            await self.execute_one_trial(task, trial_iter=iter)
            iter += 1
    
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
            action = await self.__next_act__()
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
        