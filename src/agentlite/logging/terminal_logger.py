import os

from agentlite.commons import AgentAct, TaskPackage
from agentlite.logging.utils import *
from agentlite.utils import bcolors

from .base import BaseAgentLogger

class AgentLogger(BaseAgentLogger):
    def __init__(
        self,
        log_file_name: str = "agent.log",
        FLAG_PRINT: bool = True,
        OBS_OFFSET: int = 99999,
        PROMPT_DEBUG_FLAG: bool = False,
        SAVE_TO_JSON: bool = False,
    ) -> None:
        super().__init__(log_file_name=log_file_name)
        self.FLAG_PRINT = FLAG_PRINT  # whether print the log into terminal
        self.OBS_OFFSET = OBS_OFFSET
        self.PROMPT_DEBUG_FLAG = PROMPT_DEBUG_FLAG
        
        self.save_to_json = SAVE_TO_JSON
        if self.save_to_json:
            self.current_task_info = {}
            self.current_execution_steps = []
            self.current_thinking_steps = []
            self.current_step_info = {}
            self.all_executions = {}

    def __color_agent_name__(self, agent_name: str):
        return f"""{bcolors.OKBLUE}{agent_name}{bcolors.ENDC}"""

    def __color_task_str__(self, task_str: str):
        return f"""{bcolors.OKCYAN}{task_str}{bcolors.ENDC}"""

    def __color_act_str__(self, act_str: str):
        return f"""{bcolors.OKBLUE}{act_str}{bcolors.ENDC}"""

    def __color_obs_str__(self, act_str: str):
        return f"""{bcolors.OKGREEN}{act_str}{bcolors.ENDC}"""

    def __color_prompt_str__(self, prompt: str):
        return f"""{bcolors.WARNING}{prompt}{bcolors.ENDC}"""

    def __save_log__(self, log_str: str):
        if self.FLAG_PRINT:
            print(log_str)
        with open(self.log_file_name, "a") as f:
            f.write(str_color_remove(log_str) + "\n")

    def receive_task(self, task: TaskPackage, agent_name: str):
        task_str = (
            f"""[\n\tTask ID: {task.task_id}\n\tInstruction: {task.instruction}\n]"""
        )
        log_str = f"""Agent {self.__color_agent_name__(agent_name)} """
        log_str += f"""receives the following {bcolors.UNDERLINE}TaskPackage{bcolors.ENDC}:\n"""
        log_str += f"{self.__color_task_str__(task_str=task_str)}"
        self.__save_log__(log_str=log_str)

    def execute_task(self, task: TaskPackage = None, agent_name: str = None, **kwargs):
        log_str = f"""===={self.__color_agent_name__(agent_name)} starts execution on TaskPackage {task.task_id}===="""
        self.__save_log__(log_str=log_str)
        
        # save_to_json
        if self.save_to_json:
            self.all_executions = {}
            self.current_execution_steps = []
            self.current_thinking_steps = []
            self.current_task_info = {
                'task_id': task.task_id,
                'instruction': task.instruction,
            }

    def end_execute(self, task: TaskPackage, agent_name: str = None):
        log_str = f"""========={self.__color_agent_name__(agent_name)} finish execution. TaskPackage[ID:{task.task_id}] status:\n"""
        task_str = f"""[\n\tcompletion: {task.completion}\n\tanswer: {task.answer}\n]"""
        log_str += self.__color_task_str__(task_str=task_str)
        log_str += "\n=========="
        self.__save_log__(log_str=log_str)
        
        # save_to_json
        if self.save_to_json:
            execution_record = {
                'task_info': self.current_task_info,
                'execution_steps': self.current_execution_steps.copy(),
                'thinking_steps': self.current_thinking_steps.copy(),
                'result': task.answer,
            }
            self.all_executions = execution_record
            self.current_execution_steps = []

    def take_action(self, action: AgentAct, agent_name: str, step_idx: int, think: str = None):
        if think is None:
            act_str = f"""{{\n\tname: {action.name}\n\tparams: {action.params}\n}}"""
            log_str = f"""Agent {self.__color_agent_name__(agent_name)} takes {step_idx}-step {bcolors.UNDERLINE}Action{bcolors.ENDC}:\n"""
            log_str += f"""{self.__color_act_str__(act_str)}"""
            self.__save_log__(log_str)
        else:
            act_str = f"""{{\n\tthink: {think}\n\tname: {action.name}\n\tparams: {action.params}\n}}"""
            log_str = f"""Agent {self.__color_agent_name__(agent_name)} takes {step_idx}-step {bcolors.UNDERLINE}Action{bcolors.ENDC}:\n"""
            log_str += f"""{self.__color_act_str__(act_str)}"""
            self.__save_log__(log_str)
            
        # save_to_json
        if self.save_to_json:
            self.current_step_info = {
                'step': step_idx,
                'think': think,
                'action': action.name,
                'params': str(action.params)
            }

    def add_st_memory(self, agent_name: str):
        log_str = f"""Action and Observation added to Agent {self.__color_agent_name__(agent_name)} memory"""
        self.__save_log__(log_str)

    def get_obs(self, obs: str):
        if len(obs) > self.OBS_OFFSET:
            obs = obs[: self.OBS_OFFSET] + "[TLDR]"
        log_str = f"""Observation: {self.__color_obs_str__(obs)}"""
        self.__save_log__(log_str)
        
        # save_to_json
        if self.save_to_json:
            step_record = self.current_step_info.copy()
            step_record['observation'] = obs
            self.current_execution_steps.append(step_record)
            self.current_step_info = {}

    def get_prompt(self, prompt):
        log_str = f"""Prompt: {self.__color_prompt_str__(prompt)}"""
        if self.PROMPT_DEBUG_FLAG:
            self.__save_log__(log_str)

    def get_llm_output(self, output: str):
        log_str = f"""LLM generates: {self.__color_prompt_str__(output)}"""
        if self.PROMPT_DEBUG_FLAG:
            self.__save_log__(log_str)
    
    def get_all_executions(self):
        return self.all_executions.copy() if hasattr(self, 'all_executions') else []

    def get_thinking(self, thinking: str):
        log_str = f"""Thinking: {self.__color_prompt_str__(thinking)}"""
        if self.PROMPT_DEBUG_FLAG:
            self.__save_log__(log_str)

        if self.save_to_json:
            self.current_thinking_steps.append(thinking)        
    