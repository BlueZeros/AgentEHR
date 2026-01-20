from typing import List

from agentlite.agent_prompts.prompt_utils import (
    DEFAULT_PROMPT,
    PROMPT_TOKENS,
    task_chain_format,
)
from agentlite.commons import AgentAct, TaskPackage

BASE_INSTRUCTION = """
You have access to a patient's Electronic Health Record (EHR) system. The necessary data has been loaded and is ready for you to interact with using your tools.

The database schema has two main types of tables:
    * **Patient Tables:** These tables contain the patient's specific, event-based health information (e.g., `admissions`, `labevents`, `prescriptions`).
    * **Candidate Tables** contain candidate entities for the task. The candidate entities are references for you to complete the task.

Before begin to solving the task, you should first load the patient ehr according to the `subject_id` and `current_time` in the `<patient_info>` section.

Crucially, your task is a complex reasoning and clinical prediction task, not a simple information retrieval task. You must comprehensively retrieve patient information from all relevant EHR tables, synthesize these findings, and reason to provide all plausible predictions based on the instructions that follow. Therefore, you must conduct a multi-stage longitudinal synthesis by iteratively cross-referencing all available tables to triangulate clinical evidence, ensuring that every predictive hypothesis is rigorously validated against contradictory data and temporal trends before finalization. Note that you can adopt only one `tool_call` action at a time.
"""
# Your reasoning must strictly adhere to the ReAct (Reasoning and Acting) paradigm. Specifically, after every single tool invocation, you are required to use the `think` tool to articulate your intermediate reasoning, data interpretation, and strategic planning for the next action. This ensures transparency and robustness in the clinical inference process.
# 

TERMINATION_INSTRUCTION = """
**MAX_TURNS_REACHED.**
The agent's execution limit has been hit. No more actions or thinking steps are permitted.

You must now produce the final response with `finish` tool to the user's initial query based *only* on the cumulative information gathered in the action history.
"""

THINK_INSTRUCTION = """
Execute a highly efficient thinking step. Your resulting internal monologue **must be succinct and not exceed 200 words**.

Synthesize the knowledge derived from the last Observation, extracting the **key information** relevant to the user's objective. Identify the primary **information gap** that currently prevents a final answer. Based on this gap, articulate the single, most necessary **next step in reasoning** required to advance toward a conclusive solution.
"""

class PromptGen:
    """Prompt Generator Class"""

    def __init__(self) -> None:
        self.prompt_type = "BasePrompt"
        self.examples: dict[str, list] = {}

    def add_example(
        self,
        task: TaskPackage,
        action_chain: List[tuple[AgentAct, str]],
        example_type: str = "action",
    ):
        example_context = task_chain_format(task, action_chain)
        if example_type in self.examples:
            self.examples[example_type].append(example_context)
        else:
            self.examples[example_type] = [example_context]

    def __get_example__(self, example_type: str, index: int = -1):
        if example_type in self.examples:
            return self.examples[example_type][index]
        else:
            return None

    def __get_examples__(self, example_type: str, indices: List[int] = None) -> str:
        """get multiple examples for prompt"""
        # check if example_type exist in self.examples
        if not example_type in self.examples:
            return None
        else:
            num_examples = len(self.examples[example_type])
            if not indices:
                indices = list(range(num_examples))
            examples = [self.__get_example__(example_type, idx) for idx in indices]
            return "\n".join(examples)


class BasePromptGen(PromptGen):
    """
    this is the BasePrompt for agent to use.
    """

    def __init__(
        self,
        agent_role: str = None,
        constraint: str = DEFAULT_PROMPT["constraint"],
        instruction: str = DEFAULT_PROMPT["agent_instruction"],
    ):
        """Prompt Generator for Base Agent
        :param agent_role: the role of this agent, defaults to None
        :type agent_role: str, optional
        :param constraint: the constraint of this agent, defaults to None
        :type constraint: str, optional
        """
        super().__init__()
        self.prompt_type = "BaseAgentPrompt"
        self.agent_role = agent_role
        self.constraint = constraint
        self.instruction = instruction

    def __get_role_ins__(self):
        """use as the start of every action prompt. Highlight the role of this agent"""
        ## to-do
        return

    def __constraint_prompt__(self):
        if self.constraint:
            return f"""{PROMPT_TOKENS["constraint"]['begin']}\n{self.constraint}\n{PROMPT_TOKENS["constraint"]['end']}"""
        else:
            return ""
    
    def __instruction_prompt__(self):
        return f"""<agent_instruction>\n{BASE_INSTRUCTION}\n</agent_instruction>"""

    def __role_prompt__(self, agent_role):
        prompt = f"""{PROMPT_TOKENS["role"]['begin']}\n{agent_role}\n{PROMPT_TOKENS["role"]['end']}"""
        return prompt

    def __prompt_example__(self, prompt_example: str):
        prompt = f"""{PROMPT_TOKENS["example"]['begin']}\n{prompt_example}{PROMPT_TOKENS["example"]['end']}\n"""
        return prompt

    def __act_format_example__(self, act_call_example: str):
        prompt = f"""{DEFAULT_PROMPT["action_format"]}{PROMPT_TOKENS["action_format"]['begin']}\n{act_call_example}{PROMPT_TOKENS["action_format"]['end']}\n"""
        return prompt
    
    def thinking_prompt(self):
        return THINK_INSTRUCTION

    def termination_prompt(self):
        return TERMINATION_INSTRUCTION
    
    def action_mcp_prompt(
        self,
        task: TaskPackage,
        example_type: str = "action",
        example: str = None,
        **kwargs,
    ) -> str:
        """return the action generation prompt for agent
        :param task: the task to finish
        :type task: TaskPackage
        :param actions: the actions to take
        :type actions: List[BaseAction]
        :param action_chain: the history action-obs chain of this task
        :type action_chain: List[tuple[AgentAct, str]]
        :param labor_agents_doc: the title and description dict of the labor agent, defaults to None
        :type labor_agents_doc: dict[str, str], optional
        :param example_type: the type of example, defaults to "action"
        :type example_type: str, optional
        :param example: the example string, defaults to None
        :type example: str, optional
        :return: the prompt for agent to take action
        :rtype: str
        """
        # adding roles into prompt
        # prompt = f"""{self.__role_prompt__(self.agent_role)}\n{self.__instruction_prompt__()}\n"""
        # adding constraint into prompt
        # prompt += f"""{self.__constraint_prompt__()}\n"""
        # adding action doc into prompt

        prompt = ""
        # get task example
        if example:  # get from input
            prompt_example = example
        else:  # get from self.examples
            prompt_example = self.__get_examples__(example_type)

        if prompt_example:  # if have example, put into prompt
            prompt += self.__prompt_example__(prompt_example)

        # adding action observation chain
        # cur_session = task_chain_format(task, action_chain)

        prompt += f"{self.__instruction_prompt__()}\n\n{task.instruction}"
        return prompt