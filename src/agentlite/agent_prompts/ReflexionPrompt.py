from typing import List
from agentlite.agent_prompts.prompt_utils import (
    DEFAULT_PROMPT,
    PROMPT_TOKENS,
    task_chain_format,
)
from agentlite.commons import AgentAct, TaskPackage
from agentlite.agent_prompts.BasePrompt import BasePromptGen


REFLCTION_PROMPT = """
# Objective: 
Provide a concise and professional medical critique of the previous EHR task attempt, identifying only the most critical shortcomings.

# Task Requirements:
{task_prompt}

# Previous Trajectory:
{prev_trajecotry}

# Critique:
Analyze the previous trajectory from a professional clinical perspective. Summarize the most significant failures in a clear, brief format. Your critique must cover the following areas:
 - Information Gaps: State which critical clinical information was missed from the available tables.
 - Logical Flaws: Identify the most significant errors in clinical reasoning or data synthesis.
 - Strategy Inefficiency: Point out major inefficiencies in the overall approach to the task.
 - Output Mismatch: State how the final output failed to meet the specific requirements of the original prompt.
"""

class ReflexionPromptGen(BasePromptGen):
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
        super().__init__(
            agent_role=agent_role,
            constraint=constraint,
            instruction=instruction
        )
        self.prompt_type = "ReflexionAgentPrompt"

    def reflection_prompt(
        self,
        task: TaskPackage,
        prev_action_chain: List[tuple[AgentAct, str]]
    ) -> str:
        task_prompt = f"""{self.agent_role}\n\n{task.instruction}"""
        prev_trajectory = action_chain_format(prev_action_chain)
        prompt = REFLCTION_PROMPT.format(task_prompt=task_prompt, prev_trajecotry=prev_trajectory)
        return prompt