import os
import json
from typing import List

from agentlite.agent_prompts.prompt_utils import (
    DEFAULT_PROMPT,
    PROMPT_TOKENS,
    task_chain_format,
)
from agentlite.commons import AgentAct, TaskPackage
from agentlite.agent_prompts.BasePrompt import BasePromptGen
from agentlite.train.optimizer_utils import find_max_step

REFLCTION_PROMPT = """
# Objective: 
Provide a concise and professional medical critique of the previous EHR task attempt, identifying only the most critical shortcomings.

# Task Requirements:
{task_prompt}

# Previous Trajectory:
{prev_trajecotry}

# Ground_Truth:
{ground_truth}

# Critique:
Analyze the previous trajectory from a professional clinical perspective. Use the information provided in the Ground Truth to inform your critique, helping you identify what critical information was missed or how the reasoning went wrong. Summarize the most significant failures in a clear, brief format.

Your critique must cover the following areas:
 - Information Gaps: State which critical clinical information was missed from the available tables.
 - Logical Flaws: Identify the most significant errors in clinical reasoning or data synthesis.
 - Strategy Inefficiency: Point out major inefficiencies in the overall approach to the task.
 - Output Mismatch: State how the final output failed to meet the specific requirements of the original prompt.

** Note that your critique must not explicitly state any information from the Ground Truth!**
"""

REFINE_SEARCH = f"""You are an intelligent agent tasked with **Action Parameter Refinement**.

Your objective is to critically assess and refine the parameters of the proposed **Current Action**, considering the entire history of actions and their associated experiences.

### **Refinement Criteria**
1.  **Analyze Context:** Review the **Action History (Tool-Call History)** and **Action Experience (shown in {PROMPT_TOKENS['action_experience']['begin']})** to understand the progression toward the sub-goal.
2.  **Evaluate Current Action:**
    * **If an Observation is Provided:** Determine if the current action has successfully completed the expected sub-goal or yielded information pertinent to the problem's solution.
    * **If No Observation is Provided (Prediction):** Predict the efficacy of the proposed action in achieving its stated sub-goal.
3.  **Refine Parameters:** Modify the action's parameters to enhance its effectiveness or correct prior deficiencies. **The action type itself must remain unchanged.**
4.  **Goal Alignment & Contextual Coherence:** The refined action must strictly align with the overall **Task Description** and logically follow from the **Reasoning/Inference History (The N previous rounds of thought and analysis)**.

### **Output Constraint**
* **If refinement is unnecessary:** Return the original action with its original parameters verbatim.
* **If refinement is necessary:** Output the modified action with the updated parameters directly.

**Your final output must be the most appropriate tool-call action for the current step.**"""

SELECT_SEARCH = f"""You are an intelligent agent tasked with **Optimal Action Selection**.

Your objective is to select the **ACTION** from the **Candidate Actions (shown in {PROMPT_TOKENS['candidate_actions']['begin']})** list, which is presented in the following format, including the tool call and the subsequent observation:
```
<|im_start|>assistant
<tool_call>
ACTION
</tool_call><|im_end|>
<|im_start|>user
OBSERVATION
<|im_end|>
```

Your objective is to select the single best action from the **Candidate Actions (shown in {PROMPT_TOKENS['candidate_actions']['begin']})** list. This selection must be rigorously justified by referencing the comprehensive **Action History (Tool-Call History)** and accumulated **Action Experience (shown in {PROMPT_TOKENS['action_experience']['begin']})** .

### **Selection Criteria**
1.  **Contextual Coherence:** Assess which candidate action most effectively leverages the information gained from past tool calls.
2.  **Evaluate Current Action:**
    * **If an Observation is Provided:** Determine if the current action has successfully completed the expected sub-goal or yielded information pertinent to the problem's solution.
    * **If No Observation is Provided (Prediction):** Predict the efficacy of the proposed action in achieving its stated sub-goal.
3.  **Goal Alignment & Contextual Coherence:** Assess which candidate action, along with its observation, most effectively aligns with the **Task Description** and logically progresses the plan established in the **Reasoning/Inference History**.
4.  **Evaluate Current Action and Observation:** Based on the provided **Observation**, determine if the action has successfully completed the expected sub-goal or yielded the necessary information pertinent to the problem's solution, especially in the context of the *N* previous rounds of analysis. The most effective action/observation pair for the current step should be chosen.

### **Output Constraint**
* Your output must be the selected ACTION, including all its parameters, as it appears in the **Candidate Actions** list.
* **Do not output the observation or any explanatory text.**
"""

class ReflecToolPromptGen(BasePromptGen):
    """
    this is the BasePrompt for agent to use.
    """

    def __init__(
        self,
        agent_role: str = None,
        constraint: str = DEFAULT_PROMPT["constraint"],
        instruction: str = DEFAULT_PROMPT["agent_instruction"],
        ckpt_path: str = None
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
        self.prompt_type = "ReflecToolAgentPrompt"
        self.ckpt_path = ckpt_path

        if self.ckpt_path is not None:
            max_step = find_max_step(self.ckpt_path)
            ckpt_path = os.path.join(self.ckpt_path, f'step-{max_step}', "action_experience.json")

            print(f"Loading action experience from {ckpt_path}...")
            with open(ckpt_path, 'r') as f:
                self.action_experience = json.load(f)
        else:
            print(f"No found action experience from {self.ckpt_path}!")
            self.action_experience = None
    
    def __act_experience_prompt__(self, action_names: list):
        if self.action_experience is not None:
            action_names = list(set(action_names))
            action_experience_context = [f"""{act_name}: {self.action_experience.get(act_name, [])}""" for act_name in action_names if act_name in self.action_experience]
            action_experience_context = "\n".join(action_experience_context)
            return f"""{PROMPT_TOKENS["action_experience"]["begin"]}\n{action_experience_context}{PROMPT_TOKENS["action_experience"]["end"]}"""
        else:
            return f"""{PROMPT_TOKENS["action_experience"]["begin"]}\nNo Information{PROMPT_TOKENS["action_experience"]["end"]}"""
    
    def reflection_prompt(
        self,
        task: TaskPackage,
        prev_action_chain: List[tuple[AgentAct, str]]
    ) -> str:
        task_prompt = f"""{self.agent_role}\n\n{task.instruction}"""
        prev_trajectory = action_chain_format(prev_action_chain)
        prompt = REFLCTION_PROMPT.format(task_prompt=task_prompt, prev_trajecotry=prev_trajectory, ground_truth=task.ground_truth)
        return prompt
    
    def refine_mcp_action_prompt(
        self,
        task: TaskPackage,
        current_action: AgentAct,
        action_text: str,
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

        prompt = REFINE_SEARCH + "\n\n"
        prompt += self.__act_experience_prompt__([current_action.name]) + "\n\n"

        prompt += f"""Current Action: {action_text}"""

        return prompt

    def select_mcp_action_prompt(
        self,
        task: TaskPackage,
        candidate_actions: List[tuple[AgentAct, str]],
        candidate_action_texts: List[str],
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
        # adding action guideline 
        prompt = SELECT_SEARCH + "\n\n"
        prompt += self.__act_experience_prompt__([action.name for (action, _) in candidate_actions]) + "\n\n"

        prompt += "\n".join([f"""{PROMPT_TOKENS["candidate_actions"]['begin']}\n{action_text}\n{PROMPT_TOKENS["candidate_actions"]['end']}""" for action_text in candidate_action_texts])
        return prompt
    
    def action_fomat(action_name, action_observation):
        pass