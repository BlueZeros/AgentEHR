from typing import List

from agentlite.agent_prompts.prompt_utils import (
    DEFAULT_PROMPT,
    PROMPT_TOKENS,
    task_chain_format,
)
from agentlite.commons import AgentAct, TaskPackage
from agentlite.agent_prompts.BasePrompt import BasePromptGen, PromptGen


class ReasoningBankPromptGen(BasePromptGen):
    """
    Prompt Generator for Reasoning Bank Memory Extraction.
    
    Generates prompts for extracting memory items from agent trajectories.
    Supports both success and failure trajectories, as well as self-contrast extraction.
    Adapted for Medical/Clinical scenarios.
    """
    
    def __init__(
        self,
        agent_role: str = None,
        constraint: str = DEFAULT_PROMPT["constraint"],
        instruction: str = DEFAULT_PROMPT["agent_instruction"],
        max_items: int = 3
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
        self.prompt_type = "ReasoningBankPrompt"
        self.max_items = max_items
    
    def extract_success_prompt(
        self,
        query: str,
        trajectory: str,
        final_state: str = None,
        model_output: str = None
    ) -> str:
        """
        Build the success extraction prompt from Figure 8 (Appendix A.1).
        
        Extracts generalizable strategies that contributed to success in clinical tasks.
        
        Args:
            query: Task query
            trajectory: Execution trace
            final_state: Final state (optional, not used in current template)
            model_output: Final output (optional, not used in current template)
        
        Returns:
            str: Complete success extraction prompt
        """
        prompt = f"""You are an expert in clinical reasoning and medical data analysis. You will be given a user query, the corresponding trajectory that represents how an agent successfully accomplished the medical task.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's successful trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
- You must first think why the trajectory is successful, and then summarize the insights.
- **Focus on extracting insights about how to correlate multiple data sources to form a holistic clinical picture.**
- **Do not summarize strategies that are already mentioned in the raw instructions; focus on the implicit reasoning strategies.**
- You can extract at most {self.max_items} memory items from the trajectory.
- You must not repeat similar or overlapping items.
- Do not mention specific websites, queries, or string contents, but rather focus on the generalizable insights.

## Output Format
Your output must strictly follow the Markdown format shown below. Ensure ALL fields (Title, Description, Content) are provided.

Required Format:
```
# Memory Item i
## Title <short title, max 15 words>
## Description <one sentence summary of the memory item>
## Content <1-3 sentences describing the insights learned to successfully accomplishing the task>
```

Query: {query}
Trajectory: {trajectory}"""
        
        return prompt
    
    def extract_failure_prompt(
        self,
        query: str,
        trajectory: str,
        final_state: str = None,
        model_output: str = None
    ) -> str:
        """
        Build the failure extraction prompt from Figure 8 (Appendix A.1).
        
        Extracts preventative lessons about what went wrong in clinical tasks.
        
        Args:
            query: Task query
            trajectory: Execution trace
            final_state: Final state (optional, not used in current template)
            model_output: Final output (optional, not used in current template)
        
        Returns:
            str: Complete failure extraction prompt
        """
        prompt = f"""You are an expert in clinical reasoning and medical data analysis. You will be given a user query, the corresponding trajectory that represents how an agent attempted to resolve the medical task but failed.

## Guidelines
You need to extract and summarize useful insights in the format of memory items based on the agent's failed trajectory.
The goal of summarized memory items is to be helpful and generalizable for future similar tasks.

## Important notes
- You must first reflect and think why the trajectory failed, and then summarize what lessons you have learned or strategies to prevent the failure in the future.
- **Focus on extracting insights about how to correlate multiple data sources to form a holistic clinical picture.**
- **Do not summarize strategies that are already mentioned in the raw instructions; focus on the implicit reasoning strategies.**
- You can extract at most {self.max_items} memory items from the trajectory.
- You must not repeat similar or overlapping items.
- Do not mention specific websites, queries, or string contents, but rather focus on the generalizable insights.

## Output Format
Your output must strictly follow the Markdown format shown below. Ensure ALL fields (Title, Description, Content) are provided.

Required Format:
```
# Memory Item i
## Title <short title, max 15 words>
## Description <one sentence summary of the memory item>
## Content <1-3 sentences describing the insights learned to successfully accomplishing the task>
```

Query: {query}
Trajectory: {trajectory}"""
        
        return prompt
    
    def extract_self_contrast_prompt(
        self,
        trajectories: List[tuple],
        query: str,
        max_aggregated_items: int = 5
    ) -> str:
        """
        Build self-contrast prompt for comparing multiple trajectories in clinical tasks.
        
        From Section 3.3.1: "comparing across the k sampled trajectories to
        extract more generalized memory items".
        
        Args:
            trajectories: List of (trajectory, final_state, model_output, query, success) tuples
            query: Task query
            max_aggregated_items: Maximum number of items to extract from all trajectories
        
        Returns:
            str: Self-contrast extraction prompt
        """
        # Format trajectories for comparison
        trajectory_texts = []
        for i, (traj, final_state, output, _, success) in enumerate(trajectories, 1):
            status = "SUCCESS" if success else "FAILURE"
            trajectory_texts.append(
                f"Trajectory {i} ({status}):\n{traj}\n"
                f"Final State: {final_state}\n"
                f"Output: {output}\n"
            )
        
        trajectories_section = "\n---\n".join(trajectory_texts)
        
        prompt = f"""You are an expert in clinical reasoning and medical data analysis. You will be given a user query and multiple trajectories showing how an agent attempted the medical task. Some trajectories may be successful, and others may have failed.

## Guidelines
Your goal is to compare and contrast these trajectories to identify the most useful and generalizable strategies as memory items.
Use self-contrast reasoning:
- Identify patterns and strategies that consistently led to success.
- Identify mistakes or inefficiencies from failed trajectories and formulate preventative strategies.
- Prefer strategies that generalize beyond specific pages or exact wording.

## Important notes
- Think first: Why did some trajectories succeed while others failed?
- **Focus on extracting insights about how to correlate multiple data sources to form a holistic clinical picture.**
- **Do not summarize strategies that are already mentioned in the raw instructions; focus on the implicit reasoning strategies.**
- You can extract at most {max_aggregated_items} memory items from all trajectories combined.
- Do not repeat similar or overlapping items.
- Do not mention specific websites, queries, or string contents — focus on generalizable behaviors and reasoning patterns.
- Make sure each memory item captures actionable and transferable insights.

## Output Format
Your output must strictly follow the Markdown format shown below:
```
# Memory Item i
## Title <the title of the memory item>
## Description <one sentence summary of the memory item>
## Content <1-5 sentences describing the insights learned to successfully accomplishing the task>
```

Query: {query}
Trajectories: {trajectories_section}"""
        
        return prompt
