from typing import List

from agentlite.agent_prompts.prompt_utils import (
    DEFAULT_PROMPT,
    PROMPT_TOKENS,
    task_chain_format,
)
from agentlite.commons import AgentAct, TaskPackage
from agentlite.agent_prompts.BasePrompt import BasePromptGen


QUERY_SUMMARY_PROMPT = """You are an expert at analyzing conversation history and extracting relevant information. Your task is to thoroughly evaluate the conversation history and current question to provide a comprehensive summary that will help solve the task.

Task Guidelines 
1. Information Analysis:
   - Carefully analyze the conversation history to identify truly useful information.
   - Focus on information that directly contributes to answering the question.
   - Do NOT make assumptions, guesses, or inferences beyond what is explicitly stated in the conversation.
   - If information is missing or unclear, do NOT include it in your summary.

2. Summary Requirements:
   - Extract only the most relevant information that is explicitly present in the conversation.
   - Synthesize information from multiple exchanges when relevant.
   - Only include information that is certain and clearly stated in the conversation.
   - Do NOT output or mention any information that is uncertain, insufficient, or cannot be confirmed from the conversation.

3. Output Format: Your response should be structured as follows:
<summary>
- Essential Information: [Organize the relevant and certain information from the conversation history that helps address the question.]
</summary>

Strictly avoid fabricating, inferring, or exaggerating any information not present in the conversation. Only output information that is certain and explicitly stated.

Question
{{{question}}} 

Conversation History
{{{recent_history_messages}}}

Please generate a comprehensive and useful summary. Note that you are not permitted to invoke tools during this process.
"""


QUERY_SUMMARY_PROMPT_LAST = """You are an expert at analyzing conversation history and extracting relevant information. Your task is to thoroughly evaluate the conversation history and current question to provide a comprehensive summary that will help answer the question.

The last summary serves as your starting point, marking the information landscape previously collected. Your role is to:
- Analyze progress made since the last summary
- Identify remaining information gaps
- Generate a useful summary that combines previous and new information
- Maintain continuity, especially when recent conversation history is limited

Task Guidelines

1. Information Analysis:
   - Carefully analyze the conversation history to identify truly useful information.
   - Focus on information that directly contributes to answering the question.
   - Do NOT make assumptions, guesses, or inferences beyond what is explicitly stated.
   - If information is missing or unclear, do NOT include it in your summary.
   - Use the last summary as a baseline when recent history is sparse.

2. Summary Requirements:
   - Extract only the most relevant information that is explicitly present in the conversation.
   - Synthesize information from multiple exchanges when relevant.
   - Only include information that is certain and clearly stated.
   - Do NOT output or mention any information that is uncertain, insufficient, or cannot be confirmed.

3. Output Format: Your response should be structured as follows:
<summary>
- Essential Information: [Organize the relevant and certain information from the conversation history that helps address the question.]
</summary>

Strictly avoid fabricating, inferring, or exaggerating any information not present in the conversation. Only output information that is certain and explicitly stated.

Question
{{{question}}}

Last Summary
{{{last_summary}}}

Conversation History
{{{recent_history_messages}}}

Please generate a comprehensive and useful summary. Note that you are not permitted to invoke tools during this process.
"""

QUERY_SUMMARY_PROMPT_WITH_HISTORY = """You are an expert at analyzing conversation history. Your task is to generate a summary that consists of two distinct parts: an "Execution Log" (for process tracking) and "Essential Information" (for knowledge extraction).

### PART 1: Execution Log Requirements (CRITICAL for preventing loops)
Before analyzing information, you must first list the history of actions to ensure the agent knows what has already been attempted.
- List EVERY tool call made in the history.
- Include the specific **parameters/arguments** used.
- Record the outcome (Success, Failure, or "No results found").
- **Crucial:** Even if a tool returned nothing or failed, it MUST be listed here so the agent does not repeat it.

### PART 2: Information Analysis Requirements (Strict Adherence to Original Standards)
For the second part of the summary, follow these strict guidelines to extract knowledge:
1. Information Analysis:
   - Carefully analyze the conversation history to identify truly useful information.
   - Focus on information that directly contributes to answering the question.
   - Do NOT make assumptions, guesses, or inferences beyond what is explicitly stated.
   - If information is missing or unclear, do NOT include it in this section.

2. Summary Constraints:
   - Extract only the most relevant information that is explicitly present.
   - Synthesize information from multiple exchanges when relevant.
   - Only include information that is certain and clearly stated.
   - Do NOT output or mention any information that is uncertain, insufficient, or cannot be confirmed.

### Output Format
You must structure your response exactly as follows:

<execution_history>
- [Tool Name]: [Arguments] -> [Status/Result State]
</execution_history>

<summary>
- Essential Information: [Organize the relevant and certain information following the strict guidelines of Part 2.]
</summary>

Strictly avoid fabricating, inferring, or exaggerating any information in the <summary> section.

Question
{{{question}}} 

Conversation History
{{{recent_history_messages}}}

Please generate the comprehensive summary containing both the execution history and the essential information.

"""

QUERY_SUMMARY_PROMPT_WITH_HISTORY_LAST = """You are an expert at analyzing conversation history and extracting relevant information. Your task is to maintain and update the agent's memory state, which consists of two critical parts: an "Execution Log" (process history) and "Essential Information" (knowledge).

The `Last Summary` serves as your baseline. Your role is to **update** this baseline with new events from the `Conversation History`.

### PART 1: Execution Log Maintenance (Cumulative History)
You must maintain a complete record of what has been tried to prevent the agent from repeating actions (loops).
1. **Inherit:** Start with the execution history found in the `Last Summary`.
2. **Append:** Identify any NEW tool calls in the `Conversation History` and add them to the list.
3. **Format:** Keep the outcome description brief to save space.
   - Use: "[Tool Name]: [Arguments] -> [Status/Result State]"
   - Status examples: "Success: Facts extracted", "Success: Irrelevant data", "Error: [Reason]", "No results found".
4. **Constraint:** Do NOT remove old history items unless they are explicitly superseded or corrected. The agent needs to know what it did 5 steps ago.

### PART 2: Information Analysis (Strict Adherence to Original Standards)
For the knowledge section, you are refining the agent's understanding:
- **Analyze progress:** Combine facts from `Last Summary` with new findings in `Conversation History`.
- **Identify gaps:** Clearly see what is still missing based on the new results.
- **Strict Filtering:**
   - Focus on information that directly contributes to answering the question.
   - Do NOT make assumptions, guesses, or inferences beyond what is explicitly stated.
   - If information is missing or unclear, do NOT include it.

### Output Format
Your response must be structured as follows:

<execution_history>
- [Tool Name]: [Arguments] -> [Status/Result State]
</execution_history>

<summary>
- Essential Information: [Organize the relevant and certain information combining previous knowledge and new findings.]
</summary>

Strictly avoid fabricating, inferring, or exaggerating any information in the <summary> section.

Question
{{{question}}}

Last Summary
{{{last_summary}}}

Conversation History
{{{recent_history_messages}}}

Please generate the comprehensive summary containing both the execution history and the essential information.
"""

QUERY_EVOLVING_SUMMARY_PROMPT = """You are an expert at analyzing conversation history and extracting relevant information. Your task is to thoroughly evaluate the conversation history and current question to provide a comprehensive, self-evolving summary that consists of two distinct parts: Trajectory Guidance (for strategic direction) and Patient Information (for medical facts).

Task Guidelines 

1. Trajectory Guidance Analysis:
   - Carefully analyze the conversation history to synthesize the execution path into a methodology review.
   - Explicitly identify and mention any approaches or parameters that have already failed or yielded irrelevant results to ensure they are not repeated.
   - Formulate the initial clinical logic based on the information collected so far.
   - Provide a concrete, actionable directive for the immediate next step.

2. Patient Information Analysis:
   - Carefully analyze the conversation history to identify truly useful clinical information.
   - Focus on information that directly contributes to answering the question.
   - Do NOT make assumptions, guesses, or inferences beyond what is explicitly stated in the conversation.
   - If information is missing or unclear, do NOT include it in your summary.

3. Summary Requirements (Patient Information):
   - Extract only the most relevant information that is explicitly present in the conversation.
   - Synthesize information from multiple exchanges when relevant.
   - Only include information that is certain and clearly stated in the conversation.
   - Do NOT output or mention any information that is uncertain, insufficient, or cannot be confirmed from the conversation.

4. Output Format: Your response should be structured as follows:
<trajectory_guidance>
**Methodology & Action Review:**
[Synthesize the actions taken so far and their effectiveness. Crucial: Explicitly state what failed or yielded no results to prevent loops.]

**Current Clinical Logic:**
[Synthesize the current reasoning state based on findings and ruled-out possibilities.]

**Next Step Directive:**
[Provide a specific, one-sentence instruction for the next action.]
</trajectory_guidance>

<patient_information_summary>
- Essential Information: [Organize the relevant and certain information from the conversation history that helps address the question. Group into 'Confirmed Findings' and 'Negative Findings' if applicable.]
</patient_information_summary>

Strictly avoid fabricating, inferring, or exaggerating any information not present in the conversation. Only output information that is certain and explicitly stated.

Question
{{{question}}} 

Conversation History
{{{recent_history_messages}}}

Please generate a comprehensive and useful dual-stream summary. Note that you are not permitted to invoke tools during this process.
"""

QUERY_EVOLVING_SUMMARY_PROMPT_LAST = """You are an expert at analyzing conversation history and extracting relevant information. Your task is to thoroughly evaluate the conversation history and current question to provide a comprehensive, self-evolving summary that consists of two distinct parts: Trajectory Guidance (for strategic direction) and Patient Information (for medical facts).

The last summary serves as your starting point, marking the strategic and informational landscape previously collected. Your role is to:
- Analyze the effectiveness of actions taken since the last summary
- Synthesize the execution path into strategic insights to prevent loops
- Update the patient information with new confirmed findings
- Maintain continuity in clinical logic and provide clear future direction

Task Guidelines

1. Trajectory Guidance Analysis:
   - Synthesize the execution history into a methodology review rather than a simple list.
   - Explicitly identify and mention approaches or parameters that failed or yielded irrelevant results to ensure they are not repeated.
   - Formulate the current clinical logic based on the progress made.
   - Provide a concrete, actionable directive for the immediate next step.

2. Patient Information Analysis:
   - Carefully analyze the conversation history to identify truly useful clinical information.
   - Focus on information that directly contributes to answering the question.
   - Do NOT make assumptions, guesses, or inferences beyond what is explicitly stated in the conversation.
   - If information is missing or unclear, do NOT include it in your summary.

3. Summary Requirements (Patient Information):
   - Extract only the most relevant information that is explicitly present in the conversation.
   - Synthesize information from multiple exchanges when relevant.
   - Only include information that is certain and clearly stated in the conversation.
   - Do NOT output or mention any information that is uncertain, insufficient, or cannot be confirmed from the conversation.

4. Output Format: Your response should be structured as follows:
<trajectory_guidance>
**Methodology & Action Review:**
[Synthesize past actions and their effectiveness. Crucial: Explicitly state what failed or yielded no results to prevent loops.]

**Current Clinical Logic:**
[Synthesize the current reasoning state based on findings and ruled-out possibilities.]

**Next Step Directive:**
[Provide a specific, one-sentence instruction for the next action.]
</trajectory_guidance>

<patient_information_summary>
- Essential Information: [Organize the relevant and certain information from the conversation history that helps address the question. Group into 'Confirmed Findings' and 'Negative Findings' if applicable.]
</patient_information_summary>

Strictly avoid fabricating, inferring, or exaggerating any information not present in the conversation. Only output information that is certain and explicitly stated.

Question
{{{question}}}

Last Summary
{{{last_summary}}}

Conversation History
{{{recent_history_messages}}}

Please generate the comprehensive dual-stream summary. Note that you are not permitted to invoke tools during this process.
"""

class ReSumPromptGen(BasePromptGen):
    def __init__(
        self,
        agent_role: str = None,
        constraint: str = DEFAULT_PROMPT["constraint"],
        instruction: str = DEFAULT_PROMPT["agent_instruction"],
    ):
        super().__init__(
            agent_role=agent_role,
            constraint=constraint,
            instruction=instruction
        )
        self.prompt_type = "ReSumAgentPrompt"

    def resum_prompt(
        self,
        task: TaskPackage,
        messages: List[dict],
        last_summary: str = None
    ) -> str:
        task_prompt = f"""{self.agent_role}\n\n{task.instruction}"""
        recent_history_str = "\n".join([str(msg) for msg in messages])

        if not last_summary:
            query_prompt = QUERY_SUMMARY_PROMPT.replace("{{{question}}}", task_prompt).replace("{{{recent_history_messages}}}", recent_history_str)
        else:
            query_prompt = QUERY_SUMMARY_PROMPT_LAST.replace("{{{question}}}", task_prompt).replace("{{{recent_history_messages}}}", recent_history_str).replace("{{{last_summary}}}", last_summary)

        return query_prompt

    def build_summary_observation(self, question: str, summary: str):
        return (
            f"Question: {question}\n\n"
            "Below is a summary of the previous conversation. This summary condenses "
            "key information from earlier steps, so please consider it carefully. "
            "Assess whether the summary provides enough information to answer the question "
            "and use it as the basis for further reasoning and information gathering.\n\n"
            f"Summary: {summary}\n"
        )
    