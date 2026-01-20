from typing import List, Optional, Any
import json
import openai
from datetime import datetime

from .models import MemoryItem
from agentlite.agent_prompts.RetroSumEvolvingPrompt import RetroSumEvolvingPrompt


class MemoryExtractor:
    """
    Extract structured memory items from agent trajectories.

    Uses dual-prompt approach:
    - Success trajectories: Extract generalizable strategies
    - Failure trajectories: Extract preventative lessons

    Uses temperature=1.0 for diverse extraction (per paper).
    """

    def __init__(self, llm, max_items):
        """
        Args:
            llm: LLM model
            temperature: Temperature for LLM
            max_items: Maximum number of memory items to extract
        """
        self.llm = llm
        self.max_items = max_items
        self.prompt_gen = RetroSumEvolvingPrompt(max_items=max_items)

    def extract_memories(
        self,
        query: str,
        trajectory: list[Any],
        summarized_trajectory: str,
        ground_truth: str,
        model_output: str,
        success: bool,
        source_task_id: Optional[str] = None
    ) -> List[MemoryItem]:
        """
        Extract memory items from a trajectory.

        Automatically selects the appropriate prompt based on success/failure.

        Args:
            query: Original task query
            trajectory: Full agent execution trace (List of ReActStep)
            ground_truth: ground_truth
            model_output: Final model output
            success: Whether the trajectory was successful
            source_task_id: Optional task identifier

        Returns:
            List[MemoryItem]: Extracted memory items (max 3)

        Raises:
            ValueError: If LLM response cannot be parsed
        """
        # Convert trajectory list to string if it is a list
        if isinstance(trajectory, list):
            # Token limit handling: Truncate if too long
            # Target < 60000 tokens. Approx 4 chars/token -> 240,000 chars.
            # Using 200,000 chars as safe limit.
            
            step_strs = trajectory
            
            # Simple token limit handling based on _truncate_messages logic
            MAX_TOKENS = 32000
            
            # Helper to count tokens (similar to agent_llms.py)
            def count_tokens(strs):
                text = "".join(strs)
                if hasattr(self.llm, 'tokenizer') and self.llm.tokenizer:
                    return len(self.llm.tokenizer.encode(text))
                return len(text) // 4  # Fallback approximation

            current_tokens = count_tokens(step_strs)
            
            # Truncate while keeping the last step (similar to keeping last message)
            if len(step_strs) > 1:
                while count_tokens(step_strs) > MAX_TOKENS and len(step_strs) > 0:
                    del step_strs[0]  # Remove the oldest history step (从前往后截断)
            
            trajectory_str = "\n".join(step_strs)
        else:
            trajectory_str = str(trajectory)

        # Select appropriate prompt using prompt generator
        reasoning_prompt = self.prompt_gen.extract_reasoning_prompt(query, trajectory_str, ground_truth, model_output)
        summary_prompt = self.prompt_gen.extract_summary_prompt(query, trajectory_str, summarized_trajectory, ground_truth, model_output)


        # Call LLM with temperature=1.0
        response = self._call_llm(reasoning_prompt)
        # Parse response to extract memory items
        reasoning_memory_items = self._parse_extraction_response(
            response,
            success_signal=success,
            source_task_id=source_task_id
        )
        response = self._call_llm(summary_prompt)
        # Parse response to extract memory items
        summary_memory_items = self._parse_extraction_response(
            response,
            success_signal=success,
            source_task_id=source_task_id
        )

        # Enforce max items limit
        if len(reasoning_memory_items) > self.max_items:
            reasoning_memory_items = reasoning_memory_items[:self.max_items]
        if len(summary_memory_items) > self.max_items:
            summary_memory_items = summary_memory_items[:self.max_items]

        return reasoning_memory_items, summary_memory_items
    
    def _call_llm(self, prompt: str) -> str:
        """
        Call the LLM with retry logic.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            str: The LLM response
        """
        max_try = 5
        try_iter = 0
        response = None
        while try_iter < max_try:
            try:
                # Wrap prompt in messages list for VLLMServer compatibility
                messages = [{"role": "user", "content": prompt}]
                llm_output = self.llm.run(messages, available_tools=None, n=1)
                
                # Normalize output to string
                if hasattr(llm_output, 'choices') and hasattr(llm_output.choices[0], 'message'):
                    # Handle OpenAI ChatCompletion object (VLLMServer)
                    response = llm_output.choices[0].message.content
                elif isinstance(llm_output, dict) and 'output' in llm_output:
                    # Handle dict output (VLLM / HFLocalLLM)
                    response = llm_output['output']
                elif isinstance(llm_output, str):
                    response = llm_output
                else:
                    # Fallback
                    response = str(llm_output)
                    
                break
            except Exception as e:
                try_iter += 1
                print(f"Warning: LLM call failed: {e}, retrying {try_iter}/{max_try}...")

        return response

    def _parse_extraction_response(
        self,
        response: str,
        success_signal: bool,
        source_task_id: Optional[str] = None
    ) -> List[MemoryItem]:
        """
        Parse LLM response to extract memory items from Markdown format.

        Args:
            response: LLM response text (Markdown format as per paper Figure 8)
            success_signal: Whether this was a success or failure trajectory
            source_task_id: Optional task identifier

        Returns:
            List[MemoryItem]: Parsed memory items

        Raises:
            ValueError: If response cannot be parsed as Markdown
        """
        # Clean response
        response = response.strip()

        # Remove markdown code fences if present
        if response.startswith("```"):
            lines = response.split("\n")
            # Remove first line (```)
            lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response = "\n".join(lines)

        # Split response into memory item sections
        # Each section starts with "# Memory Item"
        memory_items = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Split by "# Memory Item" headers
        sections = []
        current_section = []

        for line in response.split("\n"):
            if line.strip().startswith("# Memory Item"):
                if current_section:
                    sections.append("\n".join(current_section))
                current_section = [line]
            else:
                current_section.append(line)

        # Don't forget the last section
        if current_section:
            sections.append("\n".join(current_section))

        # Parse each section
        for section in sections:
            if not section.strip():
                continue

            # Extract title, description, and content
            title = None
            description = None
            content = None

            lines = section.split("\n")
            i = 0
            while i < len(lines):
                line = lines[i].strip()

                # Look for ## Title
                if line.startswith("## Title"):
                    # Extract title (can be on same line or next line)
                    title_text = line[8:].strip()  # Remove "## Title"
                    if not title_text and i + 1 < len(lines):
                        # Title on next line
                        title = lines[i + 1].strip()
                        i += 1
                    else:
                        title = title_text

                # Look for ## Description
                elif line.startswith("## Description"):
                    # Extract description (can be on same line or next line)
                    desc_text = line[14:].strip()  # Remove "## Description"
                    if not desc_text and i + 1 < len(lines):
                        # Description on next line
                        description = lines[i + 1].strip()
                        i += 1
                    else:
                        description = desc_text

                # Look for ## Content
                elif line.startswith("## Content"):
                    # Extract content (can be on same line or next lines)
                    content_text = line[10:].strip()  # Remove "## Content"
                    if not content_text and i + 1 < len(lines):
                        # Content on following lines
                        content_lines = []
                        i += 1
                        while i < len(lines) and not lines[i].strip().startswith("##") and not lines[i].strip().startswith("#"):
                            if lines[i].strip():
                                content_lines.append(lines[i].strip())
                            i += 1
                        content = " ".join(content_lines)
                        i -= 1  # Back up one since loop will increment
                    else:
                        content = content_text

                i += 1

            # Validate and create MemoryItem
            if title and content:
                # Handle missing description (LLM sometimes skips it)
                if not description:
                    description = title

                memory_item = MemoryItem(
                    title=title,
                    description=description,
                    content=content,
                    source_task_id=source_task_id,
                    success_signal=success_signal,
                    extraction_timestamp=timestamp
                )
                memory_items.append(memory_item)

        if not memory_items:
            raise ValueError(f"No valid memory items found in response. Response: {response[:500]}...")

        return memory_items

    def extract_with_self_contrast(
        self,
        trajectories: List[tuple[str, str, str, str, bool]],
        query: str,
        source_task_id: Optional[str] = None,
        max_aggregated_items: int = 5
    ) -> List[MemoryItem]:
        """
        Extract memories by comparing multiple trajectories (MaTTS parallel).

        This implements the self-contrast approach from Section 3.3.1.
        Compares multiple trajectories to identify robust patterns.

        Args:
            trajectories: List of (trajectory, final_state, model_output, query, success) tuples
            query: Original task query
            source_task_id: Optional task identifier
            max_aggregated_items: Maximum number of items to extract from all trajectories

        Returns:
            List[MemoryItem]: Aggregated memory items from comparison
        """
        # Build self-contrast prompt using prompt generator
        prompt = self.prompt_gen.extract_self_contrast_prompt(
            trajectories, 
            query, 
            max_aggregated_items=max_aggregated_items
        )

        # Call LLM
        response = self._call_llm(prompt)

        # Parse response
        memory_items = self._parse_extraction_response(
            response,
            success_signal=None,  # Mixed success/failure
            source_task_id=source_task_id
        )

        # Enforce max items limit
        if len(memory_items) > max_aggregated_items:
            memory_items = memory_items[:max_aggregated_items]

        return memory_items
