from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import hashlib


@dataclass
class ReActStep:
    """
    Single step in ReAct (Reasoning + Acting) trajectory.

    Each step contains:
    - Action taken
    - Parameters used
    - Observation received
    """
    step_num: int
    action: str
    params: Dict[str, Any]
    observation: str

    def to_string(self) -> str:
        """Format as string for trajectory"""
        return f"<action>{self.action}</action>\n<params>{str(self.params)}</params>\n<observation>{self.observation}</observation>"


@dataclass
class Block:
    """Block trajectory 数据结构"""
    last_summary: str
    steps: List[ReActStep]
    new_summary: str
    block_id: Optional[str] = None
    
    def __post_init__(self):
        if self.block_id is None:
            self.block_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """生成唯一标识符"""
        content = f"{self.last_summary}_{len(self.steps)}_{self.new_summary}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def to_text(self) -> str:
        """转换为文本用于 embedding"""
        steps_text = "\n".join([
            f"Action={step.action}, Params={step.params}, Observation={step.observation}"
            for step in self.steps
        ])
        return f"Last Summary: {self.last_summary}\n\nSteps:\n{steps_text}\n\nNew Summary: {self.new_summary}"
