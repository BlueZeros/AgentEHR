from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
import json


@dataclass
class MemoryItem:
    """
    Structured knowledge unit extracted from trajectory.

    Represents a distilled reasoning strategy or insight that can be
    reused across similar tasks.
    """
    title: str
    description: str
    content: str
    source_task_id: Optional[str] = None
    success_signal: Optional[bool] = None
    extraction_timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        """Create from dictionary"""
        return cls(**data)

    def __str__(self) -> str:
        return f"MemoryItem(title='{self.title}', success={self.success_signal})"


@dataclass
class MemoryEntry:
    """
    Complete memory bank entry including task, trajectory, and extracted items.

    This is the fundamental storage unit in ReasoningBank, containing:
    - Original task query
    - Agent trajectory (action history)
    - Success/failure signal
    - Extracted memory items
    """
    id: str
    subject_id: int
    timestamp: str
    trajectory: str
    success: bool
    reasoning_memory_items: Optional[List[MemoryItem]] = None
    summary_memory_items: Optional[List[MemoryItem]] = None
    ground_truth: Optional[str] = None
    model_output: Optional[str] = None
    steps_taken: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        if self.reasoning_memory_items:
            data['reasoning_memory_items'] = [item.to_dict() for item in self.reasoning_memory_items]
        if self.summary_memory_items:
            data['summary_memory_items'] = [item.to_dict() for item in self.summary_memory_items]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create from dictionary"""
        reasoning_memory_items = [MemoryItem.from_dict(item) for item in data['reasoning_memory_items']]
        data['reasoning_memory_items'] = reasoning_memory_items
        summary_memory_items = [MemoryItem.from_dict(item) for item in data['summary_memory_items']]
        data['summary_memory_items'] = summary_memory_items
        return cls(**data)

    def __str__(self) -> str:
        return f"MemoryEntry(id={self.id[:8]}, success={self.success}, items={len(self.memory_items)})"


@dataclass
class TrajectoryResult:
    """
    Result of agent execution on a single task.

    Contains complete information about the agent's interaction with
    the environment, including thinking steps, actions, and outcomes.
    """
    subject_id: int
    full_trajectory: str
    ground_truth: str
    model_output: str
    steps_taken: int
    success: Optional[bool] = None
    reasoning_memory_items: Optional[List[MemoryItem]] = None
    summary_memory_items: Optional[List[MemoryItem]] = None
    entry_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        if self.reasoning_memory_items:
            data['reasoning_memory_items'] = [item.to_dict() for item in self.reasoning_memory_items]
        if self.summary_memory_items:
            data['summary_memory_items'] = [item.to_dict() for item in self.summary_memory_items]
            
        return data


@dataclass
class ReActStep:
    """
    Single step in ReAct (Reasoning + Acting) trajectory.

    Each step contains:
    - Thinking/reasoning process
    - Action taken
    - Observation received
    """
    step_num: int
    think: str
    action: str
    observation: str

    def to_string(self) -> str:
        """Format as string for trajectory"""
        return f"<think>{self.think}</think>\n<action>{self.action}</action>\n<observation>{self.observation}</observation>"


@dataclass
class MaTTSResult:
    """
    Result from Memory-aware Test-Time Scaling.

    Contains multiple trajectories and aggregated insights from
    parallel or sequential scaling strategies.
    """
    query: str
    best_trajectory: TrajectoryResult
    all_trajectories: List[TrajectoryResult]
    aggregated_memories: List[MemoryItem]
    entry_id: str
    scaling_mode: str  # "parallel" or "sequential"
    scaling_factor: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'query': self.query,
            'best_trajectory': self.best_trajectory.to_dict(),
            'all_trajectories': [t.to_dict() for t in self.all_trajectories],
            'aggregated_memories': [m.to_dict() for m in self.aggregated_memories],
            'entry_id': self.entry_id,
            'scaling_mode': self.scaling_mode,
            'scaling_factor': self.scaling_factor
        }