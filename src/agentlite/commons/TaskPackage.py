import time
import uuid

from typing import Optional
from pydantic import BaseModel

class TaskPackage(BaseModel):
    task_id: str = str(uuid.uuid4())
    executor: str = ""
    instruction: str = ""
    completion: str = "active"

    task: str = ""
    subject_id: int = 0
    timestamp: str = time.time()
    ground_truth: list = []
    
    answer: str = ""
    prediction: list = []
    score: dict = {}

    execution_steps: list = []
    thinking_steps: list = []

    execution_time: float = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def __str__(self):
        return f"""Task ID: {self.task_id}\nInstruction: {self.instruction}\nTask Completion:{self.completion}\nAnswer: {self.answer}\nTask Executor: {self.executor}"""

    def get_task_info(self):
        return {
            "subject_id": self.subject_id,
            "prediction_time": self.timestamp,
            "ground_truth": self.ground_truth,
            "task": self.task
        }