from typing import List, Dict, Any
import json
import uuid


class SumEvolvingReviewer:
    """从 block trajectory 中提取经验，用于改进未来的 summary"""
    
    def __init__(self, llm, max_experiences: int = 3):
        """
        Args:
            llm: LLM model
            max_experiences: 每个 block 最多提取的经验数量
        """
        self.llm = llm
        self.max_experiences = max_experiences
    
    def review_block_trajectory_summary(self, block_trajectory_history: List[tuple]) -> List[Dict[str, Any]]:
        """
        从 block trajectory history 中提取经验
        
        Args:
            block_trajectory_history: List of (last_summary, formatted_steps, new_summary)
            
        Returns:
            List of experience items with block_id and experience text
        """
        if not block_trajectory_history:
            return []
        
        experiences = []
        
        for idx, (last_summary, formatted_steps, new_summary) in enumerate(block_trajectory_history):
            prompt = self._build_review_prompt(last_summary, formatted_steps, new_summary)
            
            try:
                messages = [{"role": "user", "content": prompt}]
                response = self.llm.run(messages)
                content = response.choices[0].message.content.strip()
                
                # 解析 JSON 格式的经验
                experience_data = self._parse_experience(content)
                
                if experience_data:
                    # 直接用uuid生成全局唯一block_id，避免内容依赖和冲突
                    block_id = uuid.uuid4().hex
                    
                    experience_data['block_id'] = block_id
                    experience_data['last_summary'] = last_summary
                    experience_data['formatted_steps'] = formatted_steps
                    experience_data['new_summary'] = new_summary
                    experiences.append(experience_data)
                    
            except Exception as e:
                print(f"Error reviewing block {idx}: {e}")
                continue
        
        return experiences
    
    def _build_review_prompt(self, last_summary: str, formatted_steps: str, new_summary: str) -> str:
        """构建 review prompt"""
        prompt = f"""You are reviewing a trajectory block where an agent summarized medical records.

**Last Summary:**
{last_summary if last_summary else "None (first block)"}

**New Steps Taken:**
{formatted_steps}

**New Summary Generated:**
{new_summary}

**Your Task:**
Extract up to {self.max_experiences} key experiences that would help improve future summaries. Focus on:
1. What information was successfully captured or organized
2. What patterns or relationships were identified
3. What summarization strategies worked well
4. What could be improved in the summary approach

Return your response in JSON format:
{{
    "experiences": [
        {{
            "title": "Brief title of the experience",
            "description": "Detailed description of what was learned",
            "applicable_when": "When this experience should be applied"
        }}
    ],
    "summary_quality": "high/medium/low",
    "key_insights": "Main insights from this block"
}}

Only return the JSON, no additional text."""
        
        return prompt
    
    def _parse_experience(self, content: str) -> Dict[str, Any]:
        """解析 LLM 返回的经验"""
        try:
            # 尝试提取 JSON
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()
            
            data = json.loads(content)
            return data
            
        except Exception as e:
            print(f"Error parsing experience JSON: {e}")
            return None
    
    def review_full_trajectory(self, query: str, trajectory_steps: List, 
                              final_state: str, model_output: str, success: bool) -> List[Dict[str, Any]]:
        """
        从完整轨迹中提取经验（为未来的 full trajectory evolving 预留）
        
        Args:
            query: 原始查询
            trajectory_steps: 完整轨迹步骤
            final_state: 最终状态
            model_output: 模型输出
            success: 是否成功
            
        Returns:
            List of experience items
        """
        # TODO: 实现 full trajectory review
        return []
