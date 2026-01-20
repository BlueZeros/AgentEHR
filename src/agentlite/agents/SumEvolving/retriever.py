import numpy as np
import json
import os
from typing import List, Dict, Callable


class SumEvolvingRetriever:
    def __init__(
        self,
        top_k: int,
        similarity_threshold: float,
        encode_fn: Callable[[str], np.ndarray],
        block_cache_path: str,
    ):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.encode_fn = encode_fn
        self.block_cache_path = block_cache_path
    
    def _load_cache(self, cache_path: str) -> Dict[str, np.ndarray]:
        """加载 embedding cache"""
        if not os.path.exists(cache_path):
            return {}
        
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            # 将 list 转换为 numpy array
            return {k: np.array(v) for k, v in cache_data.items()}
        except Exception as e:
            print(f"Error loading cache: {e}")
            return {}
    
    def retrieve_block(self, last_summary, temp_sum_messages: List, memories: List, top_k: int = None):
        """检索最相似的 block，返回 top_k 结果"""
        if top_k is None:
            top_k = self.top_k
        
        if last_summary is None:
            last_summary = ""
        
        # 加载 embedding cache
        block_embedding_cache = self._load_cache(self.block_cache_path)
        if not block_embedding_cache:
            return []
        
        # 构建 query
        formatted_steps = "\n".join([
            f"Action={step.action}, Params={step.params}, Observation={step.observation}"
            for step in temp_sum_messages
        ])
        query = f"Last Summary: {last_summary}\n\nCurrent Steps:\n{formatted_steps}"
        
        # 对 query 进行 embedding
        query_embedding = self.encode_fn(query)
        
        # 计算相似度
        similarities = []
        for block_id, block_embedding in block_embedding_cache.items():
            similarity = float(np.dot(query_embedding, block_embedding))
            if similarity >= self.similarity_threshold:
                # 从 memories 中找到对应的 block
                memory = next((m for m in memories if m.get('block_id') == block_id), {})
                similarities.append((similarity, memory))
        
        # 排序并返回 top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:top_k]

    def retrieve_full(self, query, trajectory_steps, final_state, model_output, success):
        pass