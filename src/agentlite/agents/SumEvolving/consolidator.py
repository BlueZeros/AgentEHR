from typing import List, Dict, Any, Callable
import json
import os
import numpy as np


class SumEvolvingConsolidator:
    """
    管理 SumEvolving 的 memory 和 embedding cache
    
    当前版本: 基础版 - 只添加不删除
    未来优化: 可以添加去重、质量过滤、容量限制等功能
    """
    
    def __init__(
        self,
        block_memory_path: str,
        block_cache_path: str,
        encode_fn: Callable[[str], np.ndarray],
        enable_logging: bool = True,
    ):
        """
        Args:
            block_memory_path: block memory 存储路径
            block_cache_path: block embedding cache 存储路径
            encode_fn: embedding 函数
            enable_logging: 是否启用日志
        """
        self.block_memory_path = block_memory_path
        self.block_cache_path = block_cache_path
        self.encode_fn = encode_fn
        self.enable_logging = enable_logging
        
        # 内存中的 memories
        self.block_memories: List[Dict[str, Any]] = []
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.block_memory_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.block_cache_path), exist_ok=True)
        
        # 加载现有数据
        self._load_memories()
    
    def _load_memories(self):
        """加载已保存的 memories"""
        if os.path.exists(self.block_memory_path):
            try:
                with open(self.block_memory_path, 'r') as f:
                    self.block_memories = json.load(f)
                if self.enable_logging:
                    print(f"Loaded {len(self.block_memories)} block memories from {self.block_memory_path}")
            except Exception as e:
                if self.enable_logging:
                    print(f"Error loading block memories: {e}")
                self.block_memories = []
        else:
            self.block_memories = []
    
    def _save_memories(self):
        """保存 memories 到文件"""
        try:
            with open(self.block_memory_path, 'w') as f:
                json.dump(self.block_memories, f, indent=2)
            if self.enable_logging:
                print(f"Saved {len(self.block_memories)} block memories to {self.block_memory_path}")
        except Exception as e:
            if self.enable_logging:
                print(f"Error saving block memories: {e}")
            raise
    
    def _load_cache(self) -> Dict[str, List[float]]:
        """加载 embedding cache"""
        if os.path.exists(self.block_cache_path):
            try:
                with open(self.block_cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                if self.enable_logging:
                    print(f"Error loading embedding cache: {e}")
                return {}
        return {}
    
    def _save_cache(self, cache: Dict[str, List[float]]):
        """保存 embedding cache 到文件"""
        try:
            with open(self.block_cache_path, 'w') as f:
                json.dump(cache, f, indent=2)
            if self.enable_logging:
                print(f"Saved {len(cache)} embeddings to {self.block_cache_path}")
        except Exception as e:
            if self.enable_logging:
                print(f"Error saving embedding cache: {e}")
            raise
    
    def update_block_memory(self, block_memory_items: List[Dict[str, Any]]):
        """
        更新 block memory 和 embedding cache
        
        当前版本: 基础版 - 只添加不删除
        
        Args:
            block_memory_items: reviewer 提取的经验列表
                每个 item 包含: block_id, experiences, summary_quality, key_insights, etc.
        """
        if not block_memory_items:
            return
        
        # 加载现有 cache
        embedding_cache = self._load_cache()
        
        added_count = 0
        for item in block_memory_items:
            block_id = item.get('block_id')
            if not block_id:
                if self.enable_logging:
                    print(f"Warning: block_memory_item missing block_id, skipping")
                continue
            
            # TODO: [优化点1] 去重检查
            # 可以检查 block_id 是否已存在，或基于内容相似度去重
            # 示例:
            # if block_id in existing_ids:
            #     continue
            # if self._is_duplicate_by_content(item):
            #     continue
            
            # TODO: [优化点2] 质量过滤
            # 可以根据 summary_quality 字段过滤低质量经验
            # 示例:
            # if item.get('summary_quality') == 'low':
            #     continue
            
            try:
                # 1. 添加到 memory
                self.block_memories.append(item)
                
                # 2. 生成 embedding
                # 构建用于 embedding 的文本
                block_text = self._construct_block_text(item)
                embedding = self.encode_fn(block_text)
                
                # 3. 添加到 cache
                embedding_cache[block_id] = embedding.tolist()
                
                added_count += 1
                
            except Exception as e:
                if self.enable_logging:
                    print(f"Error processing block_id {block_id}: {e}")
                continue
        
        # TODO: [优化点3] 容量限制
        # 当 memory/cache 数量超过阈值时，可以删除旧的或低质量的
        # 示例:
        # if len(self.block_memories) > MAX_CAPACITY:
        #     self.block_memories = self._prune_memories(self.block_memories)
        #     embedding_cache = self._prune_cache(embedding_cache)
        
        # 保存到文件
        if added_count > 0:
            self._save_memories()
            self._save_cache(embedding_cache)
            
            if self.enable_logging:
                print(f"Added {added_count} new block memories")
    
    def _construct_block_text(self, item: Dict[str, Any]) -> str:
        """
        从 memory item 构建用于 embedding 的文本
        
        策略: 将 last_summary, steps, new_summary 和 experiences 组合
        """
        parts = []
        
        if item.get('last_summary'):
            parts.append(f"Last Summary: {item['last_summary']}")
        
        if item.get('formatted_steps'):
            parts.append(f"Steps: {item['formatted_steps']}")
        
        if item.get('new_summary'):
            parts.append(f"New Summary: {item['new_summary']}")
        
        # 添加提取的经验
        if item.get('experiences'):
            exp_texts = []
            for exp in item['experiences']:
                exp_text = f"{exp.get('title', '')}: {exp.get('description', '')}"
                exp_texts.append(exp_text)
            parts.append(f"Experiences: {' | '.join(exp_texts)}")
        
        return "\n\n".join(parts)
    
    def get_block_trajectory_memories(self) -> List[Dict[str, Any]]:
        """
        获取所有 block trajectory memories
        
        Returns:
            List of memory items
        """
        # 重新加载以获取最新数据
        self._load_memories()
        return self.block_memories
    
    def update_full_memory(self, full_memory_items: List[Dict[str, Any]]):
        """
        更新 full trajectory memory（预留接口）
        
        TODO: 实现 full trajectory memory 的存储和管理
        """
        # TODO: 实现 full trajectory memory 逻辑
        pass
    
    # ===== 以下是预留的优化方法，标记为 TODO =====
    
    def _is_duplicate_by_content(self, new_item: Dict[str, Any]) -> bool:
        """
        TODO: [优化点1] 基于内容的去重检查
        
        可以使用:
        - 文本相似度比较
        - embedding 余弦相似度
        - 关键字段匹配
        """
        pass
    
    def _filter_by_quality(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        TODO: [优化点2] 基于质量的过滤
        
        可以根据:
        - summary_quality 字段
        - experiences 数量和质量
        - 使用频率/相似度分数
        """
        pass
    
    def _prune_memories(self, memories: List[Dict[str, Any]], max_size: int = 1000) -> List[Dict[str, Any]]:
        """
        TODO: [优化点3] Memory 容量限制和修剪
        
        策略:
        - 保留最新的 N 个
        - 保留高质量的
        - 基于使用频率
        - 基于时间衰减
        """
        pass
    
    def _prune_cache(self, cache: Dict[str, List[float]], max_size: int = 1000) -> Dict[str, List[float]]:
        """
        TODO: [优化点3] Cache 容量限制和修剪
        
        与 _prune_memories 保持一致
        """
        pass
