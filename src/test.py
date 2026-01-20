import os
import json
import sqlite3
import asyncio
import argparse
import logging
import sys
import re
import pandas as pd
from typing import Optional
from dataclasses import dataclass, field
from tqdm import tqdm
from transformers import HfArgumentParser
from agentlite.commons import TaskPackage
from agentlite.agents import get_agent
from agentlite.dataloader import DataManager

@dataclass
class ModelConfig:
    model_name_or_path: str = field(
        metadata={"help": "预训练模型名或本地路径，比如 bert-base-uncased"}
    )

    vllm_server_url: str = field(
        default=None,
        metadata={"help": "VLLM 服务模型的网址"}
    )

    temperature: float = field(
        default=0.7,
        metadata={"help": "0.7 for qwen3, 0.0 for other models"}
    )

    top_p: float = field(
        default=0.8,
        metadata={"help": ""}
    )

    top_k: int = field(
        default=20,
        metadata={"help": ""}
    )

    presence_penalty: float = field(
        default=0.0,
        metadata={"help": ""}
    )

    max_new_tokens: int = field(
        default=4096,
        metadata={"help": ""}
    )

    max_seq_len: int = field(
        default=64000,
        metadata={"help": ""}
    )

    enable_thinking: bool = field(
        default=False,
        metadata={"help": "only work for qwen3 serise."}
    )

    gpu_memory_utilization: float = field(
        default=0.7,
        metadata={"help": ""}
    )


@dataclass
class DataConfig:
    task: str = field(
        default="diagnoses_icd",
        metadata={"help": "Task name"}
    )
    data_path: str = field(
        default="../../EHRAgentBench/",
        metadata={"help": "Benchmark root path"}
    )
    output_path: str = field(
        default="../../results",
        metadata={"help": "Output path"}
    )
    exp_name: str = field(
        default=None,
        metadata={"help": "name of experiments"}
    )
    debug: bool = field(
        default=False,
        metadata={"help": ""}
    )
    resume: bool = field(
        default=False,
        metadata={"help": "Whether to resume from the last data"}
    )
    start_index: int = field(
        default=0,
        metadata={"help": ""}
    )
    num_runs: int = field(
        default=1,
        metadata={"help": "Number of runs per task"}
    )
    score_strategy: str = field(
        default='avg',
        metadata={"help": "Score aggregation strategy: 'avg' or 'max'"}
    )
    eval_num: int = field(
        default=-1,
        metadata={"help": "Number of sample for evaluation"}
    )
    chunk_num: int = field(
        default=1,
        metadata={"help": "num of the chunked data"}
    )
    chunk_idx: int = field(
        default=0,
        metadata={"help": "index of the chunked data"}
    )

@dataclass
class AgentConfig:
    agent_type: str = field(
        default="react",
        metadata={"help": "choose the type of agent"}
    )
    max_exec_steps: int = field(
        default=30,
        metadata={"help": "max execute step of the agent"}
    )
    max_same_steps: int = field(
        default=5,
        metadata={"help": "max same step of the agent"}
    )
    mcp_url: str = field(
        default=None,
        metadata={"help": "the url of the MCP server, only work for `mcp` agent"}
    )
    mcp_gpu_id: int = field(
        default=0,
        metadata={"help": "the gpu id of the local MCP server, only work for `mcp` agent"}
    )
    ehr_path: str = field(
        default=None,
        metadata={"help": "ehr path used for mcp server start local or check whether the db file of the sample exists."}
    )
    ckpt_path: str = field(
        default=None,
        metadata={"help": "the path of the checkpoint to load, only work for agent required optimization step"}
    )

    reflect_iter: int = field(
        default=1,
        metadata={"help": "max reflection iter of the agent, only work for `reflexion` agent"}
    )

    search_strategy: str = field(
        default="select",
        metadata={"help": "the strategy to select actions, only work for `reflectool` agent"}
    )
    search_size: int = field(
        default=2,
        metadata={"help": "the size of action candidates for search, only work for `reflectool` agent"}
    )
    load_scores: bool = field(
        default=False,
        metadata={"help": "whether to load the scores from the database, only work for `ReasoningBank` agent"}
    )
    update_memory: bool = field(
        default=False,
        metadata={"help": "whether to update the memory bank, only work for `ReasoningBank` agent"}
    )
    memory_cache_path: str = field(
        default=None,
        metadata={"help": "the path of the memory cache, only work for `ReasoningBank` agent"}
    )
    task_name: str = field(
        default=None,
        metadata={"help": "the name of the task, only work for `ReasoningBank and MCPAgentFold` agent"}
    )
    retriever_path: str = field(
        default="/sfs/data/ShareModels/Embeddings/bge-m3",
        metadata={"help": "the device to use for embedding model, only work for `ReasoningBank` agent"}
    )
    retriever_device: int = field(
        default=0,
        metadata={"help": "the device to use for embedding model, only work for `ReasoningBank` agent"}
    )
    memory_inject: str = field(
        default="both",
        metadata={"help": "adopt memory on actor, summarizer, or both."}
    )
    retrospect_context: str = field(
        default="both",
        metadata={"help": "adopt retrospect context on actor, summarizer, or both."}
    )
    summary_type: str = field(
        default="origin",
        metadata={"help": "select the type of summary strategy for single tool use, can only chose from ['origin', 'new', 'think', 'None'], 'None' for disable, only work for `summary-based` agent"}
    )
    summary_mode: str = field(
        default="add",
        metadata={"help": "the mode for single tool summary, can only chose from ['add', 'replace']"}
    )
    summary_content: str = field(
        default="patient",
        metadata={"help": "Configure the content type of the agent summary, can only chose from ['patient', 'trajectory']"}
    )
    block_sum: str = field(
        default="origin",
        metadata={"help": "select the type of summary strategy for block tool use, can only chose from ['origin'], 'None' for disable, only work for `summary-based` agent"}
    )
    sum_per_step: int = field(
        default=10,
        metadata={"help": "Configure the agent to summarize the context every few steps."}
    )
    enable_memory_injection: bool = field(
        default=False,
        metadata={"help": "Enable memory injection, only work for reasoningbanckagent or retrosumagent"}
    )

def f1_score(output, standard_answer):
    predictions = set()
    try:
        if isinstance(output, str):
            pattern = r"```json\s*(.*?)\s*```"
            match = re.search(pattern, output, re.DOTALL)
            if match:
                # 提取捕获组中的内容
                output = match.group(1).strip()

            pattern = r'\[(.*?)\]'
            match = re.search(pattern, output, re.DOTALL)
            if match:
                # match.group(0) 提取整个匹配的字符串（包括方括号）
                output = match.group(0).strip()

            output = output.replace("\\", "")
            output = eval(output)
        
        if isinstance(output, list):
            if isinstance(output[0], str):
                for item in output:
                    predictions.add(item)
            
            elif isinstance(output[0], dict):
                for item in output:
                    predictions.add(list(item.values())[0])
            
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError
    except (json.JSONDecodeError, Exception) as e:
        print(f"Warning: Could not parse JSON: {e}")
    
    ground_truth = set()
    for answer in standard_answer:
        if answer['name'] is not None and not pd.isna(answer['name']) and isinstance(answer['name'], str):
            ground_truth.add(answer['name'].strip())

    if len(predictions) == 0:
        return {'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0}, list(predictions), list(ground_truth)
    predictions_lower = {pred.lower() for pred in predictions}
    ground_truth_lower = {gt.lower() for gt in ground_truth}
    
    intersection = predictions_lower.intersection(ground_truth_lower)
    precision = len(intersection) / len(predictions_lower)
    recall = len(intersection) / len(ground_truth_lower) if len(ground_truth_lower) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    score = {
        'f1_score': f1_score,
        'precision': precision,
        'recall': recall
    }
    return score, list(predictions), list(ground_truth)

def calculate_score(task_package: TaskPackage):
    output = task_package.answer
    standard_answer = task_package.ground_truth
    
    score, predictions, _ = f1_score(output, standard_answer)

    task_package.prediction = predictions
    task_package.score = score

def setup_logging(log_file_path="/home/ma-user/work/liaoyusheng/projects/EHRAgent/results/500/Qwen3-30B-A3B-Instruct-2507/evaluation_errors.log"):
    
    # 创建一个根 Logger 实例
    logger = logging.getLogger('EvaluationLogger')
    logger.setLevel(logging.INFO) # 设置最低日志级别
    
    # 避免重复添加 Handler
    if not logger.handlers:
        # 1. 创建文件 Handler，用于写入文件
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(logging.ERROR) # 文件只记录 ERROR 及以上级别
        
        # 2. 创建控制台 Handler，用于输出到屏幕
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO) # 控制台记录 INFO 及以上级别

        # 定义日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 将 Handler 添加到 Logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger

def re_evaluation(data_manager):
    new_result_package = []
    for result in tqdm(data_manager.results):
        task_package = TaskPackage(
            task=result["task_info"]["task"],
            subject_id=result["task_info"]["subject_id"],
            timestamp=result["task_info"]["prediction_time"],
            prediction=result["prediction"],
            ground_truth=result["task_info"]["ground_truth"],
            execution_steps=result["execution_steps"],
            thinking_steps=result["thinking_steps"],
            answer=str(result["execution_steps"][-1]["observation"])
        )

        calculate_score(task_package)
        new_result_package.append(task_package)
    
    data_manager.results = []
    for task_package in new_result_package:
        data_manager.add_results(task_package)


async def evaluation(model_args=None, data_args=None, agent_args=None):
    start_index = getattr(data_args, 'start_index', 0)
    
    if start_index < 0:
        start_index = 0
        error_logger.warning("start_index was negative, resetting to 0.")
    error_logger.info(f"Evaluation will START from index: {start_index}")
    data_manager = DataManager(
        data_args.data_path, 
        data_args.output_path, 
        agent_args.ehr_path,
        resume=data_args.resume,
        num_runs=data_args.num_runs,
        eval_num=data_args.eval_num,
        chunk_num=data_args.chunk_num,
        chunk_idx=data_args.chunk_idx,
        score_strategy=data_args.score_strategy
    )
    re_evaluation(data_manager=data_manager)
        
    agent = get_agent(data_args, model_args, agent_args) # MCPBaseAgent类

    if agent_args.load_scores:
        scores = data_manager.load_scores()
        agent.scores = scores

    try:
        await agent.connect_to_server()
        error_logger.info("Successfully connected to server.")
    except Exception as e:
        error_logger.critical(
            f"FATAL: Failed to connect to server. Error: {e}",
            exc_info=True
        )
        return 

    iterable_data = iter(data_manager)

    for i in range(start_index):
        try:
            next(iterable_data)
            error_logger.debug(f"Skipped item {i}")
        except StopIteration:
            error_logger.error(
                f"Cannot skip to index {start_index}. "
                f"Data manager has fewer than {start_index} items. Evaluation terminated."
            )
            return

    try:
        total_items = len(data_manager)
    except TypeError:
        total_items = None # 如果不支持，则 tqdm 无法显示总数
        error_logger.warning("DataManager does not support len(), tqdm will not show total count.")

    for p_id, task_package in tqdm(
        enumerate(iterable_data, start=start_index),
        initial=start_index,
        total=total_items,
        desc="Processing Tasks"
    ):
        
        task_id_info = task_package.get('id', str(p_id)) if isinstance(task_package, dict) else str(p_id)
        error_logger.info(f"Processing task ID: {task_id_info}")

        # try:
        await agent(task_package)
        calculate_score(task_package)
        data_manager.add_results(task_package)
        data_manager.save()
        error_logger.info(f"Task ID {task_id_info} completed and results saved.")
            
        # except Exception as e:
        #     error_message = (
        #         f"!!! Error occurred on loop index: p_id = {p_id}. "
        #         f"Data package ID: {task_id_info}. "
        #         f"Error type: {type(e).__name__}. Message: {e}"
        #     )
        #     error_logger.error(
        #         error_message, 
        #         exc_info=True
        #     )
        #     # 保持原有的遇到错误就中断的逻辑
        #     error_logger.error(f"Breaking loop after error at p_id={p_id}.")
        #     break
    data_manager.save()

def parse_args():
    parser = HfArgumentParser((ModelConfig, DataConfig, AgentConfig))
    model_args, data_args, agent_args = parser.parse_args_into_dataclasses()

    if data_args.exp_name is None:
        data_args.exp_name = agent_args.agent_type

    data_args.data_name = os.path.basename(data_args.data_path).rsplit(".", 1)[0]
    data_args.output_path = os.path.join(data_args.output_path, data_args.data_name, data_args.exp_name)
    os.makedirs(data_args.output_path, exist_ok=True)

    return model_args, data_args, agent_args

if __name__ == "__main__":
    model_args, data_args, agent_args = parse_args()

    error_logger = setup_logging(log_file_path=os.path.join(data_args.output_path, "error.log"))
    error_logger.info("Application starting...")

    asyncio.run(evaluation(
        model_args=model_args,
        data_args=data_args,
        agent_args=agent_args
    ))

    print("Test completed. All results have been saved incrementally.")
