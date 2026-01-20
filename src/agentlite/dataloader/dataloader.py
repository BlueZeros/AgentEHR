import os
import json
import math
from agentlite.commons import TaskPackage
from agentlite.agent_prompts.task_prompt import TASK_PROMPT


def get_chunk_data(data_list, chunk_num, chunk_idx):
    """
    将列表分成 chunk_num 份，并返回第 chunk_idx 份的数据。
    
    :param data_list: 原始列表
    :param chunk_num: 总分块数
    :param chunk_idx: 目标块的索引 (从 0 开始)
    :return: 子列表
    """
    if not 0 <= chunk_idx < chunk_num:
        raise ValueError("chunk_idx 必须在 0 到 chunk_num-1 之间")

    n = len(data_list)
    
    # 计算基础步长
    size = n // chunk_num
    # 计算余数（即前多少块需要多分配一个元素，以保证不漏掉数据）
    remainder = n % chunk_num
    
    # 计算当前块的起始和结束位置
    # 如果 remainder > 0，前面的块会比后面的块多一个元素
    if chunk_idx < remainder:
        start = chunk_idx * (size + 1)
        end = start + (size + 1)
    else:
        start = chunk_idx * size + remainder
        end = start + size
        
    return data_list[start:end]

def calculate_true_best_at_k(scores_list, k, metric='f1'):
    """
    计算 Best@K 的期望值：从 N 个结果中随机选 k 个，最大值的期望
    """
    # 1. 提取目标指标并排序 (从小到大)
    s = sorted([item[metric] for item in scores_list])
    n = len(s)
    
    if k > n:
        k = n  # 如果 k 大于样本总数，退化为取全部里的最大值
    
    # 2. 根据组合公式计算期望
    # Best@K = sum_{i=k to n} [ comb(i-1, k-1) / comb(n, k) * s[i-1] ]
    expected_max = 0
    total_combinations = math.comb(n, k)
    
    for i in range(k, n + 1):
        # 索引从 0 开始，所以是 s[i-1]
        weight = math.comb(i - 1, k - 1) / total_combinations
        expected_max += weight * s[i - 1]
        
    return expected_max

class DataManager:
    def __init__(
        self, 
        data_path, 
        output_path, 
        ehr_path=None,
        resume=False, 
        num_runs=8, 
        eval_num=-1,
        chunk_num=1,
        chunk_idx=0,
        score_strategy='avg'
    ):
        self.data_path = data_path
        self.output_path = output_path
        self.ehr_path = ehr_path
        self.num_runs = num_runs  # Number of runs per task
        self.eval_num = eval_num
        self.chunk_num = chunk_num
        self.chunk_idx = chunk_idx
        self.score_strategy = score_strategy  # 'avg' or 'max'
        self.index = 0
        self.completed_runs = 0  # Track how many runs have been completed

        if self.chunk_num > 1:
            self.save_suffix = f"_{self.chunk_num}_{self.chunk_idx}"
        else:
            self.save_suffix = ""

        self.datas = self.load_data()
        
        self.results = []
        self.scores = []

        if resume:
            self.resume()
    
    def resume(self):
        if not os.path.exists(os.path.join(self.output_path, f"results{self.save_suffix}.json")):
            return

        with open(os.path.join(self.output_path, f"results{self.save_suffix}.json"), 'r') as f:
            results = json.load(f)
        
        self.results = results
        self.completed_runs = len(results)
        self.scores = [r['score'] for r in results]

    def load_data(self):
        with open(self.data_path, 'r', encoding='utf-8') as f:
            raw_datas = json.load(f)
        print(f"Loading {len(raw_datas)} data from {self.data_path}...")
        
        datas = []
        for data in raw_datas:
            subject_id = data['subject_id']
            if self.ehr_path is not None:
                if not os.path.exists(os.path.join(self.ehr_path, "database", f"patient_{subject_id}.db")) and not os.path.exists(os.path.join(self.ehr_path, f"patient_{subject_id}.db")):
                    continue
            datas.append(data)
        print(f"Maintaining {len(datas)} data with EHR file...")

        if self.eval_num > 0:
            datas = datas[:self.eval_num]
        print(f"Detecting {self.eval_num=}, maintaining first {self.eval_num}...")

        datas = get_chunk_data(datas, self.chunk_num, self.chunk_idx)

        return datas
    
    def load_scores(self):
        return self.scores
    
    def __len__(self):
        return len(self.datas) * self.num_runs - self.completed_runs
    
    def __getitem__(self, index):
        # Adjust index to account for completed runs
        actual_index = index + self.completed_runs
        data_index = actual_index // self.num_runs
        data = self.datas[data_index]
        subject_id = data['subject_id']
        prediction_time = data['prediction_time']
        ground_truth = data['label']
        task = data['task']

        task_package = TaskPackage(
            task=task,
            subject_id=subject_id,
            instruction=TASK_PROMPT[task].format(current_time=prediction_time, subject_id=subject_id),
            timestamp=prediction_time,
            ground_truth=ground_truth,
        )

        return task_package

    def add_results(self, task: TaskPackage):
        run_index = (len(self.results) % self.num_runs) + 1
        result_info = {
            "task_info": task.get_task_info(),
            "run_index": run_index,
            "execution_steps": task.execution_steps,
            "thinking_steps": task.thinking_steps,
            "prediction": task.prediction,
            "score": task.score,
            "execution_time": task.execution_time,
            "input_tokens": task.input_tokens,
            "output_tokens": task.output_tokens

        }
        self.results.append(result_info)
        self.scores.append(task.score)
    
    def save_result(self):
        with open(os.path.join(self.output_path, f"results{self.save_suffix}.json"), 'w') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=4)
    
    def statistic_tool(self):
        tool_num = {}
        for result in self.results:
            for step in result["execution_steps"]:
                if step["action"] not in tool_num:
                    tool_num[step["action"]] = 0
                
                tool_num[step["action"]] += 1

        
        for tool in tool_num:
            tool_num[tool] /= len(self.results)
        
        return tool_num
    
    def statistic_turn(self):
        turn_list = [len(result["execution_steps"]) for result in self.results]
        return sum(turn_list) / len(turn_list)
    
    def statistic_computational_consumption(self):
        consumption_dict = {}
        for key in ["execution_time", "input_tokens", "output_tokens"]:
            info_score = [result[key] for result in self.results]
            consumption_dict[key] = sum(info_score) / len(info_score)
        
        return consumption_dict
        

    def save_score(self):
        if not self.scores:
            return
        
        total_score = {"sample_num": len(self.results), "score": {"max": {}, "avg": {}}, "info": {}}
        num_original_tasks = len(self.scores) // self.num_runs
        
        # Group scores by original task and aggregate
        for i in range(num_original_tasks):
            task_scores = self.scores[i * self.num_runs: (i + 1) * self.num_runs]
            
            for metric in task_scores[0]:
                if metric not in total_score["score"]["max"]:
                    total_score["score"]["max"][metric] = 0
                if metric not in total_score["score"]["avg"]:
                    total_score["score"]["avg"][metric] = 0
                
                total_score["score"]["avg"][metric] += sum([score[metric] for score in task_scores]) / self.num_runs
                total_score["score"]["max"][metric] += max([score[metric] for score in task_scores])

                # add best@k
                for k in range(self.num_runs):
                    if f"best@{k+1}" not in total_score["score"]:
                        total_score["score"][f"best@{k+1}"] = {}
                    
                    if metric not in total_score["score"][f"best@{k+1}"]:
                        total_score["score"][f"best@{k+1}"][metric] = 0

                    total_score["score"][f"best@{k+1}"][metric] += calculate_true_best_at_k(task_scores, k+1, metric)

        
        # Average across all original tasks
        for score_type in total_score["score"]:
            for metric in total_score["score"][score_type]:
                total_score["score"][score_type][metric] /= num_original_tasks

        total_score["info"]["avg_turn"] = self.statistic_turn()
        total_score["info"]["avg_tool"] = self.statistic_tool()
        total_score["info"]["computational_consumption"] = self.statistic_computational_consumption()

        with open(os.path.join(self.output_path, f"scores{self.save_suffix}.json"), 'w') as f:
            json.dump(total_score, f, ensure_ascii=False, indent=4)
    
    def save(self):
        self.save_result()
        self.save_score()
