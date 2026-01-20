import os
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm


# def get_sample_weight(args, task_name, data_list):
#     if args.sample_mode == "normal":
#         return [1] * len(data_list)

#     item_file = os.path.join(args.item_path, f"{task_name}.csv")
#     item_df = pd.read_csv(item_file, header=None, names=['name', 'num'])

#     data_weight_list = []
#     for data in tqdm(data_list):
#         weight_list = []
#         for label in data["label"]:
#             if isinstance(label["name"], str) and label["name"].strip() in item_df["name"].values:
#                 label_weight = 1 / item_df[item_df["name"] == label["name"].strip()]["num"].item()
#                 weight_list.append(label_weight)
        
#         sample_weight = sum(weight_list) / (len(weight_list) + 1e-9)
#         data_weight_list.append(sample_weight)

#     avg_weight = sum(data_weight_list) / len(data_weight_list)

#     if args.sample_mode == "common":
#         data_weight_list = [weight if weight <= avg_weight else 0 for weight in data_weight_list]
    
#     elif args.sample_mode == "rare":
#         data_weight_list = [weight if weight >= avg_weight else 0 for weight in data_weight_list]
    
#     return data_weight_list

def get_sample_weight(args, task_name, data_list):
    # 模式为 "normal" 时，直接返回 [1] * len(data_list)
    if args.sample_mode == "normal":
        return [1] * len(data_list)

    # 1. 加载数据并预处理：创建 {name: 1/num} 的字典，实现 O(1) 查找
    item_file = os.path.join(args.item_path, f"{task_name}.csv")
    try:
        item_df = pd.read_csv(item_file, header=None, names=['name', 'num'])
    except FileNotFoundError:
        print(f"Error: Item file not found at {item_file}")
        return [0] * len(data_list)

    # 确保 'name' 列是字符串类型，并去除首尾空格
    item_df['name'] = item_df['name'].astype(str).str.strip()
    
    # 避免除以零，将 num=0 的行过滤掉，或按需处理
    item_df = item_df[item_df['num'] > 0] 

    # 创建 {name: 1/num} 的映射字典
    # 使用 pd.Series.to_dict() 比逐行迭代效率更高
    item_weight_map = (1 / item_df.set_index('name')['num']).to_dict()

    data_weight_list = []
    
    # 2. 优化主循环：使用字典 O(1) 查找，避免 Pandas 内部查找
    for data in tqdm(data_list, desc="Calculating sample weights"):
        
        # 使用列表推导式高效地计算所有有效 label 的权重
        weight_list = [
            item_weight_map.get(label["name"].strip())
            for label in data["label"]
            if isinstance(label["name"], str) and label["name"].strip() in item_weight_map
        ]
        
        # 计算样本平均权重
        if weight_list:
            sample_weight = sum(weight_list) / len(weight_list)
        else:
            # 如果 weight_list 为空 (没有找到任何匹配的 label)，权重设置为 0 或 1，取决于您的业务逻辑
            # 这里我设置为 0，因为分母为 1e-9 的情况可能意味着没有找到有效 label
            sample_weight = 0 
        
        data_weight_list.append(sample_weight)

    # 检查 data_weight_list 是否为空，避免除以零
    if not data_weight_list:
        return []

    # 3. 计算并应用过滤逻辑
    avg_weight = sum(data_weight_list) / len(data_weight_list)

    if args.sample_mode == "common":
        # 使用列表推导式高效地进行过滤
        return [weight if weight <= avg_weight else 0 for weight in data_weight_list]
    
    elif args.sample_mode == "rare":
        # 使用列表推导式高效地进行过滤
        return [weight if weight >= avg_weight else 0 for weight in data_weight_list]
    
    return data_weight_list

def main(args):
    for file in tqdm(os.listdir(args.input_path)):
        task_name = file.rsplit("_", 1)[0]

        if os.path.exists(os.path.join(args.output_path, f"{task_name}_{args.sample_num}{args.suffix}.json")) and args.resume:
            continue
        
        print(f"Procssing Task={task_name}")
        with open(os.path.join(args.input_path, file), "r") as f:
            data_list = json.load(f)

        data_weight_list = get_sample_weight(args, task_name, data_list)
        df_weights = pd.DataFrame(
            {
                'data_index': list(range(len(data_list))),
                'weight': data_weight_list
            }
        )

        sampled_weights_df = df_weights.sample(
            n=args.sample_num, 
            weights='weight', 
            replace=False,
            random_state=42 # 固定随机种子，方便演示
        )

        sampled_indices = sampled_weights_df["data_index"].tolist()
        sampled_list = [data_list[indice] for indice in sampled_indices]

        with open(os.path.join(args.output_path, f"{task_name}_{args.sample_num}{args.suffix}.json"), "w") as f:
            json.dump(sampled_list, f, indent=4, ensure_ascii=False)

def parse_args():
    parser = argparse.ArgumentParser(prog="Task Meta Data Generation")

    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--item_path", type=str, default="/sfs/rhome/liaoyusheng/data/Datasets/EHRAgent/EHRAgentBench/item_set")
    parser.add_argument("--sample_mode", type=str, choices=["common", "normal", "rare"], default="normal")
    parser.add_argument("--sample_num", type=int, default=100)
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--resume", action="store_true", default=False)

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    return args


if __name__ == '__main__':

    args = parse_args()
    main(args)


