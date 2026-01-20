import os
import argparse
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from joblib import Parallel, delayed
import random
import json


def generate_diagnose_icd_task(args):
    task_file = os.path.join(args.data_path, "hosp", "diagnoses_icd.csv")
    label_mapping_file = os.path.join(args.data_path, "hosp", "d_icd_diagnoses.csv")
    admission_file = os.path.join(args.data_path, "hosp", "admissions.csv")

    task_df = pd.read_csv(task_file)
    admission_df = pd.read_csv(admission_file)
    label_mapping_df = pd.read_csv(label_mapping_file)

    # filter by subject_id or subject_id_file
    if args.subject_id_file is not None:
        with open(args.subject_id_file, 'r') as f:
            subject_ids = [int(line.strip()) for line in f if line.strip().isdigit()]
        task_df = task_df[task_df["subject_id"].isin(subject_ids)]
    elif args.subject_id is not None:
        task_df = task_df[task_df["subject_id"] == int(args.subject_id)]

    # merge with admission data to get admission time
    task_df = task_df.merge(admission_df[["hadm_id", "admittime"]], on="hadm_id", how="left")
    
    # select first admission for each subject_id (earliest admittime)
    # first_admissions = task_df.groupby("subject_id")["admittime"].min().reset_index()
    # task_df = task_df.merge(first_admissions, on=["subject_id", "admittime"], how="inner")
    
    hadm_list = task_df["hadm_id"].unique().tolist()
    num = min(args.sample_num, len(hadm_list))
    sampled_hadm_list = random.sample(hadm_list, num)

    meta_datas = []
    for hadm_id in tqdm(sampled_hadm_list):
        discharge_time = admission_df[admission_df["hadm_id"] == hadm_id]["dischtime"].tolist()[0]

        # 检查 discharge_time 是否为有效字符串，跳过 NaN 值
        if pd.isna(discharge_time) or not isinstance(discharge_time, str):
            continue

        # set diagnose time 2 minutes before discharge time
        diagnose_time = (datetime.strptime(discharge_time, "%Y-%m-%d %H:%M:%S") - timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S")
        task_info = task_df[task_df["hadm_id"]==hadm_id].to_dict(orient="records")
        meta_data = {
            "subject_id": task_info[0]["subject_id"],
            "hadm_id": hadm_id,
            "prediction_time": diagnose_time,
            "task": "diagnoses_icd",
            "label": [
                {
                    "name": label_mapping_df[(label_mapping_df["icd_version"] == label_info["icd_version"]) & (label_mapping_df["icd_code"] == label_info["icd_code"])]["long_title"].tolist()[0],
                    "icd_code": label_info["icd_code"],
                    "icd_version": label_info["icd_version"],
                    "seq_num": label_info["seq_num"]
                } for label_info in task_info
            ]
        }
        meta_datas.append(meta_data)
    
    return num, meta_datas


def generate_diagnose_ccs_task(args):
    task_file = os.path.join(args.data_path, "hosp", "diagnoses_icd.csv")
    label_mapping_file = os.path.join(args.data_path, "hosp", "d_ccs_diagnoses.csv")
    admission_file = os.path.join(args.data_path, "hosp", "admissions.csv")

    task_df = pd.read_csv(task_file)
    admission_df = pd.read_csv(admission_file)
    label_mapping_df = pd.read_csv(label_mapping_file)

    # filter by subject_id or subject_id_file
    if args.subject_id_file is not None:
        with open(args.subject_id_file, 'r') as f:
            subject_ids = [int(line.strip()) for line in f if line.strip().isdigit()]
        task_df = task_df[task_df["subject_id"].isin(subject_ids)]
    elif args.subject_id is not None:
        task_df = task_df[task_df["subject_id"] == int(args.subject_id)]

    # merge with admission data to get admission time
    # task_df = task_df.merge(admission_df[["subject_id", "admittime"]], on="subject_id", how="left")
    
    # select first admission for each subject_id (earliest admittime)
    # first_admissions = task_df.groupby("subject_id")["admittime"].min().reset_index()
    # task_df = task_df.merge(first_admissions, on=["subject_id", "admittime"], how="inner")

    # --- 从这里开始修改，实现随机选择一次入院 ---

    # 1. 选取 admission_df 中需要的列，并确保 admittime 是 datetime 对象
    admission_info = admission_df[["subject_id", "hadm_id", "admittime", "dischtime"]].copy()
    admission_info["admittime"] = pd.to_datetime(admission_info["admittime"])
    admission_info["dischtime"] = pd.to_datetime(admission_info["dischtime"]) # 提前转换 dischtime
    
    # 2. 对每个 subject_id，随机选择一次 hadm_id/admittime
    # 使用 groupby 和 sample(1) 来保证每个 subject_id 只保留一条入院记录
    # 使用 random_state 确保采样结果可复现
    sampled_admissions_df = admission_info.groupby("subject_id").apply(
        lambda x: x.sample(1, random_state=42)
    ).reset_index(drop=True)

    # 3. 将诊断数据 (task_df) 与随机选中的入院记录进行合并
    # 使用 subject_id 和 hadm_id 进行内连接，这样 task_df 中只保留了
    # 与被随机选中的 hadm_id 对应的诊断记录。
    task_df = task_df.merge(
        sampled_admissions_df[["subject_id", "hadm_id", "admittime", "dischtime"]], 
        on=["subject_id", "hadm_id"], 
        how="inner"
    )

    # 3. **加速点 A: 预先计算诊断时间**
    # 将 dischtime 转换为 datetime，并计算 diagnose_time
    task_df["dischtime"] = pd.to_datetime(task_df["dischtime"])
    task_df["diagnose_time"] = (task_df["dischtime"] - timedelta(minutes=2)).dt.strftime("%Y-%m-%d %H:%M:%S")

    # 4. **加速点 B: 预先创建 Label 映射字典**
    # 将 label_mapping_df 转换为字典，加速在循环内的查找 (O(1) 替代 O(N))
    # 创建一个复合键 (icd_version, icd_code) 到 long_title 的映射
    label_map_dict = label_mapping_df.set_index(["icd_version", "icd_code"])["long_title"].to_dict()

    # 5. 采样 subject_id
    subject_list = task_df["subject_id"].unique().tolist()
    num = min(args.sample_num, len(subject_list))
    sampled_subject_list = random.sample(subject_list, num)

    # 6. **加速点 C: 仅处理采样的 subject_id 并使用 groupby/apply**
    # 过滤 task_df，只保留采样的 subject_id
    sampled_task_df = task_df[task_df["subject_id"].isin(sampled_subject_list)].copy()

    # 将所有诊断记录按 subject_id 分组，并使用 apply 来生成 meta_data
    def create_meta_data(group):
        # 排除 dischtime 为 NaT 的记录
        if pd.isna(group["dischtime"].iloc[0]):
             return None
             
        # dischtime, diagnose_time 和 hadm_id 在组内是相同的，直接取第一个
        subject_id = group["subject_id"].iloc[0]
        hadm_id = group["hadm_id"].iloc[0]
        prediction_time = group["diagnose_time"].iloc[0]
        
        labels = []
        for _, row in group.iterrows():
            icd_version = row["icd_version"]
            icd_code = row["icd_code"]
            
            # 使用预先创建的字典进行 O(1) 查找
            long_title = label_map_dict.get((icd_version, icd_code), "UNKNOWN_TITLE")
            
            labels.append({
                "name": long_title,
                "icd_code": icd_code,
                "icd_version": icd_version,
                "seq_num": row["seq_num"]
            })

        return {
            "subject_id": int(subject_id),
            "hadm_id": int(hadm_id),
            "prediction_time": prediction_time,
            "task": "diagnoses_ccs",
            "label": labels
        }

    # 使用 apply 替换 for 循环，显著提高性能
    meta_datas = sampled_task_df.groupby("subject_id").apply(create_meta_data).tolist()
    
    # 移除 apply 产生的 None 值（对应于 dischtime 为 NaN 的记录）
    meta_datas = [data for data in meta_datas if data is not None]
    
    # --- 到这里修改结束 ---
    # subject_list = task_df["subject_id"].unique().tolist()
    # num = min(args.sample_num, len(subject_list))
    # sampled_subject_list = random.sample(subject_list, num)

    # meta_datas = []
    # for subject_id in tqdm(sampled_subject_list):
    #     discharge_time = admission_df[admission_df["subject_id"] == subject_id]["dischtime"].tolist()[0]

    #     # 检查 discharge_time 是否为有效字符串，跳过 NaN 值
    #     if pd.isna(discharge_time) or not isinstance(discharge_time, str):
    #         continue

    #     # set diagnose time 2 minutes before discharge time
    #     diagnose_time = (datetime.strptime(discharge_time, "%Y-%m-%d %H:%M:%S") - timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S")
    #     task_info = task_df[task_df["subject_id"]==subject_id].to_dict(orient="records")
    #     meta_data = {
    #         "subject_id": subject_id,
    #         "hadm_id": task_info[0]["hadm_id"],
    #         "prediction_time": diagnose_time,
    #         "task": "diagnoses_ccs",
    #         "label": [
    #             {
    #                 "name": label_mapping_df[(label_mapping_df["icd_version"] == label_info["icd_version"]) & (label_mapping_df["icd_code"] == label_info["icd_code"])]["long_title"].tolist()[0],
    #                 "icd_code": label_info["icd_code"],
    #                 "icd_version": label_info["icd_version"],
    #                 "seq_num": label_info["seq_num"]
    #             } for label_info in task_info
    #         ]
    #     }
    #     meta_datas.append(meta_data)
    
    return num, meta_datas


def choose_data_by_time(df, time_column, time_point=None, num=100):
    if time_point is not None:
        df = df[df[time_column] == time_point]
    
    # ensure each subject_id appears only once by randomly selecting one record for each subject_id
    unique_subject_df = df.groupby('subject_id').apply(lambda x: x.sample(1)).reset_index(drop=True)
    
    # sample from unique subject_ids
    num = min(num, len(unique_subject_df))
    sampled_pairs = unique_subject_df.sample(n=num)
    sample_pairs_list = sampled_pairs[['subject_id', time_column]].to_dict(orient="records")

    return sample_pairs_list


def generate_procedures_icd_task(args):
    task_file = os.path.join(args.data_path, "hosp", "procedures_icd.csv")
    label_mapping_file = os.path.join(args.data_path, "hosp", "d_icd_procedures.csv")

    task_df = pd.read_csv(task_file)
    label_mapping_df = pd.read_csv(label_mapping_file)

    # filter by subject_id or subject_id_file
    if args.subject_id_file is not None:
        with open(args.subject_id_file, 'r') as f:
            subject_ids = [int(line.strip()) for line in f if line.strip().isdigit()]
        task_df = task_df[task_df["subject_id"].isin(subject_ids)]
    elif args.subject_id is not None:
        task_df = task_df[task_df["subject_id"] == int(args.subject_id)]

    sample_pairs_list = choose_data_by_time(task_df, "chartdate", time_point=args.chartdate, num=args.sample_num)

    meta_datas = []
    for pair in sample_pairs_list:
        chartdate = pair["chartdate"]
        subject_id = pair["subject_id"]

        # 检查 chartdate 是否为有效字符串，跳过 NaN 值
        if pd.isna(chartdate) or not isinstance(chartdate, str):
            continue

        charttime = chartdate + " 23:59:00"
        prediction_time = (datetime.strptime(charttime, "%Y-%m-%d %H:%M:%S") - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
        task_info = task_df[(task_df["subject_id"] == subject_id) & (task_df["chartdate"] == chartdate)].to_dict(orient="records")
        meta_data = {
            "subject_id": subject_id,
            "prediction_time": prediction_time,
            "task": "procedures_icd",
            "label": [
                {
                    "name": label_mapping_df[(label_mapping_df["icd_version"] == label_info["icd_version"]) & (label_mapping_df["icd_code"] == label_info["icd_code"])]["long_title"].tolist()[0],
                    "icd_code": label_info["icd_code"],
                    "icd_version": label_info["icd_version"],
                    "seq_num": label_info["seq_num"]
                } for label_info in task_info
            ]
        }
        meta_datas.append(meta_data)
    
    return len(sample_pairs_list), meta_datas


def generate_procedures_ccs_task(args):
    task_file = os.path.join(args.data_path, "hosp", "procedures_icd.csv")
    label_mapping_file = os.path.join(args.data_path, "hosp", "d_ccs_procedures.csv")

    task_df = pd.read_csv(task_file)
    label_mapping_df = pd.read_csv(label_mapping_file)

    # filter by subject_id or subject_id_file
    if args.subject_id_file is not None:
        with open(args.subject_id_file, 'r') as f:
            subject_ids = [int(line.strip()) for line in f if line.strip().isdigit()]
        task_df = task_df[task_df["subject_id"].isin(subject_ids)]
    elif args.subject_id is not None:
        task_df = task_df[task_df["subject_id"] == int(args.subject_id)]

    sample_pairs_list = choose_data_by_time(task_df, "chartdate", time_point=args.chartdate, num=args.sample_num)

    meta_datas = []
    for pair in sample_pairs_list:
        chartdate = pair["chartdate"]
        subject_id = pair["subject_id"]

        # 检查 chartdate 是否为有效字符串，跳过 NaN 值
        if pd.isna(chartdate) or not isinstance(chartdate, str):
            continue

        charttime = chartdate + " 23:59:00"
        prediction_time = (datetime.strptime(charttime, "%Y-%m-%d %H:%M:%S") - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
        task_info = task_df[(task_df["subject_id"] == subject_id) & (task_df["chartdate"] == chartdate)].to_dict(orient="records")
        meta_data = {
            "subject_id": subject_id,
            "prediction_time": prediction_time,
            "task": "procedures_ccs",
            "label": [
                {
                    "name": label_mapping_df[(label_mapping_df["icd_version"] == label_info["icd_version"]) & (label_mapping_df["icd_code"] == label_info["icd_code"])]["long_title"].tolist()[0],
                    "icd_code": label_info["icd_code"],
                    "icd_version": label_info["icd_version"],
                    "seq_num": label_info["seq_num"]
                } for label_info in task_info
            ]
        }
        meta_datas.append(meta_data)
    
    return len(sample_pairs_list), meta_datas


def process_labevents(pair, task_df, timestamp_col, label_mapping_df):
        """
        处理单个 (subject_id, charttime) pair 的逻辑（Labevents 任务）。
        """
        charttime = pair[timestamp_col]
        subject_id = pair["subject_id"]

        # 检查 charttime 是否为有效字符串，跳过 NaN 值
        if pd.isna(charttime) or not isinstance(charttime, str):
            return None  # 返回 None 表示跳过

        # 预测时间比 charttime 早一分钟
        try:
            # 注意：这里需要确保 datetime.strptime 格式与 charttime 的实际格式匹配
            dt_charttime = datetime.strptime(charttime, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            # 如果格式不匹配，跳过
            return None

        prediction_time = (dt_charttime - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
        
        # 筛选出当前 subject_id 和 charttime 对应的任务信息
        task_info = task_df[
            (task_df["subject_id"] == subject_id) & (task_df[timestamp_col] == charttime)
        ].to_dict(orient="records")

        # 构建 label 列表
        labels = []
        for label_info in task_info:
            itemid = label_info["itemid"]
            # 从 label_mapping_df 中查找对应的标签信息
            match_df = label_mapping_df[label_mapping_df["itemid"] == itemid]
            
            if not match_df.empty:
                # 假设每个 itemid 只有一行匹配
                match = match_df.iloc[0]
                
                labels.append({
                    "name": match["label"],
                    "itemid": itemid,
                    "fluid": match["fluid"],
                    "category": match["category"],
                })
            # else: 如果找不到对应的 itemid，可以选择跳过或使用默认值

        meta_data = {
            "subject_id": subject_id,
            "prediction_time": prediction_time,
            "task": "labevents", # 任务类型更新为 labevents
            "label": labels
        }
        return meta_data

def generate_labevents_task_vectorized(args):
    """
    使用 Pandas merge (向量化操作) 加速生成 labevents 任务。
    """
    # --- 1. 数据加载和预处理 ---
    task_file = os.path.join(args.data_path, "hosp", "labevents.csv")
    label_mapping_file = os.path.join(args.data_path, "hosp", "d_labitems.csv")

    try:
        task_df = pd.read_csv(task_file)
        label_mapping_df = pd.read_csv(label_mapping_file)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return 0, []

    # 识别时间戳列名
    cols = task_df.columns.tolist()
    timestamp_col = next((col for col in cols if 'time' in col.lower() or 'date' in col.lower()), None)
    
    if timestamp_col is None:
        raise ValueError("Could not find a timestamp column (containing 'time' or 'date').")

    # --- 2. 筛选 subject_id ---
    if args.subject_id_file is not None:
        try:
            with open(args.subject_id_file, 'r') as f:
                subject_ids = [int(line.strip()) for line in f if line.strip().isdigit()]
            task_df = task_df[task_df["subject_id"].isin(subject_ids)]
        except FileNotFoundError as e:
            print(f"Error reading subject ID file: {e}")
    elif args.subject_id is not None:
        task_df = task_df[task_df["subject_id"] == int(args.subject_id)]

    # --- 3. 获取样本数据和预处理 ---
    # 假设 choose_data_by_time 返回一个列表，其中包含 subject_id 和 timestamp_col
    # 这一步仍使用原函数以保持逻辑一致性
    sample_pairs_list = choose_data_by_time(task_df, timestamp_col, time_point=args.chartdate, num=args.sample_num)
    
    if not sample_pairs_list:
        return 0, []

    # 将样本列表转换为 DataFrame
    sample_df = pd.DataFrame(sample_pairs_list)
    # 确保用于合并的列类型一致
    sample_df[timestamp_col] = sample_df[timestamp_col].astype(str)
    
    total_samples = len(sample_pairs_list)

    # --- 4. 向量化合并 (代替 for 循环中的 task_info 查找) ---
    
    # 第一次合并：将选定的样本 (subject_id, charttime) 与完整的任务数据合并
    # 得到所有需要作为标签的任务记录
    merged_task_df = pd.merge(
        sample_df[[timestamp_col, 'subject_id']], 
        task_df, 
        on=['subject_id', timestamp_col], 
        how='left' # 使用 left merge 确保只包含 sample_df 中的 (subject_id, charttime) 对
    )
    
    # 确保合并成功后，丢弃所有NaN行，即没有匹配到 labevents 数据的样本
    merged_task_df = merged_task_df.dropna(subset=['itemid', 'subject_id', timestamp_col]).copy()
    
    # --- 5. 向量化标签查找 (代替 for 循环中的 label_mapping_df 查找) ---
    
    # 仅选择 label_mapping_df 中需要的列进行第二次合并
    label_cols = ['itemid', 'label', 'fluid', 'category']
    final_df = pd.merge(
        merged_task_df,
        label_mapping_df[label_cols],
        on='itemid',
        how='left'
    )

    # --- 6. 向量化计算 prediction_time ---
    
    # 转换为 datetime 对象，并计算 prediction_time
    final_df['charttime_dt'] = pd.to_datetime(final_df[timestamp_col], format="%Y-%m-%d %H:%M:%S", errors='coerce')
    # 使用 .dt.strftime 进行向量化时间格式化
    final_df['prediction_time'] = (final_df['charttime_dt'] - timedelta(minutes=1)).dt.strftime("%Y-%m-%d %H:%M:%S")

    # --- 7. 分组构建最终结果列表 ---
    
    meta_datas = []

    # 按 (subject_id, charttime) 分组，构建最终的 JSON 结构
    # 使用 subject_id 和原始时间戳列进行分组
    group_keys = ['subject_id', timestamp_col]
    
    # 遍历每个样本组
    for (subject_id, charttime), group in final_df.groupby(group_keys):
        
        # 确保 charttime 有效
        if pd.isna(charttime) or not isinstance(charttime, str):
            continue

        labels = []
        # 遍历组内的所有标签信息
        for _, row in group.iterrows():
            # 确保标签信息查找成功 (即 'label' 列不为空)
            if pd.notna(row['label']): 
                 labels.append({
                    "name": row["label"],
                    "itemid": int(row["itemid"]),
                    "fluid": row["fluid"],
                    "category": row["category"],
                 })
        
        # 提取第一个 prediction_time (组内所有行的 prediction_time 应该相同)
        # 如果组是空的或prediction_time无效，则跳过
        if group.empty or pd.isna(group['prediction_time'].iloc[0]):
             continue
             
        prediction_time = group['prediction_time'].iloc[0]

        meta_data = {
            "subject_id": int(subject_id),
            "prediction_time": prediction_time,
            "task": "labevents",
            "label": labels
        }
        meta_datas.append(meta_data)
        
    return total_samples, meta_datas

# def generate_labevents_task(args):
#     task_file = os.path.join(args.data_path, "hosp", "labevents.csv")
#     label_mapping_file = os.path.join(args.data_path, "hosp", "d_labitems.csv")

#     task_df = pd.read_csv(task_file)
#     label_mapping_df = pd.read_csv(label_mapping_file)

#     # filter by subject_id or subject_id_file
#     if args.subject_id_file is not None:
#         with open(args.subject_id_file, 'r') as f:
#             subject_ids = [int(line.strip()) for line in f if line.strip().isdigit()]
#         task_df = task_df[task_df["subject_id"].isin(subject_ids)]
#     elif args.subject_id is not None:
#         task_df = task_df[task_df["subject_id"] == int(args.subject_id)]

#     cols = task_df.columns.tolist()
#     timestamp_col = next((col for col in cols if 'time' in col.lower() or 'date' in col.lower()), None)
#     sample_pairs_list = choose_data_by_time(task_df, timestamp_col, time_point=args.chartdate, num=args.sample_num)

#     meta_datas = []
#     for pair in tqdm(sample_pairs_list):
#         charttime = pair[timestamp_col]
#         subject_id = pair["subject_id"]

#         # 检查 charttime 是否为有效字符串，跳过 NaN 值
#         if pd.isna(charttime) or not isinstance(charttime, str):
#             continue

#         prediction_time = (datetime.strptime(charttime, "%Y-%m-%d %H:%M:%S") - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
#         task_info = task_df[(task_df["subject_id"] == subject_id) & (task_df[timestamp_col] == charttime)].to_dict(orient="records")
#         meta_data = {
#             "subject_id": subject_id,
#             "prediction_time": prediction_time,
#             "task": "labevents",
#             "label": [
#                 {
#                     "name": label_mapping_df[(label_mapping_df["itemid"] == label_info["itemid"])]["label"].tolist()[0],
#                     "itemid": label_info["itemid"],
#                     "fluid": label_mapping_df[(label_mapping_df["itemid"] == label_info["itemid"])]["fluid"].tolist()[0],
#                     "category": label_mapping_df[(label_mapping_df["itemid"] == label_info["itemid"])]["category"].tolist()[0],
#                 } for label_info in task_info
#             ]
#         }
#         meta_datas.append(meta_data)

    # return len(sample_pairs_list), meta_datas


def generate_prescriptions_task(args):
    task_file = os.path.join(args.data_path, "hosp", "prescriptions.csv")
    label_mapping_file = os.path.join(args.data_path, "hosp", "d_atc_prescriptions.csv")

    task_df = pd.read_csv(task_file)
    label_mapping_df = pd.read_csv(label_mapping_file)

    # filter by subject_id or subject_id_file
    if args.subject_id_file is not None:
        with open(args.subject_id_file, 'r') as f:
            subject_ids = [int(line.strip()) for line in f if line.strip().isdigit()]
        task_df = task_df[task_df["subject_id"].isin(subject_ids)]
    elif args.subject_id is not None:
        task_df = task_df[task_df["subject_id"] == int(args.subject_id)]

    # 将task_df["ndc"]这一列的每个值都进行如下处理：去掉小数点后面的部分，然后左侧补0到11位
    if "ndc" in task_df.columns:
        task_df["ndc"] = task_df["ndc"].astype(str).str.split(".", n=1).str[0].str.zfill(11)

    cols = task_df.columns.tolist()
    timestamp_col = next((col for col in cols if 'time' in col.lower() or 'date' in col.lower()), None)
    sample_pairs_list = choose_data_by_time(task_df, timestamp_col, time_point=args.chartdate, num=args.sample_num)

    meta_datas = []
    for pair in tqdm(sample_pairs_list):
        charttime = pair[timestamp_col]
        subject_id = pair["subject_id"]

        # 检查 charttime 是否为有效字符串，跳过 NaN 值
        if pd.isna(charttime) or not isinstance(charttime, str):
            continue

        prediction_time = (datetime.strptime(charttime, "%Y-%m-%d %H:%M:%S") - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
        task_info = task_df[(task_df["subject_id"] == subject_id) & (task_df[timestamp_col] == charttime)].to_dict(orient="records")
        
        meta_data = {
            "subject_id": subject_id,
            "prediction_time": prediction_time,
            "task": "prescriptions",
            "label": [
                {
                    "name": label_mapping_df[(label_mapping_df["ndc_code"] == label_info["ndc"])]["atc_name"].tolist()[0] if not label_mapping_df[(label_mapping_df["ndc_code"] == label_info["ndc"])]["atc_name"].empty else None,
                    # "long_drug_name": label_mapping_df[(label_mapping_df["ndc_code"] == label_info["ndc"])]["long_drug_name"].tolist()[0] if not label_mapping_df[(label_mapping_df["ndc_code"] == label_info["ndc"])]["long_drug_name"].empty else None,
                    "ndc_code": label_info["ndc"],
                } for label_info in task_info
            ]
        }
        meta_datas.append(meta_data)

    return len(sample_pairs_list), meta_datas


def generate_microbiologyevents_task(args):
    task_file = os.path.join(args.data_path, "hosp", "microbiologyevents.csv")
    
    task_df = pd.read_csv(task_file)

    # filter by subject_id or subject_id_file
    if args.subject_id_file is not None:
        with open(args.subject_id_file, 'r') as f:
            subject_ids = [int(line.strip()) for line in f if line.strip().isdigit()]
        task_df = task_df[task_df["subject_id"].isin(subject_ids)]
    elif args.subject_id is not None:
        task_df = task_df[task_df["subject_id"] == int(args.subject_id)]
    
    cols = task_df.columns.tolist()
    timestamp_col = next((col for col in cols if 'time' in col.lower() or 'date' in col.lower()), None)
    sample_pairs_list = choose_data_by_time(task_df, timestamp_col, time_point=args.chartdate, num=args.sample_num)

    meta_datas = []
    for pair in sample_pairs_list:
        charttime = pair[timestamp_col]
        subject_id = pair["subject_id"]

        # 检查 charttime 是否为有效字符串，跳过 NaN 值
        if pd.isna(charttime) or not isinstance(charttime, str):
            continue

        prediction_time = (datetime.strptime(charttime, "%Y-%m-%d %H:%M:%S") - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
        task_info = task_df[(task_df["subject_id"] == subject_id) & (task_df[timestamp_col] == charttime)].to_dict(orient="records")
        meta_data = {
            "subject_id": subject_id,
            "prediction_time": prediction_time,
            "task": "microbiologyevents",
            "label": [
                {
                    "name": label_info["test_name"],
                } for label_info in task_info
            ]
        }
        meta_datas.append(meta_data)

    return len(sample_pairs_list), meta_datas


def generate_radiology_task(args):
    task_file = os.path.join(args.data_path, "note", "radiology.csv")
    label_mapping_file = os.path.join(args.data_path, "note", "radiology_detail.csv")

    task_df = pd.read_csv(task_file)
    label_mapping_df = pd.read_csv(label_mapping_file)

    # filter by subject_id or subject_id_file
    if args.subject_id_file is not None:
        with open(args.subject_id_file, 'r') as f:
            subject_ids = [int(line.strip()) for line in f if line.strip().isdigit()]
        task_df = task_df[task_df["subject_id"].isin(subject_ids)]
    elif args.subject_id is not None:
        task_df = task_df[task_df["subject_id"] == int(args.subject_id)]
    
    cols = task_df.columns.tolist()
    timestamp_col = next((col for col in cols if 'time' in col.lower() or 'date' in col.lower()), None)
    sample_pairs_list = choose_data_by_time(task_df, timestamp_col, time_point=args.chartdate, num=args.sample_num)

    meta_datas = []
    for pair in sample_pairs_list:
        charttime = pair[timestamp_col]
        subject_id = pair["subject_id"]

        # 检查 charttime 是否为有效字符串，跳过 NaN 值
        if pd.isna(charttime) or not isinstance(charttime, str):
            continue

        prediction_time = (datetime.strptime(charttime, "%Y-%m-%d %H:%M:%S") - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
        task_info = task_df[(task_df["subject_id"] == subject_id) & (task_df[timestamp_col] == charttime)].to_dict(orient="records")
        meta_data = {
            "subject_id": subject_id,
            "prediction_time": prediction_time,
            "task": "radiology",
            "label": [
                {
                    "name": label_mapping_df[(label_mapping_df["note_id"] == label_info["note_id"]) & (label_mapping_df["field_name"] == "exam_name")]["field_value"].tolist()[0],
                    "note_id": label_info["note_id"],
                } for label_info in task_info
            ]
        }
        meta_datas.append(meta_data)

    return len(sample_pairs_list), meta_datas


def generate_transfers_task(args):
    task_file = os.path.join(args.data_path, "hosp", "transfers.csv")

    task_df = pd.read_csv(task_file)
    task_df = task_df[task_df["eventtype"] == "transfer"]

    # filter by subject_id or subject_id_file
    if args.subject_id_file is not None:
        with open(args.subject_id_file, 'r') as f:
            subject_ids = [int(line.strip()) for line in f if line.strip().isdigit()]
        task_df = task_df[task_df["subject_id"].isin(subject_ids)]
    elif args.subject_id is not None:
        task_df = task_df[task_df["subject_id"] == int(args.subject_id)]
    
    cols = task_df.columns.tolist()
    timestamp_col = next((col for col in cols if 'time' in col.lower() or 'date' in col.lower()), None)
    sample_pairs_list = choose_data_by_time(task_df, timestamp_col, time_point=args.chartdate, num=args.sample_num)

    # fileter data
    timestamp_list = [sample[timestamp_col] for sample in sample_pairs_list if pd.isna(sample[timestamp_col]) or not isinstance(sample[timestamp_col], str)]
    task_df = task_df[task_df[timestamp_col].isin(timestamp_list)]

    meta_datas = []
    for pair in sample_pairs_list:
        charttime = pair[timestamp_col]
        subject_id = pair["subject_id"]

        # 检查 charttime 是否为有效字符串，跳过 NaN 值
        if pd.isna(charttime) or not isinstance(charttime, str):
            continue

        prediction_time = (datetime.strptime(charttime, "%Y-%m-%d %H:%M:%S") - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
        task_info = task_df[(task_df["subject_id"] == subject_id) & (task_df[timestamp_col] == charttime)].to_dict(orient="records")
        meta_data = {
            "subject_id": subject_id,
            "prediction_time": prediction_time,
            "task": "transfers",
            "label": [
                {
                    "name": label_info["careunit"],
                } for label_info in task_info
            ]
        }
        meta_datas.append(meta_data)
        task_df = task_df[~((task_df["subject_id"] == subject_id) & (task_df[timestamp_col] == charttime))]

    return len(sample_pairs_list), meta_datas


def get_task_function(task_name):
    # add more tasks here
    task_functions = {
        "diagnoses_icd": generate_diagnose_icd_task,
        "diagnoses_ccs": generate_diagnose_ccs_task,
        "procedures_icd": generate_procedures_icd_task,
        "procedures_ccs": generate_procedures_ccs_task,
        "labevents": generate_labevents_task_vectorized,
        "prescriptions": generate_prescriptions_task,
        "microbiologyevents": generate_microbiologyevents_task,
        "radiology": generate_radiology_task,
        "transfers": generate_transfers_task,
    }
    return task_functions.get(task_name)


def parse_args():
    parser = argparse.ArgumentParser(prog="Task Meta Data Generation")

    parser.add_argument("--data_path", type=str, default="/sfs/rhome/liaoyusheng/data/Datasets/EHRAgent/MIMICIV-2.2")
    parser.add_argument("--task", type=str, default="transfers")
    parser.add_argument("--sample_num", type=int, default=100)
    parser.add_argument("--subject_id", type=str, default=None)
    parser.add_argument("--subject_id_file", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="/remote-home/chuanxuan/ehragent/EHRAgentBench")
    parser.add_argument("--chartdate", type=str, default=None)

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    return args
    
if __name__ == '__main__':
    args = parse_args()

    task_func = get_task_function(args.task)
    if task_func is None:
        print(f"Task {args.task} not supported")
        exit(1)
        
    num, meta_datas = task_func(args)

    if num != args.sample_num:
        num = "all"

    with open(os.path.join(args.output_path, f"{args.task}_{num}.json"), "w") as f:
        json.dump(meta_datas, f, indent=4, ensure_ascii=False)
