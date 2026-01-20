#!/bin/bash

# 指定包含 json 文件的目录
DIR="/sfs/rhome/liaoyusheng/data/Datasets/EHRAgent/EHRAgentBench/common/prescriptions_1000v2.json"

python /sfs/rhome/liaoyusheng/projects/EHRAgent/EHRAgent/data_preprocess/sql_preprocess/patient_event2db.py \
        --root_path "/sfs/rhome/liaoyusheng/data/Datasets/EHRAgent/MIMICIV-2.2" \
        --output_path "/sfs/rhome/liaoyusheng/data/Datasets/EHRAgent/EHRAgentBench/database" \
        --data_file_path "${DIR}"

# 遍历目录下的所有 .json 文件
# for file in "$DIR"/*.json; do
#     # 检查文件是否存在（防止没有匹配到文件时报错）
#     [ -e "$file" ] || continue

#     echo "正在处理: $file"
    
#     # --- 在这里执行你的命令 ---
#     # "$file" 包含了完整的路径
#     python /sfs/rhome/liaoyusheng/projects/EHRAgent/EHRAgent/data_preprocess/sql_preprocess/patient_event2db.py \
#         --root_path "/sfs/rhome/liaoyusheng/data/Datasets/EHRAgent/MIMICIV-2.2" \
#         --output_path "/sfs/rhome/liaoyusheng/data/Datasets/EHRAgent/EHRAgentBench/database" \
#         --data_file_path "${file}"
    
#     # 示例：如果是 python 脚本
#     # python process_data.py --input "$file"
# done