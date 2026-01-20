#!/bin/bash

# single task file
DIR="./data/MIMICIVAgentBench/common/prescriptions_500.json"
OUTPUT_PATH="${YOUR_OUTPUT_PATH}"
python ./data_preprocess/patient_event2db.py \
        --root_path "${PATH_TO_MIMICIV}" \
        --output_path "${OUTPUT_PATH}" \
        --data_file_path "${DIR}"


# multiple task files within a subset
DIR="./data/MIMICIVAgentBench/common"
OUTPUT_PATH="${YOUR_OUTPUT_PATH}"
for file in "$DIR"/*.json; do
    [ -e "$file" ] || continue

    echo "正在处理: $file"
    python ./data_preprocess/patient_event2db.py \
        --root_path "${PATH_TO_MIMICIV}" \
        --output_path "${OUTPUT_PATH}" \
        --data_file_path "${file}"
    
done