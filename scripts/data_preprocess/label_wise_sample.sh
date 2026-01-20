
SAMPLE_NUM=1000
SAMPLE_MODE="common"

python ./data_preprocess/data_processing/label_wise_sample.py \
    --input_path "./data/MIMICIVAgentBench/all" \
    --output_path "./data/MIMICIVAgentBench/${SAMPLE_MODE}" \
    --sample_mode ${SAMPLE_MODE} \
    --sample_num ${SAMPLE_NUM} \
    --suffix "" \
    --resume
