
SAMPLE_NUM=1000
SAMPLE_MODE="common"

python /sfs/rhome/liaoyusheng/projects/EHRAgent/EHRAgent/data_preprocess/data_processing/label_wise_sample.py \
    --input_path "/sfs/rhome/liaoyusheng/data/Datasets/EHRAgent/EHRAgentBench/all" \
    --output_path "/sfs/rhome/liaoyusheng/data/Datasets/EHRAgent/EHRAgentBench/${SAMPLE_MODE}" \
    --sample_mode ${SAMPLE_MODE} \
    --sample_num ${SAMPLE_NUM} \
    --suffix "v2" \
    --resume
