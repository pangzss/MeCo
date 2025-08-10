#!/bin/bash
set -e

# anno_path=${1:-"data/QVhighlights/annotations/r2-tuning/qvhighlights_val.jsonl"}
# data_path=${2:-"data/QVhighlights/videos"}
# stage3_path=${3:-"work_dirs/meco-stage-3-phi3"}
# pred_path=${4:-"qvhighlights"}

anno_path=${1:-"data/Charades/annotations/r2-tuning/charades_test.jsonl"}
data_path=${2:-"data/Charades/videos"}
stage3_path=${3:-"work_dirs/meco-stage-3-phi3"}
pred_path=${4:-"charades"}
dtype=${5:-"fp16"} # Change to 'bf16' if meco-stage-3-minicpmv_qwen2 is used
export PYTHONPATH="./:$PYTHONPATH"

python meco/eval/infer_charades_qvh.py \
    --anno_path $anno_path \
    --data_path $data_path \
    --pred_path $stage3_path/$pred_path \
    --model_path $stage3_path \
    --dtype $dtype \
    --verbose