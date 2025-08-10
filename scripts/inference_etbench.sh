#!/bin/bash
set -e

stage3_path=${1:-"work_dirs/meco-stage-3-phi3"}
task=${2:-"all"}
pred_dir=${3:-"etbench"}
dtype=${4:-"fp16"} # Change to 'bf16' if meco-stage-3-minicpmv_qwen2 is used
export PYTHONPATH="./:$PYTHONPATH"

python meco/eval/infer_etbench.py \
    --anno_path data/ETBench/annotations/vid \
    --data_path data/ETBench/videos_compressed \
    --pred_path $stage3_path/$pred_dir \
    --model_path $stage3_path \
    --task $task \
    --dtype $dtype \
    --verbose \
    "${@:5}"