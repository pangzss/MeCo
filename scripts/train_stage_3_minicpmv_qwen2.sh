#!/bin/bash

set -e

stage2_path=${1:-"work_dirs/meco-stage-2-minicpmv_qwen2"}
stage3_path="work_dirs/meco-stage-3-minicpmv_qwen2"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="./:$PYTHONPATH"

python -m torch.distributed.run --nproc_per_node 4 meco/train/train.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path $stage2_path \
    --language_model qwen2 \
    --conv_type qwen2 \
    --fast_tokenizer True \
    --vision_tower eva_vit \
    --vision_processor clip_center_224 \
    --vision_output_layer -2 \
    --vision_output_token patch \
    --mm_projector qformer \
    --fps 1 \
    --lora_enable True \
    --lora_lr 5e-5 \
    --tuning_mode attention \
    --use_matching True \
    --use_time_tag False \
    --bi_attention True \
    --alpha 2.0 \
    --min_video_len 5 \
    --max_video_len 350 \
    --num_train_epochs 1 \
    --output_dir $stage3_path \
    --save_full_model True \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --bf16 True \
    --fp16 False \
    --anno_path /home1/pangzs/workdir/data/video_datasets/ET-Instruct-164K/et_instruct_164k_meco.json \
    --video_path /home1/pangzs/workdir/data/video_datasets/ET-Instruct-164K/videos \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --report_to wandb \
    --use_wandb True \
    --localization True \
    --max_num_words 500