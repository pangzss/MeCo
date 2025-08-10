#!/bin/bash

set -e

stage1_path="work_dirs/meco-stage-1-minicpmv_qwen2"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="./:$PYTHONPATH"

torchrun --nproc_per_node 4 meco/train/train.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path checkpoints/MiniCPM-V-2_6_QWen2 \
    --language_model qwen2 \
    --conv_type plain \
    --fast_tokenizer True \
    --vision_tower eva_vit \
    --vision_processor clip_center_224 \
    --vision_output_layer -2 \
    --vision_output_token patch \
    --mm_projector qformer \
    --pretrain_vision_tower checkpoints/eva_vit_g.pth \
    --pretrain_qformer checkpoints/timechat/instruct_blip_vicuna7b_trimmed.pth \
    --anno_path data/llava_558k_with_webvid.json \
    --image_path data/llava_pretrain/images \
    --video_path data/webvid/videos \
    --fps 1 \
    --tuning_mode projector \
    --use_matching False \
    --use_time_tag False \
    --bi_attention False \
    --alpha 2.0 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --output_dir $stage1_path \
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
    --bf16 true \
    --fp16 False \
    --report_to wandb