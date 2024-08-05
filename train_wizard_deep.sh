#!/bin/bash


# export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=.
export OMP_NUM_THREADS=10
export DS_ACCELERATOR="cuda"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
#export WANDB_DISABLED="true"
export HF_DATASETS_IN_MEMORY_MAX_SIZE=10000000000
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export TRITON_CACHE_DIR=./triton_cache/ 
CUDA_VISIBLE_DEVICES=0 nohup python -m torch.distributed.run --nproc_per_node=1 --master_port=20012 fastchat/train/train_mem.py \
    --model_name_or_path ../Meta-Llama-3.1-8B  \
    --data_path ../chatfine/total_gpt_full_merge_b16_20240729.json\
    --bf16 True \
    --tf32 True \
    --output_dir fastchat-vicuna-3-8b-20240728 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 300 \
    --eval_steps 300 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --deepspeed "default_offload_opt_param.json" \
    --report_to "wandb" \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess False >run_wizard.out 2>run_wizard.err&

    #--deepspeed "./ds_flan_t5_z3_config_bf16.json" \
