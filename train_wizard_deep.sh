export CUDA_VISIBLE_DEVICES=1,3,4,5
export PYTHONPATH=.
export OMP_NUM_THREADS=10
export DS_ACCELERATOR="cuda"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
#export WANDB_DISABLED="true"
export HF_DATASETS_IN_MEMORY_MAX_SIZE=20000000000
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
torchrun --nproc_per_node=4 --master_port=20012 fastchat/train/train_mem.py \
    --model_name_or_path ../Meta-Llama-3-8B  \
    --data_path ../chatfine/total_gpt_full_merge_b16_20240420 \
    --bf16 True \
    --tf32 True \
    --output_dir fastchat-vicuna-3-8b-20240420 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 40 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "wandb" \
    --model_max_length 8192 \
    --deepspeed "default_offload_opt_param.json" \
    --gradient_checkpointing True \
    --lazy_preprocess True >run_wizard.out 2>run_wizard.err&

    #--deepspeed "./ds_flan_t5_z3_config_bf16.json" \
