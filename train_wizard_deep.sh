export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export PYTHONPATH=.
export OMP_NUM_THREADS=1
export DS_ACCELERATOR="cuda"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export WANDB_DISABLED="true"
torchrun --nproc_per_node=6 --master_port=20012 fastchat/train/train_mem.py \
    --model_name_or_path ../llama-2-7b  \
    --data_path ../chatfine/total_filtered_20230720.json \
    --tf32 True \
    --bf16 True \
    --output_dir output_wizard_deep_0720 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
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
    --model_max_length 4096 \
    --deepspeed "default_offload_opt_param.json" \
    --gradient_checkpointing True \
    --lazy_preprocess True >run_wizard.out 2>run_wizard.err&

    #--deepspeed "./ds_flan_t5_z3_config_bf16.json" \
