export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=.
torchrun --nproc_per_node=8 --master_port=20013 fastchat/train/train_mem.py \
    --model_name_or_path ../falcon-1b  \
    --data_path ../chatfine/total_filtered.json \
    --bf16 True \
    --output_dir output_falcon \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --resume_from_checkpoint True\
    --gradient_checkpointing True \
    --deepspeed "./default_offload_opt_param.json" \
    --model_max_length 2048 \
    --lazy_preprocess False > run_falcon.out 2> run_falcon.err & 
    # --model_max_length 2048 \
    #--tf32 True \
