export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.
torchrun --nproc_per_node=1 --master_port=20012 fastchat/train/train_mem.py \
    --model_name_or_path /home/fangbing/minillm/Wizard-Vicuna-7B-Uncensored  \
    --data_path /home/fangbing/minillm/chatfine/total_filtered.json \
    --bf16 True \
    --output_dir output_wizard_deep \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --deepspeed "./default_offload_opt_param.json" \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess False >run_wizard.out 2>run_wizard.err&

