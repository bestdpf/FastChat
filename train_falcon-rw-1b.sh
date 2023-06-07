export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.:/home/fangbing/minillm/falcon-rw-1b
torchrun --nproc_per_node=1 --master_port=20013 fastchat/train/train_mem.py \
    --model_name_or_path /home/fangbing/minillm/falcon-rw-1b  \
    --data_path /home/fangbing/minillm/chatfine/total_filtered.json \
    --bf16 True \
    --output_dir output_vicuna \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --resume_from_checkpoint True\
    --lazy_preprocess True > run_falcon.out 2> run_falcon.err & 
    # --model_max_length 2048 \
    # --gradient_checkpointing True \
    #--tf32 True \
