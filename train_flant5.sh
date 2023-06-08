export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.
torchrun --nproc_per_node=1 --master_port=9778 fastchat/train/train_flant5.py \
    --model_name_or_path /home/fangbing/minillm/flan-t5-base  \
    --data_path /home/fangbing/minillm/chatfine/total_filtered.json \
    --bf16 True \
    --output_dir ./checkpoints_flant5_base \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap T5Block \
    --model_max_length 2662 \
    --preprocessed_path ./preprocessed_data/processed.json \
    --lazy_preprocess False > run_fant5.out 2> run_fant5.err &

#    --gradient_checkpointing True
    #    --tf32 True \
