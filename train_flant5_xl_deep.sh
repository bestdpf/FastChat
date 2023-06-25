export TF_ENABLE_ONEDNN_OPTS=0
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=.
export DS_ACCELERATOR="cuda"
torchrun --nproc_per_node=8 --master_port=9788 fastchat/train/train_flant5.py \
    --model_name_or_path ../flan-t5-xl \
    --data_path ../chatfine/total_filtered.json \
    --output_dir ./output_flant5_xl \
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
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --bf16 True \
    --deepspeed "./ds_flan_t5_z3_config_bf16.json" \
    --preprocessed_path ./preprocessed_data/pre-xl.json \
    --lazy_preprocess True > run_flant5_xl.out 2> run_flant5_xl.err &

#    --gradient_checkpointing True
    #    --tf32 True \
    # --deepspeed "./ds_flan_t5_z3_config_bf16.json" \
