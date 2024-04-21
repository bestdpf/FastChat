export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export CUDA_VISIBLE_DEVICES=1,3,4,5
export HF_DATASETS_IN_MEMORY_MAX_SIZE=20000000000
export HF_DATASETS_OFFLINE=0
export PYTHONPATH=.
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"

python -m torch.distributed.run --nproc_per_node=1 --master_port=20011 fastchat/train/train_mem.py \
    --model_name_or_path ../Meta-Llama-3-8B  \
    --data_path ../chatfine/total_gpt_full_merge_b16_20240420 \
    --bf16 True \
    --tf32 True \
    --output_dir fastchat-vicuna-3-8b-20240420 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap offload" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True >run_wizard.out 2>run_wizard.err&

