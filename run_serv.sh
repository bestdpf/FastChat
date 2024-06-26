#!/bin/bash
export PYHONPATH=.
export DS_ACCELERATOR="cuda"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
python3 -m fastchat.serve.controller --host localhost --port 21001 >run_serv_controller.out 2> run_serv_controller.err &
CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.model_worker --port 21003 --worker-address 'http://localhost:21003' --model-name 'vicuna-2-7b' --model-path ../fastchat-vicuna-20230819 --device cuda --limit-model-concurrency 10 > run_serv_worker.out 2> run_serv_worker.err &
python3 -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 9998 > run_serv_api.out 2> run_serv_api.err&
