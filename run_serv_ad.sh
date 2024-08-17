#!/bin/bash
export PYHONPATH=.
export DS_ACCELERATOR="cuda"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
# python3 -m fastchat.serve.controller --host 127.0.0.1 --port 21002 >run_serv_controller.out 2> run_serv_controller.err &
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.model_worker --host 127.0.0.1 --port 21004 --controller-address 'http://127.0.0.1:21002' --worker-address 'http://127.0.0.1:21004' --model-name 'vicuna-3-8b' --model-path /root/autodl-tmp/fastchat-vicuna-31-8b-20240728/ --device cuda --limit-model-concurrency 4 > run_serv_worker.out 2> run_serv_worker.err &
# python3 -m fastchat.serve.openai_api_server --host 127.0.0.1 --port 9998 --controller-address 'http://127.0.0.1:21002' > run_serv_api.out 2> run_serv_api.err&
