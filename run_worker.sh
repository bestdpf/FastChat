#!/bin/bash
export PYHONPATH=.
python3 -m fastchat.serve.controller > run_serv_controller.out 2 > run_serv_controller.err &
python3 -m fastchat.serve.model_worker --model-name 'vicuna-wizard-7b' --model-path ./fastchat-vicuna-wizardbf16 --device cuda > run_serv_worker.out 2> run_serv_worker.err &
python3 -m fastchat.serve.openai_api_server --host 0.0.0.0 --port 9999 > run_serv_api.out 2> run_serv_api.err&
