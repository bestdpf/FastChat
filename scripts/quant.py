# Modified from https://github.com/soulteary/docker-llama2-chat/blob/main/llama2-7b-cn-4bit/quantization_4bit.py

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def convert_int4(in_dir, out_dir):
    model = AutoModelForCausalLM.from_pretrained(
        in_dir,
        torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        ),
        device_map='cuda'
    )

    model.save_pretrained(out_dir)
    print(f"convert {in_dir} to {out_dir} done")