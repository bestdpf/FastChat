from transformers import AutoTokenizer, AutoModelForCausalLM

import transformers
import torch


def run_model():
    model_path = "./output_falcon"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(tokenizer.eos_token_id)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_path,
        tokenizer=tokenizer,
        # torch_dtype=torch.float16,
        trust_remote_code=True,
        # use_ipex=True,
        # device_map="cpu",
    )
    while True:
        prompt = input('input your prompt:\n')
        sequences = pipeline(
            prompt,
            max_length=200,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )
        for seq in sequences:
            print(f"Result: {seq['generated_text']}")


if __name__ == '__main__':
    run_model()

