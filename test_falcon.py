from transformers import AutoTokenizer, AutoModelForCausalLM

import transformers
import torch
from transformers_stream_generator import init_stream_support


def run_model():
    init_stream_support()
    model_path = "./output_falcon"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 trust_remote_code=True,
                                                 )
    model = model.eval()
    while True:
        with torch.no_grad():
            prompt = input('Input your text:\n')
            print(f'User: {prompt}')
            input_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids
            generator = model.generate(
                input_ids,
                max_new_tokens=180,
                do_sample=True,
                top_k=10,
                top_p=0.85,
                temperature=0.35,
                repetition_penalty=1.2,
                early_stopping=True,
                seed=0,
                do_stream=True,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
            )
            stream_result = ""
            words = ""
            last_tokens = []
            last_decoded_tokens = []

            for index, x in enumerate(generator):
                tokens = x.cpu().numpy().tolist()
                tokens = last_tokens + tokens
                word = tokenizer.decode(tokens, skip_special_tokens=True)
                if True:
                    if " " in tokenizer.decode(
                            last_decoded_tokens + tokens, skip_special_tokens=True
                    ):
                        word = " " + word
                    last_tokens = []
                    last_decoded_tokens = tokens
                stream_result += word
            print(f'Assistant: {stream_result}')


if __name__ == '__main__':
    run_model()

