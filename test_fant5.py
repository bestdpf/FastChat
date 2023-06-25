from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration

import transformers
import torch


def user_input():
    print("Enter/Paste your content. Ctrl-D or Ctrl-Z ( windows ) to save it.")
    contents = []
    while True:
        line = input()
        if line == 'break':
            break
        contents.append(line)
    ret = '\n'.join(contents)
    print(f'your input is :\n{ret}')
    return ret


def run_model():
    model_path = "./output_flant5_base_full"
    model_path = "output_flant5_base/checkpoint-11700"

    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path, 
            device_map="auto")

    while True:
        input_text = user_input()
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
        outputs = model.generate(input_ids,
                        max_new_tokens=150,
                )
        print('the output:\n')
        for output in outputs:
            print(output)
            print(tokenizer.decode(output))


if __name__ == '__main__':
    run_model()

