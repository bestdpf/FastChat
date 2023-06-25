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
    return '\n'.join(contents)


def run_model():
    model_path = "./output_flant5_base_full"

    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 trust_remote_code=True,
                                                 device_map="auto",
                                                 )

    while True:
        input_text = user_input()
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids

        outputs = model.generate(input_ids)
        print(tokenizer.decode(outputs[0]))


if __name__ == '__main__':
    run_model()

