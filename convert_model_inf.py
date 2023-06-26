import torch
import os

from fastchat.model import compression
from transformers_neuronx.llama.model import LlamaForCausalLM, LlamaForSampling
from transformers_neuronx.module import save_pretrained_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers_neuronx.config import NeuronConfig, QuantizationConfig

os.environ['NEURON_CC_FLAGS'] = '--model-type=transformer-inference'


def convert_to_inf(model_path, inf_path):
    # model_cpu = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu",
    #                                                  torch_dtype=torch.bfloat16)

    # Set the weight storage config use int8 quantization and bf16 dequantization

    neuron_config = NeuronConfig(
        quant=QuantizationConfig(quant_dtype='s8', dequant_dtype='bf16'),
    )

    model_neuron = LlamaForSampling.from_pretrained(model_path, batch_size=1, tp_degree=2, n_positions=2048, amp='bf16',
                                                    neuron_config=neuron_config,
                                                    load_in_8bit=True,
                                                    unroll=None)
    model_neuron.to_neuron()
    model_neuron.save_pretrained(inf_path)


def compress_model(model_path, out_path):
    model_cpu = compression.load_compress_model(model_path, "cpu", torch_dtype=torch.bfloat16)
    model_cpu.save_pretrained(out_path)


def compress_model2(model_path, out_path):
    model_cpu = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu",
                                                     torch_dtype=torch.bfloat16, load_in_8bit=True)
    model_cpu.save_pretrained(out_path)


if __name__ == '__main__':
    compress_model('./fastchat-vicuna-wizardbf16', './fastchat-vicuna-wizardint8')
    # convert_to_inf('./fastchat-vicuna-wizardbf16', './fastchat-vicuna-inf7b')
