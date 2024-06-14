import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import evaluate
import bitsandbytes

from language_gan import LanguageGAN
from test import calculate_metric, test_question_answering, visualize_weights, get_memory_history


dataset_tofu = load_dataset("locuslab/TOFU", "full")
print(dataset_tofu)

train_tofu = Dataset.from_dict(dataset_tofu['train'][:50])
test_tofu = Dataset.from_dict(dataset_tofu['train'][100:150])


device = torch.device('cuda:0')
# device = torch.device('cpu')
# model_id = "TheBloke/Llama-2-7b-Chat-GPTQ"
model_id = 'openai-community/gpt2'

tokenizer = AutoTokenizer.from_pretrained(model_id)
# tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)
print(model)

config = GenerationConfig.from_pretrained(model_id)
# config.do_sample = True
# config.output_hidden_states=True
# config.temperature = 0.7
# config.top_p = 0.7
# config.penalty_alpha = 0.8
config.max_new_tokens = 100
config.pad_token_id = tokenizer.eos_token_id

results = calculate_metric(model, test_tofu, tokenizer, config, max_length_input=100, max_new_tokens=50, device=device)
print(f'{"for base model":=^20}:')
print('ROUGE:')
print(results[0])
print()
print('BLEURT:')
print(sum(results[1]['scores']) / len(results[1]['scores']))

def change_layer(embedding_layer, new_value):
    # model.model.embed_tokens = new_value
    model.transformer.wte = new_value

gan_model = LanguageGAN(model, model.transformer.wte, change_layer, 100, 50, device)

results = gan_model.train(dataset=train_tofu, tokenizer=tokenizer,
                          epochs=1, generation_config=config, batch_size=1, save_model=False, save_weights=False)

test_question_answering(gan_model, tokenizer, config, test_tofu[0]['question'], device)
print()

results = calculate_metric(gan_model, test_tofu, tokenizer, config, max_length_input=100, max_new_tokens=50, device=device)
print(f'{"for tuned model":=^20}:')
print('ROUGE:')
print(results[0])
print()
print('BLEURT:')
print(sum(results[1]['scores']) / len(results[1]['scores']))

visualize_weights(results[0], "frozen weights")
visualize_weights(results[1], "tunable weights")

get_memory_history(gan_model, train_tofu, tokenizer, config, 'cpu_history')