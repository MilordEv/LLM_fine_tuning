import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, GPT2LMHeadModel, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import evaluate
import bitsandbytes

from language_gan import LanguageGAN
from test import calculate_metric


dataset_tofu = load_dataset("locuslab/TOFU", "full")
print(dataset_tofu)

train_tofu = Dataset.from_dict(dataset_tofu['train'][:50])
test_tofu = Dataset.from_dict(dataset_tofu['train'][100:150])

device = torch.device('cuda:0')
model_id = 'openai-community/gpt2'

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)

config = GenerationConfig.from_pretrained(model_id)
config.do_sample = True
config.output_hidden_states=True
config.temperature = 0.7
config.top_p = 0.7
config.penalty_alpha = 0.8
config.max_new_tokens = 100
config.pad_token_id = tokenizer.eos_token_id    

config_lora = LoraConfig(r=8,
                         lora_alpha=32,
                         target_modules=["c_attn", "c_proj"],
                         lora_dropout=0.05,
                         bias="none",
                         task_type="CAUSAL_LM")

model = get_peft_model(model, config_lora)

def generate_prompt(data_point):
    return f"""<human>: {data_point["question"]}
    <assistant>: {data_point["answer"]}
    """.strip()

def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenizer(full_prompt, padding=True, truncation=True)
    return tokenized_full_prompt

train_data = train_tofu.map(generate_and_tokenize_prompt)
train_data.set_format('torch')

training_args = TrainingArguments(per_device_train_batch_size=1,
                                  gradient_accumulation_steps=8,
                                  num_train_epochs=29,
                                  learning_rate=2e-4,
                                  fp16=True,
                                  save_total_limit=3,
                                  logging_steps=1,
                                  output_dir="experiments",
                                  optim="paged_adamw_8bit",
                                  lr_scheduler_type="cosine",
                                  warmup_ratio=0.05
)

trainer = Trainer(model=model,
                  train_dataset=train_data,
                  args=training_args,
                  data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()

results = calculate_metric(model, test_tofu, tokenizer, config, max_length_input=100, max_new_tokens=50, device=device)
print(f'{"for LoRA":=^20}:')
print('ROUGE:')
print(results[0])
print()
print('BLEURT:')
print(sum(results[1]['scores']) / len(results[1]['scores']))