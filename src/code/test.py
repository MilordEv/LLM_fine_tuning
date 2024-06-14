import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, GPT2LMHeadModel, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import evaluate

from language_gan import tokenize_func

def calculate_metric(model, dataset, tokenizer, generation_config, max_length_input = 100, max_new_tokens = 100, device=torch.device('cpu')):
    data_size = len(dataset)
    real_answer = dataset['answer']

    dataset = tokenize_func(tokenizer, dataset, max_length_input, max_new_tokens, True)
    dataset.set_format('torch')

    input_ids = dataset['question_input_ids'].to(device)
    attention_mask = dataset['question_attention_mask'].to(device)
    with torch.no_grad():
        model_output = model.generate(input_ids = input_ids,
                                      attention_mask = attention_mask,
                                      generation_config = generation_config)
    pred_answer = tokenizer.batch_decode(model_output[:, max_length_input:])

    rouge = evaluate.load('rouge')
    results_rouge = rouge.compute(predictions=pred_answer, references=real_answer)
    bleurt = evaluate.load("bleurt", 'bleurt-large-512', module_type="metric")
    results_bleurt = bleurt.compute(predictions=pred_answer, references=real_answer)
    return results_rouge, results_bleurt

def visualize_weights(weights, title):

    weights_number = weights.shape[1]
    t = np.arange(1, weights.shape[0] + 1)

    plt.figure(figsize=(10, 5))
    for i in range(weights_number):
        plt.plot(t, weights[:, i])

    plt.title(title)
    plt.xlabel("t")
    plt.ylabel("$w(t)$")
    plt.show()

def test_question_answering(model, tokenizer, config, question, device):
    tokenized_question = tokenizer(question, padding='do_not_pad')
    input_ids = torch.Tensor(tokenized_question['input_ids']).type(torch.long).view(1, -1).to(device)
    attention_mask = torch.Tensor(tokenized_question['attention_mask']).type(torch.long).view(1, -1).to(device)
    tokenized_question_size = input_ids.shape[1]

    config.max_new_tokens = 50
    model_output = model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=config)
    answer = tokenizer.decode(model_output[0][tokenized_question_size:])

    print("Question:")
    print(question)
    print()
    print("Model answer:")
    print(answer)

def get_memory_history(model, dataset, tokenizer, config, file_name):
    torch.cuda.memory._record_memory_history(max_entries=200000)

    model.train(dataset=dataset, tokenizer=tokenizer, epochs=1, 
                generation_config=config, batch_size=1, save_model=False, save_weights=False)

    torch.cuda.memory._dump_snapshot(f"{file_name}.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)