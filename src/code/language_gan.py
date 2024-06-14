from tqdm.notebook import tqdm
import numpy as np

import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from embedding_generator import EmbeddingGenerator
from discriminator import Discriminator


def tokenize_func_map(tokenizer, examples, column, max_length, truncation):
    return tokenizer(examples[column], padding='max_length', max_length = max_length, truncation = truncation)

def tokenize_func(tokenizer, dataset, max_length_question, max_length_answer, truncation):
    tokenizer.padding_side = 'left'
    dataset = dataset.map(lambda x: tokenizer(x['question'], padding='max_length',
                                              max_length = max_length_question, truncation = truncation))
    dataset = dataset.rename_column('input_ids', 'question_input_ids')
    dataset = dataset.rename_column('attention_mask', 'question_attention_mask')

    tokenizer.padding_side = 'right'
    dataset = dataset.map(lambda x: tokenizer(x['answer'], padding='max_length',
                                              max_length = max_length_answer, truncation = truncation))
    dataset = dataset.rename_column('input_ids', 'answer_input_ids')
    dataset = dataset.remove_columns(['question', 'answer', 'attention_mask'])

    return dataset

class LanguageGAN(nn.Module):

    def __init__(self, model, embedding_layer, func_set_embedding_layer, max_length_input=100, max_new_tokens=50, device=torch.device('cpu')):
        super(LanguageGAN, self).__init__()

        self.embedding_generator_layer = EmbeddingGenerator(embedding_layer, device).to(device)
        func_set_embedding_layer(model, self.embedding_generator_layer)

        self.discriminator = Discriminator(max_length_input + max_new_tokens).to(device)
        self.model = model

        self.max_length_input = max_length_input
        self.max_new_tokens = max_new_tokens
        self.device = device

        self.g_optimizer = None
        self.d_optimizer = None

    def forward(self, **param):
        return self.model(**param)

    def generate(self, input_ids, attention_mask, generation_config):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                   generation_config=generation_config)

    def generator_step(self, input_ids, attention_mask, generation_config):
        self.embedding_generator_layer.eval()
        self.discriminator.eval()
        with torch.no_grad():
            trajectory = self.generate(input_ids = input_ids,
                                       attention_mask = attention_mask,
                                       generation_config = generation_config)
            rewards = self.discriminator(trajectory.type(torch.float32)).view(1, -1)


        self.embedding_generator_layer.train()
        log_probs = []

        state_attention_mask = attention_mask.clone()
        batch_size = trajectory.shape[0]
        one_tensor = torch.IntTensor([1] * batch_size).view(-1, 1).to(self.device)
        for i in range(self.max_new_tokens):
            state = trajectory[:, :self.max_length_input + i]
            output = self(input_ids=state, attention_mask=state_attention_mask)[0]

            probs = nn.functional.softmax(output.sum(dim=1), dim=1)
            m = torch.distributions.categorical.Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            log_probs.append(log_prob.view(-1, 1))

            state_attention_mask = torch.concat((state_attention_mask, one_tensor), dim=1)
        log_probs = torch.concat(log_probs, dim=1).sum(dim=1)

        policy_loss = []
        for log_prob, reward in zip(log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        policy_loss = torch.concat(policy_loss).sum() / batch_size

        self.g_optimizer.zero_grad()
        policy_loss.backward()
        self.g_optimizer.step()

        return policy_loss

    def discriminator_step(self, input_ids, attention_mask, answer_input_ids, generation_config, batch_size):
        self.embedding_generator_layer.eval()
        self.discriminator.train()
        generator_output = self.generate(input_ids = input_ids,
                                         attention_mask = attention_mask,
                                         generation_config = generation_config)

        fake_output = self.discriminator(generator_output.type(torch.float32))
        real_output = self.discriminator(answer_input_ids.type(torch.float32))

        loss = nn.BCELoss()
        loss_fake = loss(fake_output, torch.zeros((batch_size, 1), device = self.device))
        loss_real = loss(real_output, torch.ones((batch_size, 1), device = self.device))
        d_loss = loss_real + loss_fake

        self.d_optimizer.zero_grad()
        d_loss.backward()
        self.d_optimizer.step()

        return d_loss

    def train(self, dataset, tokenizer, epochs, generation_config, batch_size=1, save_model=False, save_weights=False):
        generation_config.max_new_tokens = self.max_new_tokens

        if self.g_optimizer is None:
            self.g_optimizer = torch.optim.Adam(self.embedding_generator_layer.generator_parameters(), lr=0.001)
        if self.d_optimizer is None:
            self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0005)

        dataset = tokenize_func(tokenizer, dataset, self.max_length_input, self.max_new_tokens, True)
        dataset.set_format('torch')
        data_size = len(dataset)

        frozen_weights = []
        tunable_weights = []

        for epoch in tqdm(range(1, epochs + 1)):
            if save_weights:
                frozen_weights.append((self.embedding_generator_layer.embedding_layer.weight.data.clone()[0][0].cpu().numpy(),
                                    self.embedding_generator_layer.embedding_layer.weight.data.clone()[0][1].cpu().numpy(),
                                    self.embedding_generator_layer.embedding_layer.weight.data.clone()[0][2].cpu().numpy(),
                                    self.embedding_generator_layer.embedding_layer.weight.data.clone()[0][3].cpu().numpy()))

                tunable_weights.append((self.embedding_generator_layer.layer1[0].weight.data.clone()[0][0].cpu().numpy(),
                                    self.embedding_generator_layer.layer1[0].weight.data.clone()[0][1].cpu().numpy(),
                                    self.embedding_generator_layer.layer1[0].weight.data.clone()[0][2].cpu().numpy(),
                                    self.embedding_generator_layer.layer1[0].weight.data.clone()[0][3].cpu().numpy()))

            if save_model and (epoch % 5 == 0):
                file_name = 'model_state' + str(epoch) + '.pth'
                torch.save(self.model.state_dict(), file_name)

            dataset = dataset.shuffle()
            input_ids = dataset['question_input_ids']
            attention_mask = dataset['question_attention_mask']
            answer_input_ids = torch.concat((input_ids, dataset['answer_input_ids']), dim=1)

            for i in range(0, data_size, batch_size):
                batch_input_ids = input_ids[i:i + batch_size].to(self.device)
                batch_attention_mask = attention_mask[i:i + batch_size].to(self.device)
                batch_answer_input_ids = answer_input_ids[i:i + batch_size].to(self.device)


                self.discriminator_step(input_ids=batch_input_ids, attention_mask=batch_attention_mask,
                                        answer_input_ids=batch_answer_input_ids, generation_config=generation_config,
                                        batch_size=batch_size)

                self.generator_step(input_ids=batch_input_ids, attention_mask=batch_attention_mask,
                                    generation_config = generation_config)

        frozen_weights = np.array(frozen_weights)
        tunable_weights = np.array(tunable_weights)

        return frozen_weights, tunable_weights