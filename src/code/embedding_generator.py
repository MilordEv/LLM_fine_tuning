import itertools

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class EmbeddingGenerator(nn.Module):

    def __init__(self, embedding_layer, device):
        super(EmbeddingGenerator, self).__init__()
        self.device = device

        self.embedding_layer = embedding_layer
        self.embedding_layer_type = embedding_layer.weight.data.dtype

        generator_layer_size =  embedding_layer.embedding_dim + 100
        self.layer1 = nn.Sequential(nn.Linear(in_features=generator_layer_size, out_features=generator_layer_size // 2,
                                              dtype = self.embedding_layer_type),
                                    nn.LeakyReLU(), nn.Dropout(0.2))
        self.layer2 = nn.Sequential(nn.Linear(in_features=generator_layer_size // 2, out_features=generator_layer_size // 4,
                                              dtype = self.embedding_layer_type),
                                    nn.LeakyReLU(), nn.Dropout(0.2))
        self.layer3 = nn.Sequential(nn.Linear(in_features=generator_layer_size // 4, out_features=generator_layer_size // 2,
                                              dtype = self.embedding_layer_type),
                                    nn.LeakyReLU(), nn.Dropout(0.2))
        self.layer4 = nn.Linear(in_features=generator_layer_size // 2, out_features=generator_layer_size - 100,
                                              dtype = self.embedding_layer_type)

    def forward(self, x):

        embedding_output = self.embedding_layer(x)

        embedding_output_shape = list(embedding_output.shape)
        embedding_output_shape[-1] = 100

        random_input = torch.randn(embedding_output_shape, dtype=self.embedding_layer_type)
        x = torch.concat((random_input.to(self.device), embedding_output), dim=-1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x += embedding_output
        return x

    def generator_parameters(self):
        params = [self.layer1.parameters(), self.layer2.parameters(),
                  self.layer3.parameters(), self.layer4.parameters()]
        return itertools.chain(*params)

    def get_weight_type(self):
        return self.embedding_layer_type

    def get_embedding_size(self):
        return self.embedding_layer_size