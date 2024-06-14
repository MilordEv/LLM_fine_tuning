import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, input_layer_size):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_features = input_layer_size, out_features=1024),
                                    nn.LeakyReLU(), nn.Dropout(0.2))

        self.layer2 = nn.Sequential(nn.Linear(in_features=1024, out_features=512),
                                    nn.LeakyReLU(), nn.Dropout(0.2))

        self.layer3 = nn.Sequential(nn.Linear(in_features=512, out_features=256),
                                    nn.LeakyReLU(), nn.Dropout(0.2))

        self.layer4 = nn.Sequential(nn.Linear(in_features=256, out_features=1),
                                    nn.Sigmoid())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x