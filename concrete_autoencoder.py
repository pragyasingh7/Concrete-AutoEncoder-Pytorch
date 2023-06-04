import torch
from torch import nn
import torch.nn.functional as F


class ConcreteEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, start_temp=10.0, min_temp=0.1, alpha=0.99999):
        super().__init__()
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.alpha = alpha

        self.temp = start_temp
        self.logits = nn.Parameter(torch.empty(output_dim, input_dim))
        nn.init.xavier_normal_(self.logits)

    def forward(self, X, train=True):
        uniform = torch.rand(self.logits.shape)
        gumbel = -torch.log(-torch.log(uniform))
        self.temp = max([self.temp * self.alpha, self.min_temp])
        noisy_logits = (self.logits + gumbel) / self.temp
        samples = F.softmax(noisy_logits, dim=-1)

        discrete_logits = F.one_hot(torch.argmax(self.logits, dim=-1), self.logits.shape[1]).float()

        selection = samples if train else discrete_logits
        Y = torch.matmul(X, torch.transpose(selection, -1, -2))

        return Y


class Decoder(nn.Module):
    def __init__(self, decoder_type, input_dim, output_dim):
        super().__init__()
        if decoder_type == 'linear_regressor':
            self.decoder = nn.Linear(input_dim, output_dim)
        elif decoder_type == 'mlp':
            self.decoder = nn.Sequential(
                nn.Linear(input_dim, 320), nn.LeakyReLU(), nn.Dropout(0.1),
                nn.Linear(320, 320), nn.LeakyReLU(), nn.Dropout(0.1),
                nn.Linear(320, output_dim)
            )
        else:
            raise ValueError(f'Unsupported decoder type: {decoder_type}')

    def forward(self, X):
        return self.decoder(X)


class ConcreteAutoEncoder(nn.Module):
    def __init__(self, input_dim, k, start_temp=10.0, min_temp=0.1, alpha=0.99999, decoder_type='linear_regressor'):
        super().__init__()
        self.encoder = ConcreteEncoder(input_dim, k, start_temp=start_temp, min_temp=min_temp, alpha=alpha)
        self.decoder = Decoder(decoder_type, k, input_dim)

    def get_mean_max(self):
        return torch.mean(torch.max(F.softmax(self.encoder.logits, -1)))

    def get_prob(self):
        return F.softmax(self.encoder.logits, -1)

    def get_indices(self):
        return torch.argmax(self.encoder.logits, dim=-1)

    def forward(self, X, train=True):
        selected_features = self.encoder(X, train=train)
        outputs = self.decoder(selected_features)
        return outputs