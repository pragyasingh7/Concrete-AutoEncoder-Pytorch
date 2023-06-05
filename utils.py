import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.distributions.binomial import Binomial
from torch.distributions.uniform import Uniform

from concrete_autoencoder import ConcreteAutoEncoder


def load_model(file_nm, input_dim, k, alpha=0.99999, decoder_type='mlp'):
    model = ConcreteAutoEncoder(input_dim, k, alpha=alpha, decoder_type=decoder_type)
    model.load_state_dict(torch.load(file_nm)['model_state_dict'])
    return model


def mask_features(row):
    missing_prob = Uniform(0.4, 0.6).sample(row.shape)
    mask = Binomial(1, missing_prob).sample()
    return row*mask


def load_dataset(dataset, train=True, download=True):
    if dataset == 'mnist':
        dt = datasets.MNIST(root='./data', download=download, train=train,
                            transform=T.Compose([T.ToTensor(), T.Lambda(lambda x: torch.flatten(x))]))
        input_dim = 28*28
    elif dataset == 'mnist_partial':
        dt = datasets.MNIST(root='./data', download=download, train=train,
                            transform=T.Compose([T.ToTensor(), T.Lambda(lambda x: mask_features(torch.flatten(x)))]))
        input_dim = 28*28
    else:
        raise ValueError(f'Unsupported dataset {dataset}')
    return dt, input_dim


def create_mnist_figures(model, images, tag=''):
    samples_per_feature = [1, 3]
    probs = model.get_prob()
    for i in samples_per_feature:
        top = torch.topk(probs, i)
        im = torch.zeros(28*28)
        for idx in top.indices:
            c = 1 if i == 1 else np.random.uniform(0, 1)
            im[idx] = c
        plt.imshow(torch.reshape(im, (28, 28)))
        plt.savefig(f'./data/figures/selected_features{tag}_{i}.png')

    fig, axs = plt.subplots((len(images)*2 + 4) // 5, 5)
    for i, ax in enumerate(axs.flatten()[:10]):
        ax.imshow(torch.reshape(images[i], (28, 28)))
    for i, ax in enumerate(axs.flatten()[10:]):
        pred = model(images[i], train=False)
        ax.imshow(np.reshape(pred.detach().numpy(), (28, 28)))
    plt.savefig(f'./data/figures/reconstructed_images{tag}.png')
