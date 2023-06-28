import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from concrete_autoencoder import ConcreteAutoEncoder


def jaccard_index(a, b):
    a = set(a.numpy()) if type(a) != set else a
    b = set(b.numpy()) if type(b) != set else b
    a_and_b = a.intersection(b)
    a_or_b = a.union(b)

    return len(a_and_b) / len(a_or_b)


def train_masked_mean(dt, train_idx):
    x, x_mask = dt.data[train_idx].squeeze(), dt.data_mask[train_idx].squeeze()
    masked_x = x * x_mask
    masked_mean = torch.sum(masked_x, dim=0) / torch.sum(x_mask, dim=0)

    return masked_mean


def impute_mean(x, x_mask, masked_mean):
    inv_mask = x_mask.int() ^ 1
    impute_value = inv_mask * masked_mean
    x = x + impute_value

    return x


def load_model(file_nm, input_dim, k, alpha=0.99999, decoder_type='mlp'):
    model = ConcreteAutoEncoder(input_dim, k, alpha=alpha, decoder_type=decoder_type)
    model.load_state_dict(torch.load(file_nm)['model_state_dict'])
    return model


def create_mnist_figures(model, images, image_masks=None, folder='./data/figures/', tag=''):
    samples_per_feature = [1, 3]
    probs = model.get_prob()

    if not os.path.exists(folder):
        os.makedirs(folder)

    for i in samples_per_feature:
        top = torch.topk(probs, i)
        im = torch.zeros(28 * 28)
        c = i
        for idx in top.indices.transpose(0, 1):
            im[idx] = c
            c -= 1
        plt.imshow(torch.reshape(im, (28, 28)))
        plt.savefig(f'{folder}/selected{tag}_{i}.png')

    fig, axs = plt.subplots((len(images) * 2 + 4) // 5, 5)
    for i, ax in enumerate(axs.flatten()[:10]):
        im = images[i] if image_masks is None else images[i]*image_masks[i]
        ax.imshow(torch.reshape(im, (28, 28)))
    for i, ax in enumerate(axs.flatten()[10:]):
        X_mask = image_masks[i] if image_masks is not None else None
        pred = model(images[i], train=False, X_mask=X_mask)
        ax.imshow(np.reshape(pred.detach().numpy(), (28, 28)))
    plt.savefig(f'{folder}/reconstructed{tag}.png')


def plot_performance(losses, folder='./data/figures/', tag=''):
    if not os.path.exists(folder):
        os.makedirs(folder)
    _, ax = plt.subplots()
    for key, loss in losses.items():
        ax.plot(loss, label=key)
    ax.set_xlabel('Number of epochs')
    ax.set_ylabel('MSE loss')
    ax.legend()
    ax.grid()
    plt.savefig(f'{folder}/performance{tag}.png')
