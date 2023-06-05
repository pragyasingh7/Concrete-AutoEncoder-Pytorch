import math

import torch
from torch.utils.data import DataLoader

from utils import *
from concrete_autoencoder import ConcreteAutoEncoder

DATASET = 'mnist'
NUM_TRYOUTS = 1
START_TEMP = 10.0
MIN_TEMP = 0.01
MEAN_MAX_TARGET = 0.998
BATCH_SIZE = 64
DECODER_TYPE = 'mlp'
LEARNING_RATE = 1e-3
DOWNLOAD_DATA = True

k = 20
num_epochs = 300

trainset, input_dim = load_dataset(DATASET, train=True, download=DOWNLOAD_DATA)
train_dataloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset, _ = load_dataset(DATASET, train=False, download=DOWNLOAD_DATA)
testset = torch.stack([x[0] for x in testset], dim=0)

for i in range(NUM_TRYOUTS):
    alpha = math.exp(math.log(MIN_TEMP / START_TEMP) / (num_epochs * len(train_dataloader)))
    model = ConcreteAutoEncoder(input_dim, k, start_temp=START_TEMP, min_temp=MIN_TEMP, alpha=alpha,
                                decoder_type=DECODER_TYPE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()

    for j in range(num_epochs):
        epoch_loss = 0
        for X, _ in train_dataloader:
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, X)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_dataloader)
        print(f'For Epoch{j}, Loss: {epoch_loss}')

    torch.save({
        'tryout_num': i+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'./data/saved_models/mnist_{DECODER_TYPE}_{i+1}.pt')

    model.eval()
    test_output = model(testset, train=False)
    test_loss = criterion(test_output, testset)
    print(f'Test loss for tryout {i}: {test_loss}')

    if DATASET == 'mnist':
        idx = np.random.randint(0, len(testset), 10)
        create_mnist_figures(model, testset[idx])

    if model.get_mean_max() >= MEAN_MAX_TARGET:
        print(f'mean_max_target reached during tryout {i}, exiting training loop')
        break

    num_epochs *= 2

# model = load_model(f'./data/saved_models/mnist_{DECODER_TYPE}_1.pt', input_dim, k)
