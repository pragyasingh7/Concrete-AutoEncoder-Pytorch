import codecs
import sys
import os
import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.distributions.binomial import Binomial
from torch.distributions.uniform import Uniform


class BaseDataset(ABC, Dataset):
    def __init__(self, train=False, missing_data=False):
        super().__init__()
        self.train = train
        self.missing_data = missing_data

        self.data_mask = None
        self.data, self.targets = self._load_data()


    @abstractmethod
    def _load_data(self):
        return NotImplementedError


    @abstractmethod
    def _transform(self):
        return NotImplementedError


    def _mask_features(self, x):
        missing_prob = Uniform(0.4, 0.6).sample(x.shape)
        mask = Binomial(1, missing_prob).sample()
        return x, mask


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        if self.missing_data:
            img_mask = self.data_mask[index]
            return [img, img_mask], target
        return img, target


class Superconductivity(BaseDataset):
    def __init__(self, train=False, missing_data=False):
        super().__init__(train=train, missing_data=missing_data)


    def _load_data(self, train_ratio=0.8):
        data = pd.read_csv('./data/superconductivty+data/train.csv')
        n_samples = data.shape[0]

        # Setting local seed for same train-test split
        indices = list(range(n_samples))
        rng = np.random.default_rng(99)
        rng.shuffle(indices)

        train_idx, test_idx = indices[:int(train_ratio*n_samples)], indices[int(train_ratio*n_samples):]
        data = data.iloc[train_idx if self.train else test_idx]

        targets = data['critical_temp'].to_numpy()
        data = data.drop('critical_temp', axis=1)

        return self._transform(data), targets


    def _transform(self, data):
        n_features = data.shape[1]
        data = torch.tensor(data.to_numpy(), dtype=torch.get_default_dtype())
        # Apply min-max scaling
        data = (data - torch.min(data, 0).values) / (torch.max(data, 0).values - torch.min(data, 0).values)
        data = data.reshape((-1, 1, n_features))

        if self.missing_data:
            _, self.data_mask = self._mask_features(data)

        return data


class MNIST(BaseDataset):
    def __init__(self, train=False, missing_data=False):
        super().__init__(train=train, missing_data=missing_data)


    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join('./data/MNIST/raw', image_file))
        data = self._transform(data)

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join('./data/MNIST/raw', label_file))

        return data, targets


    def _transform(self, data):
        data = data.to(torch.get_default_dtype()).div(255)
        data = data.reshape((-1, 1, 28 * 28))

        if self.missing_data:
            _, self.data_mask = self._mask_features(data)

        return data


# ----------------------------------------------------------
# Utilities to load raw MNIST data
# ----------------------------------------------------------

def get_int(b: bytes) -> int:
    return int(codecs.encode(b, "hex"), 16)


def _flip_byte_order(t: torch.Tensor) -> torch.Tensor:
    return (
        t.contiguous().view(torch.uint8).view(*t.shape, t.element_size()).flip(-1).view(*t.shape[:-1], -1).view(t.dtype)
    )


SN3_PASCALVINCENT_TYPEMAP = {
    8: torch.uint8,
    9: torch.int8,
    11: torch.int16,
    12: torch.int32,
    13: torch.float32,
    14: torch.float64,
}


def read_sn3_pascalvincent_tensor(path: str, strict: bool = True) -> torch.Tensor:
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
    Argument may be a filename, compressed filename, or file object.
    """
    # read
    with open(path, "rb") as f:
        data = f.read()
    # parse
    magic = get_int(data[0:4])
    nd = magic % 256
    ty = magic // 256
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    torch_type = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]

    parsed = torch.frombuffer(bytearray(data), dtype=torch_type, offset=(4 * (nd + 1)))

    # The MNIST format uses the big endian byte order, while `torch.frombuffer` uses whatever the system uses. In case
    # that is little endian and the dtype has more than one byte, we need to flip them.
    if sys.byteorder == "little" and parsed.element_size() > 1:
        parsed = _flip_byte_order(parsed)

    assert parsed.shape[0] == np.prod(s) or not strict
    return parsed.view(*s)


def read_label_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    if x.dtype != torch.uint8:
        raise TypeError(f"x should be of dtype torch.uint8 instead of {x.dtype}")
    if x.ndimension() != 1:
        raise ValueError(f"x should have 1 dimension instead of {x.ndimension()}")
    return x.long()


def read_image_file(path: str) -> torch.Tensor:
    x = read_sn3_pascalvincent_tensor(path, strict=False)
    if x.dtype != torch.uint8:
        raise TypeError(f"x should be of dtype torch.uint8 instead of {x.dtype}")
    if x.ndimension() != 3:
        raise ValueError(f"x should have 3 dimension instead of {x.ndimension()}")
    return x
