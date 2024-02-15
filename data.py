import random

import torch
import torchvision

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def data_to_tensor(data):
    """ Loads dataset to memory, applies transform"""
    loader = torch.utils.data.DataLoader(data, batch_size=len(data))
    img, label = next(iter(loader))
    return img, label
def noniid_partition_loader(data, bsz=8, m_per_shard=50, n_shards_per_client = 1):
    """ semi-pathological client sample partition
        1. sort examples by label, form shards of size 300 by grouping points
           successively
        2. each client is 2 random shards
        most clients will have 2 digits, at most 4
        """

    # load data into memory
    img, label = data_to_tensor(data)

    # sort
    idx = torch.argsort(label)
    img = img[idx]
    label = label[idx]

    # split into n_shards of size m_per_shard
    m = len(data)
    # assert m % m_per_shard == 0
    n_shards = m // m_per_shard
    shards_idx = [
        torch.arange(m_per_shard * i, m_per_shard * (i + 1))
        for i in range(n_shards)
    ]
    random.shuffle(shards_idx)  # shuffle shards

    # pick shards to create a dataset for each client
    assert n_shards % n_shards_per_client == 0
    n_clients = n_shards // n_shards_per_client
    client_data = [
        torch.utils.data.TensorDataset(
            torch.cat([img[shards_idx[j]] for j in range(
                i * n_shards_per_client, (i + 1) * n_shards_per_client)]),
            torch.cat([label[shards_idx[j]] for j in range(
                i * n_shards_per_client, (i + 1) * n_shards_per_client)])
        )
        for i in range(n_clients)
    ]

    # make dataloaders
    client_loader = [
        torch.utils.data.DataLoader(x, batch_size=bsz, shuffle=True)
        for x in client_data
    ]
    return client_loader