#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 00:03:45 2023

@author: walidajalil
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

# Define the model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Training function
def train(rank, world_size):
    torch.manual_seed(42)
    
    # Get the device
    device = xm.xla_device()

    # Define the data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_dataset = datasets.MNIST('.', train=True, download=True, transform=transform)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
    )
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)

    # Instantiate the model, loss, and optimizer
    model = SimpleNet().to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Training loop
    for epoch in range(5):  # 5 epochs
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            xm.optimizer_step(optimizer)  # Handles synchronization
            if batch_idx % 10 == 0:
                print(f"Rank {rank}, Batch {batch_idx}, Loss {loss.item()}")

# Entry point
def _mp_fn(rank, flags):
    world_size = flags['world_size']
    train(rank, world_size)

if __name__ == '__main__':
    flags = {'world_size': 8}
    xmp.spawn(_mp_fn, args=(flags,), nprocs=flags['world_size'])
