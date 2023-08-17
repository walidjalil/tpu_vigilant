#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 18:51:02 2023

@author: walidajalil
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

# VAE model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20) # mean
        self.fc22 = nn.Linear(400, 20) # logvariance

        # Decoder
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Hyperparameters
batch_size = 128
epochs = 10
lr = 0.001

# MNIST Data
transform = transforms.ToTensor()
train_data = datasets.MNIST('./data', train=True, transform=transform)

def train(index):
    device = xm.xla_device()
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    mp_device_loader = pl.MpDeviceLoader(train_loader, device)
    # Model and Optimizer
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training Loop
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, _) in enumerate(mp_device_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            #xm.optimizer_step(optimizer, barrier=True) TEST BOTH WITH AND WITHOUT BARRIER
            xm.optimizer_step(optimizer)
    
        print(f'Epoch: {epoch} \t Loss: {train_loss / len(train_loader.dataset)}')

if __name__ == '__main__':
  xmp.spawn(train, args=())