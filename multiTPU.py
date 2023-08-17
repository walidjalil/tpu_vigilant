import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
from torchvision import datasets

# Define the CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Define train and test functions
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loader = pl.MpDeviceLoader(train_loader, device)  # Parallel loader
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        xm.optimizer_step(optimizer, barrier=True)
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    test_loader = pl.MpDeviceLoader(test_loader, device)  # Parallel loader
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

# Training entry point
def _mp_fn(rank, flags):
    torch.manual_seed(flags['seed'])
    device = xm.xla_device()
    model = Net().to(device)
    batch_size = flags['batch_size']
    test_batch_size = flags['test_batch_size']
    lr = flags['lr']
    epochs = flags['epochs']
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    test_dataset = datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor()]))
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, sampler=test_sampler, num_workers=4)
    
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

# Main
if __name__ == '__main__':
    flags = {
        'batch_size': 64,
        'test_batch_size': 64,
        'lr': 0.01,
        'epochs': 10,
        'seed': 1234,
    }
    xmp.spawn(_mp_fn, args=(flags,), nprocs=8, start_method='fork')