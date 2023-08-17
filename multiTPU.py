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
class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(128 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Define train and test functions
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    dataset_length = len(train_loader.dataset)  # Save the dataset length before wrapping
    train_loader = pl.MpDeviceLoader(train_loader, device)  # Parallel loader
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        xm.optimizer_step(optimizer, barrier=True)
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), dataset_length, loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    dataset_length = len(test_loader.dataset)  # Save the dataset length before wrapping
    test_loader = pl.MpDeviceLoader(test_loader, device)  # Parallel loader
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= dataset_length
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, dataset_length, 100. * correct / dataset_length))


# Training entry point
def _mp_fn(rank, flags):
    torch.manual_seed(flags['seed'])
    device = xm.xla_device()
    model = ComplexCNN().to(device)
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
        'batch_size': 128,
        'test_batch_size': 128,
        'lr': 0.1,
        'epochs': 20,
        'seed': 1234,
    }
    xmp.spawn(_mp_fn, args=(flags,), nprocs=8, start_method='fork')
    
    
    
    