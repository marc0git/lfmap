import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

import random
import numpy as np

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 16, 3),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            ),
                nn.Sequential(
                nn.Conv2d(16, 16, 3),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(16, 32, 3, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(32, 32, 3),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ),
                nn.Sequential(
                nn.Conv2d(32, 32, 3),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, 3, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, 3, padding="valid"),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
        ])

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        #self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.avg(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        #x = self.softmax(x)
        return x
    

    def get_activations(self, x, layer_indices):
        activations = []
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx in layer_indices:
                activations.append(x)
        if 'avg' in layer_indices:
            x = self.avg(x)
            activations.append(x)
        if 'fc' in layer_indices:
            if not 'avg' in layer_indices:
                x = self.avg(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            activations.append(x)
        return activations




def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({100. * correct / len(test_loader.dataset):.0f}%)\n')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # Hyperparameters
    batch_size = 256
    test_batch_size = 1000
    epochs = 10
    lr = 0.001
    momentum = 0.9
    log_interval = 10
    num_seeds = 10  # Number of different seeds
    save_dir = './model_weights'

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loaders

    train_dataset=datasets.CIFAR10('../data', train=True, download=True)

    #test_dataset=datasets.CIFAR10('../data', train=False, download=True)


    #mean,std= train_dataset.data.astype('float32').mean()/255,train_dataset.data.astype('float32').std()/255
    mean,std=train_dataset.data.mean((0,1,2))/255,train_dataset.data.std((0,1,2))/255
    #meant,stdt= test_dataset.data.astype('float32').mean()/255,test_dataset.data.astype('float32').std()/255

    print(mean,std)
    # Data loaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    train_loader = DataLoader(
        datasets.CIFAR10('../data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )

    test_loader = DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transform),
        batch_size=test_batch_size, shuffle=False
    )

    # Loop over different seeds
    for seed in range(num_seeds):
        print(f'\nTraining with seed: {seed}')
        set_seed(seed)

        # Model, loss function, and optimizer
        model = CNN(num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training and testing the model
        for epoch in range(1, epochs + 1):
            train(model, device, train_loader, optimizer, criterion, epoch)
            test(model, device, test_loader, criterion)

        # Save the model weights
        model_path = os.path.join(save_dir, f'model_seed_{seed}.pth')
        torch.save(model.state_dict(), model_path)
        print(f'Model weights saved to {model_path}')



if __name__ == '__main__':
    main()