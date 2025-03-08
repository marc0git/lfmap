import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np



# Define the Convolutional Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self,channels=1):
        super(ConvAutoencoder, self).__init__()
        inn_size=7 if input_channels==1 else 8
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1), # [b, 32, 14, 14]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # [b, 64, 7, 7]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * inn_size * inn_size, 512)  # Latent space
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(512, 64 * inn_size * inn_size),
            nn.ReLU(),
            nn.Unflatten(1, (64, inn_size, inn_size)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # [b, 32, 14, 14]
            nn.ReLU(),
            nn.ConvTranspose2d(32, channels, kernel_size=3, stride=2, padding=1, output_padding=1), # [b, 1, 28, 28]
            #nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Function to train the autoencoder
def train_autoencoder(seed, train_loader, test_loader, save_dir):
    torch.manual_seed(seed)
    dataset_name='CIFAR10'
    ch=1 if dataset_name=='MNIST' or dataset_name=='FashionMNIST' else 3
    model = ConvAutoencoder(channels=ch).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for data, _ in train_loader:
            data = data.to(device)
            _, outputs = model(data)
            loss = criterion(outputs, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save embeddings
    model.eval()
    test_embeddings = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            embeddings, _ = model(data)
            test_embeddings.append(embeddings.cpu().numpy())
    
    test_embeddings = np.concatenate(test_embeddings, axis=0)
    np.save(os.path.join(save_dir, f'test_embeddings_seed_{seed}.npy'), test_embeddings)
    torch.save(model.state_dict(), os.path.join(save_dir, f'model_weights_{seed}.pth'))



def get_dataset(dataset_name):
    transform = transforms.ToTensor()
    if dataset_name == 'MNIST':
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        input_channels = 1
    elif dataset_name == 'FashionMNIST':
        train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
        input_channels = 1
    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4913997551666284, 0.48215855929893703, 0.4465309133731618), (0.24703225141799082, 0.24348516474564, 0.26158783926049628))
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
        input_channels = 3
    else:
        raise ValueError("Unsupported dataset. Choose from 'MNIST', 'FashionMNIST', or 'CIFAR10'.")
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    return train_loader, test_loader, input_channels


if __name__ == '__main__':
    # Create directory to save embeddings
    save_dir = 'checkpoints_'

    # Select dataset
    dataset_name = 'CIFAR10'  # Change to 'MNIST', 'FashionMNIST', or 'CIFAR10'

    save_dir=save_dir+ '_'+dataset_name

    os.makedirs(save_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Select dataset
    train_loader, test_loader, input_channels = get_dataset(dataset_name)

    # Train and save embeddings for two different seeds
    seeds = [42, 123]
    for seed in seeds:
        train_autoencoder(seed, train_loader, test_loader, save_dir)

    # Save model weights

    print(f"Embeddings and model weights saved in directory: {save_dir}")