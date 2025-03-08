#imports
import torch.nn as nn
import torch
import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd 
import random 
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F

def load_model(dataset_name, dim_emb, idx_model=1, device='cuda'):
    path = f"checkpoints/checkpoints_{dataset_name}{dim_emb}/m{idx_model}.ckpt"

    model= PLModel(dim_emb, ch=1 if dataset_name in ["mnist", "FashionMNIST"] else 3)
    model.load_state_dict(torch.load(path, map_location=device)['state_dict'])

    return model

#model architecture code
class Encoder(nn.Module):

    def __init__(self, encoded_space_dim, overparam=False, activation=nn.ReLU(), ch=1, bn=True, input_size=28):
        super().__init__()
        self.bn = bn
        self.activation = activation
        self.inputsize = input_size
        self.bottleneck_size = encoded_space_dim
        self.ch=ch

        # Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(ch, 8, 3, stride=2, padding=1),
            self.activation,
            nn.BatchNorm2d(8) if self.bn else nn.Identity(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16) if self.bn else nn.Identity(),
            self.activation,
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            self.activation
        )

        # Calculate the size of the linear input based on the last convolutional layer's output shape
        self.conv_output_size = self._get_conv_output_size()

        # Linear section
        if overparam:
            self.encoder_lin = nn.Sequential(
                nn.Linear(self.conv_output_size, 128),
                self.activation,
                nn.Linear(128, 128),
                self.activation,
                nn.Linear(128, 128),
                self.activation,
                nn.Linear(128, encoded_space_dim)
            )
        else:
            self.encoder_lin = nn.Sequential(
                nn.Linear(self.conv_output_size, 128),
                self.activation,
                nn.Linear(128, encoded_space_dim)
            )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.encoder_lin(x)
        return x

    def _get_conv_output_size(self):
        # Helper function to calculate the size of the linear input
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.ch,  self.inputsize, self.inputsize)  # Adjust the input size as needed
            dummy_output = self.encoder_cnn(dummy_input)
            return dummy_output.view(dummy_output.size(0), -1).size(1)

class Decoder(nn.Module):

    def __init__(self, encoded_space_dim, overparam=False, activation=nn.ReLU(), bn=True, ch=1):
        super().__init__()
        self.bn = bn
        self.activation = activation
        self.encoded_space_dim = encoded_space_dim
        self.ch=ch
        self.conv_output_size =32*3*3

        if overparam:
            self.decoder_lin = nn.Sequential(
                nn.Linear(encoded_space_dim, 128),
                self.activation,
                nn.Linear(128, 128),
                self.activation,
                nn.Linear(128, 128),
                self.activation,
                nn.Linear(128, self.conv_output_size),
                self.activation
            )
        else:
            self.decoder_lin = nn.Sequential(
                nn.Linear(encoded_space_dim, 128),
                self.activation,
                nn.Linear(128, self.conv_output_size),
                self.activation
            )
        #self.conv_output_size = self._get_conv_output_size()

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))


        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16) if self.bn else nn.Identity(),
            self.activation,
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8) if self.bn else nn.Identity(),
            self.activation,
            nn.ConvTranspose2d(8, ch, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = x.view(x.size(0), 32, 3, 3)  # Unflatten
        x = self.decoder_conv(x)
        return x

    def _get_conv_output_size(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.encoded_space_dim)  # Adjust the input size as needed
            dummy_output = self.decoder_lin(dummy_input)
            return dummy_output.view(dummy_output.size(0), -1).size(1)




class Autoencoder(nn.Module):
    def __init__(self, bottleneck_size,overparam=False,activation=nn.ReLU(),bn=True,ch=1,input_size=28):
        super().__init__()
        self.encoder=Encoder(bottleneck_size,overparam=overparam,activation=activation,bn=bn,ch=ch,input_size=input_size)
        self.decoder=Decoder(bottleneck_size,overparam=overparam,activation=activation,bn=bn,ch=ch)

    def forward(self,x):
        return self.decoder(self.encoder(x))

    def encode(self,x):
        return self.encoder(x)

    def decode(self,z):
        return self.decode(z)
    
class PLModel(nn.Module):
    def __init__(self, bottleneck_size,overparam=False,activation=nn.ReLU(),bn=True,ch=1,input_size=28):
        super().__init__()
        self.model= Autoencoder(bottleneck_size,overparam,activation,bn,ch,input_size)

    def get_embeddings(self,dataset, batch_size=256, device='cuda', indices=None):
        """
        Processes a dataset through an encoder model to obtain embeddings.

        Args:
        - dataset (torch.utils.data.Dataset): A PyTorch dataset.
        - encoder (torch.nn.Module): A PyTorch model that returns embeddings.
        - batch_size (int): Batch size for processing.
        - device (str): Device to run the computations on ('cuda' or 'cpu').

        Returns:
        - embeddings (torch.Tensor): Tensor of embeddings.
        - labels (torch.Tensor): Tensor of labels.
        """
        encoder = self.model.encoder.to(device)
        encoder.eval()
        if indices is not None:
            dataset = torch.utils.data.Subset(dataset, indices)
            
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        embeddings = []
        labels = []

        with torch.no_grad():
            for batch in loader:
                inputs, batch_labels = batch
                inputs = inputs.to(device)
                batch_embeddings = encoder(inputs)
                embeddings.append(batch_embeddings.cpu())
                labels.append(batch_labels.cpu())

        return torch.cat(embeddings), torch.cat(labels)
    
    def decode_embeddings(self,embeddings, batch_size=256, device='cuda', indices=None):
        # check if embeddings is already a dataset
        if isinstance(embeddings, torch.utils.data.Dataset):
            dataset = embeddings
        else:
            dataset = torch.utils.data.TensorDataset(embeddings)
        decoder = self.model.decoder.to(device)
        decoder.eval()
        if indices is not None:
            dataset = torch.utils.data.Subset(dataset, indices)
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        decode_results = []
        with torch.no_grad():
            for batch in loader:
                batch = batch[0].to(device)
                decode_results.append(decoder(batch).cpu())
            
        decode_results = torch.cat(decode_results, dim=0)
        return decode_results

def get_embeddings(dataset, encoder, batch_size=256, device='cuda', indices=None):
    """
    Processes a dataset through an encoder model to obtain embeddings.

    Args:
    - dataset (torch.utils.data.Dataset): A PyTorch dataset.
    - encoder (torch.nn.Module): A PyTorch model that returns embeddings.
    - batch_size (int): Batch size for processing.
    - device (str): Device to run the computations on ('cuda' or 'cpu').

    Returns:
    - embeddings (torch.Tensor): Tensor of embeddings.
    - labels (torch.Tensor): Tensor of labels.
    """
    encoder = encoder.to(device)
    encoder.eval()
    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices)
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            inputs, batch_labels = batch
            inputs = inputs.to(device)
            batch_embeddings = encoder(inputs)
            embeddings.append(batch_embeddings.cpu())
            labels.append(batch_labels.cpu())

    return torch.cat(embeddings), torch.cat(labels)

def decode_embeddings(embeddings, decoder, batch_size=256, device='cuda', indices=None):
    # check if embeddings is already a dataset
    if isinstance(embeddings, torch.utils.data.Dataset):
        dataset = embeddings
    else:
        dataset = torch.utils.data.TensorDataset(embeddings)
    decoder = decoder.to(device)
    decoder.eval()
    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    decode_results = []
    with torch.no_grad():
        for batch in loader:
            batch = batch[0].to(device)
            decode_results.append(decoder(batch).cpu())
        
    decode_results = torch.cat(decode_results, dim=0)
    return decode_results