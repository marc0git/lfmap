from collections import defaultdict
import torch
from torch.utils.data import DataLoader
import random 
import numpy as np
import torchvision
from torchvision import transforms

def sample_n_from_each_class_as_tensor(dataset, n):
        """
        Samples n instances from each class of a torchvision dataset and returns them as tensors.

        Args:
        - dataset: A torchvision dataset with an attribute 'targets' or 'labels' indicating class labels.
        - n (int): Number of samples to draw from each class.

        Returns:
        - A tuple of two tensors: (data_tensor, label_tensor).
        """

        # Check if the dataset has 'targets' or 'labels' attribute
        if hasattr(dataset, 'targets'):
            labels = dataset.targets
        elif hasattr(dataset, 'labels'):
            labels = dataset.labels
        else:
            raise ValueError("Dataset must have 'targets' or 'labels' attribute")

        # Group indices by class
        indices_per_class = defaultdict(list)
        for idx, label in enumerate(labels):
            indices_per_class[label.item()].append(idx)

        # Sample n indices from each class
        sampled_indices = []
        # for label, indices in indices_per_class.items():
        for k in range(len(np.unique(dataset.targets))):
            if len(indices_per_class[k]) >= n:
                sampled_indices.extend(random.sample(indices_per_class[k], n))
            else:
                sampled_indices.extend(indices_per_class[k])

        # Extract data and labels for the sampled indices
        sampled_data = [dataset[idx][0] for idx in sampled_indices]
        sampled_labels = [dataset[idx][1] for idx in sampled_indices]

        # Convert lists to tensors
        data_tensor = torch.stack(sampled_data)
        label_tensor = torch.tensor(sampled_labels)

        return data_tensor, label_tensor, sampled_indices

def load_dataset(dataset, data_dir='dataset'):
    """
    Load a dataset from torchvision.datasets.

    Args:
    - dataset (str): Name of the dataset to load.
    - data_dir (str): Directory to load the dataset from.

    Returns:
    - Dataset: A torchvision Dataset.
    """

    if dataset == "mnist":
        train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
        test_dataset  = torchvision.datasets.MNIST(data_dir, train=False, download=True)
    elif dataset == "cifar":
        train_dataset = torchvision.datasets.CIFAR10(data_dir, train=True, download=True)
        test_dataset  = torchvision.datasets.CIFAR10(data_dir, train=False, download=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    mean,std= train_dataset.data.float().mean()/255,train_dataset.data.float().std()/255
    print("Train mean and std: ",mean, std)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
        ])

    print("Test mean and std: ",mean, std)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
        ])
            
    train_dataset.transform = train_transform
    test_dataset.transform = test_transform

    m=len(train_dataset)

    return train_dataset, test_dataset