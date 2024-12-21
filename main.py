import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
# import matplotlib.pyplot as plt

import numpy as np

# data = [[1, 2], [3, 4]]
# x_data = torch.tensor(data)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

training_dataset = datasets.ImageFolder(root="dataset/training_data", transform=transform)
training_loader = DataLoader(training_dataset, batch_size=32, shuffle=True)