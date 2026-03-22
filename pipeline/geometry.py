import os
import random
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image

# Detect if we have a GPU (CUDA for NVIDIA, MPS for Mac M1/M2/M3) or just CPU
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


class ClockGeometryNet(nn.Module):
    def __init__(self):
        super(ClockGeometryNet, self).__init__()
        
        # We use ResNet18. It's a "Convolutional Neural Network" (CNN).
        # It is excellent at extracting shapes (like circles) and edges.
        # weights='DEFAULT' downloads pre-trained weights from ImageNet (helps learn faster).
        self.base_model = models.resnet18(weights='DEFAULT')
        
        # ResNet usually outputs 1000 numbers (probabilities for 1000 types of objects like 'cat', 'dog').
        # We need it to output exactly 3 numbers: (x, y, radius).
        # So we replace the last layer ("fc" = fully connected).
        self.base_model.fc = nn.Linear(512, 3)

    def forward(self, x):
        # We usually use Sigmoid at the end to force output between 0 and 1.
        # Why? Because our coordinates (x, y) are normalized (0.0 to 1.0).
        # However, because of our augmentation, the center might slightly exit the frame (e.g. 1.05).
        # So we leave it as raw linear output (no activation) to allow values slightly outside 0-1.
        return self.base_model(x)