import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ClockDataset(Dataset):
    def __init__(self, root_dir, subset='train', transform=None):
        """
        Custom Dataset for loading clock images and their corresponding time labels.
        Args:
            root_dir (string): Directory with all the images (e.g., './data').
            subset (string): 'train' or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = os.path.join(root_dir, subset)
        self.labels_frame = pd.read_csv(os.path.join(self.root_dir, 'labels.csv'))
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Fetching filenames from the CSV
        dig_filename = self.labels_frame.iloc[idx, 0]
        ana_filename = self.labels_frame.iloc[idx, 1]
        
        # Making full paths
        dig_path = os.path.join(self.root_dir, 'digital', dig_filename)
        ana_path = os.path.join(self.root_dir, 'analog', ana_filename)
        
        # Loading images and converting to RGB format for PyTorch compatibility
        image_digital = Image.open(dig_path).convert('RGB')
        image_analog = Image.open(ana_path).convert('RGB')
        
        # Loading labels (h, m, s)
        h = self.labels_frame.iloc[idx, 2]
        m = self.labels_frame.iloc[idx, 3]
        s = self.labels_frame.iloc[idx, 4]
        
        # Creating time tensor and normalizing
        # Normalization: h in [0,23], m in [0,59], s in [0,59]
        time_tensor = torch.tensor([h/23.0, m/59.0, s/59.0], dtype=torch.float32)

        if self.transform:
            image_digital = self.transform(image_digital)
            image_analog = self.transform(image_analog)

        return {
            'digital_img': image_digital,
            'analog_img': image_analog,
            'time_label': time_tensor, # (h, m, s) normalized
            'original_time': torch.tensor([h, m, s]) # for debugging
        }

# Local test to verify the Dataset works as expected
if __name__ == "__main__":
    # Setup transformations for the images
    data_transform = transforms.Compose([
        transforms.Resize((128, 128)), # Size images to 128x128
        transforms.ToTensor(),
    ])

    # Dataset instance
    train_dataset = ClockDataset(root_dir='./data', subset='train', transform=data_transform)
    
    # Make DataLoader for batching and shuffling
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Fetch a single batch
    batch = next(iter(train_loader))
    print("Batch images shape:", batch['digital_img'].shape) # (32, 3, 128, 128)
    print("Batch labels shape:", batch['time_label'].shape)  # (32, 3)
    print("First time in batch:", batch['original_time'][0])