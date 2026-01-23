import os
import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ClockDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Find all digital images and match them with their analog counterparts
        # We assume the naming convention: clock_{id}_{h}_{m}_{s}_digital.png
        self.digital_paths = sorted(glob.glob(os.path.join(root_dir, "*_digital.png")))
        
    def __len__(self):
        return len(self.digital_paths)

    def __getitem__(self, idx):
        # Path for digital image
        dig_path = self.digital_paths[idx]
        
        # Corresponding analog image path
        ana_path = dig_path.replace("_digital.png", "_analog.png")
        
        # Load images
        dig_img = Image.open(dig_path).convert("RGB")
        ana_img = Image.open(ana_path).convert("RGB")
        
        if self.transform:
            dig_img = self.transform(dig_img)
            ana_img = self.transform(ana_img)
            
        return dig_img, ana_img

def get_transforms(img_size=256):
    """Returns standard transforms for training."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1]
    ])
