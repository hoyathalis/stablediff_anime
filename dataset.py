import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms

class AnimeDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Create the dataset
image_dir = './images'
dataset = AnimeDataset(image_dir=image_dir, transform=transform)

# Create the dataloader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Example usage
if __name__ == "__main__":
    for images in dataloader:
        print(images.size())
        break