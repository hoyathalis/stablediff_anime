import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, image_dir, img_size=64, train=True):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.img_size = img_size
        self.train = train
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

def get_dataloader(image_dir, batch_size=32, img_size=64, num_workers=4, train=True):
    dataset = ImageDataset(image_dir, img_size=img_size, train=train)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    return dataloader

def test_dataloader():
    # Test parameters
    image_dir = "./images"
    batch_size = 4
    img_size = 64

    try:
        # Create test directory and sample image if not exists
        os.makedirs(image_dir, exist_ok=True)
        if not os.listdir(image_dir):
            # Create dummy image
            dummy_image = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            dummy_image.save(os.path.join(image_dir, "test.png"))

        # Test dataloader
        dataloader = get_dataloader(image_dir, batch_size, img_size)
        
        # Get one batch
        batch = next(iter(dataloader))
        
        # Validate batch shape
        expected_shape = (batch_size, 3, img_size, img_size)
        assert batch.shape == expected_shape, f"Expected shape {expected_shape}, got {batch.shape}"
        
        # Validate pixel range after normalization
        assert -1.5 <= batch.min() <= -0.5, f"Minimum pixel value out of range: {batch.min()}"
        assert 0.5 <= batch.max() <= 1.5, f"Maximum pixel value out of range: {batch.max()}"
        
        print("Dataloader validation passed!")
        print(f"Batch shape: {batch.shape}")
        return True
        
    except Exception as e:
        print(f"Dataloader validation failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_dataloader()