import torch
from model import UNet
from scheduler import NoiseScheduler
import os

import torchvision.utils as vutils

def generate_images(model_path="best_model.pth", img_size=64, num_images=1, save_path="generated_grid.png"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = UNet(in_channels=3, out_channels=3, time_dim=256).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    scheduler = NoiseScheduler(device=device)
    
    with torch.no_grad():
        samples = scheduler.sample(model, batch_size=num_images, img_size=img_size, channels=3)
        scheduler.save_samples(samples, save_path=save_path)
        print(f"Saved generated images to {save_path}")

if __name__ == "__main__":
    generate_images()