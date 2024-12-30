import os
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from dataset import AnimeDataset  # Make sure you have the dataset module correctly defined
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np


class NoiseScheduler:
    def __init__(self, num_timesteps=1000, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        self.betas = 0.02 * (1 - torch.cos(torch.linspace(0, torch.pi, num_timesteps, device=self.device))) / 2
        self.alphas = 1. - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x, t):
        x = x.to(self.device)
        t = t.to(self.device)
        noise = torch.randn_like(x, device=self.device)
        t_scaled = t.float() / self.num_timesteps
        sin_weight = torch.sin(t_scaled * torch.pi)[:, None, None, None]
        sqrt_alpha = torch.sqrt(self.alpha_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alpha = torch.sqrt(1. - self.alpha_cumprod[t])[:, None, None, None]
        return sqrt_alpha * x + sin_weight * sqrt_one_minus_alpha * noise, noise

    def remove_noise(self, x, t, noise):
        x = x.to(self.device)
        t = t.to(self.device)
        noise = noise.to(self.device)
        t_scaled = t.float() / self.num_timesteps
        sin_weight = torch.sin(t_scaled * torch.pi)[:, None, None, None]
        sqrt_alpha = torch.sqrt(self.alpha_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alpha = torch.sqrt(1. - self.alpha_cumprod[t])[:, None, None, None]
        return (x - sin_weight * sqrt_one_minus_alpha * noise) / sqrt_alpha

    def iterative_denoise(self, noisy_x, t_initial, steps=10):
        denoised_x = noisy_x.clone()
        current_t = t_initial.clone()
        for step in range(steps):
            if current_t.item() <= 0:
                break
            denoised_x = self.remove_noise(denoised_x, current_t, noise=None)  
            current_t = current_t - 1
        return denoised_x
    
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
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Initialize the noise scheduler
num_timesteps = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'
scheduler = NoiseScheduler(num_timesteps, device=device)  # Use 'cuda' if available

# Create logs directory if it doesn't exist
log_dir = './logs'
os.makedirs(log_dir, exist_ok=True)

# Generate and save 10 images with noise
noisy_images = []
noises = []
timestamps = []
for i, image in enumerate(dataloader):
    if i >= 10:
        break

    t = torch.randint(195, num_timesteps, (1,), device=device).long()  # Generate timestamp on the same device
    noisy_image, noise = scheduler.add_noise(image, t)
    noisy_images.append(noisy_image.cpu())  # Move to CPU for further processing
    noises.append(noise.cpu())  # Store the noise separately
    timestamps.append(t.item())

# Create a grid of noisy images
noisy_images_tensor = torch.cat(noisy_images, dim=0)
grid = vutils.make_grid(noisy_images_tensor, nrow=5, normalize=True, scale_each=True)

# Convert tensor to image
grid = grid.permute(1, 2, 0)
grid = (grid * 0.5 + 0.5) * 255
grid = grid.cpu().numpy().astype(np.uint8)
grid_image = Image.fromarray(grid)

# Add labels to the image
draw = ImageDraw.Draw(grid_image)
font = ImageFont.load_default()
for i, t in enumerate(timestamps):
    x = (i % 5) * 64
    y = (i // 5) * 64
    draw.text((x + 5, y + 5), f't={t}', (255, 255, 255), font=font)  # Offset text slightly for visibility

# Save the grid image with timestamp
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
grid_image.save(os.path.join(log_dir, f'noisy_images_grid_{timestamp_str}.png'))

print("Generated and saved a grid of 10 noisy images with timestamps in the logs folder.")

# Test denoising on the generated noisy images
denoised_images = []
for i in range(len(noisy_images)):
    noisy_image = noisy_images[i]
    noise = noises[i]  # Retrieve the corresponding noise
    t = torch.tensor([timestamps[i]], device=device).long()  # Generate timestamp on the same device
    denoised_image = scheduler.remove_noise(noisy_image.to(device), t, noise.to(device))
    denoised_images.append(denoised_image.cpu())  # Move to CPU for further processing

# Create a grid of denoised images
denoised_images_tensor = torch.cat(denoised_images, dim=0)
denoised_grid = vutils.make_grid(denoised_images_tensor, nrow=5, normalize=True, scale_each=True)

# Convert tensor to image
denoised_grid = denoised_grid.permute(1, 2, 0)
denoised_grid = (denoised_grid * 0.5 + 0.5) * 255
denoised_grid = denoised_grid.cpu().numpy().astype(np.uint8)
denoised_grid_image = Image.fromarray(denoised_grid)

# Add labels to the denoised image
draw_denoised = ImageDraw.Draw(denoised_grid_image)
for i, t in enumerate(timestamps):
    x = (i % 5) * 64
    y = (i // 5) * 64
    draw_denoised.text((x + 5, y + 5), f't={t}', (255, 255, 255), font=font)  # Offset text slightly for visibility

# Save the denoised grid image with timestamp
denoised_grid_image.save(os.path.join(log_dir, f'denoised_images_grid_{timestamp_str}.png'))

print("Generated and saved a grid of 10 denoised images with timestamps in the logs folder.")
