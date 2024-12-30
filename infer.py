import torch
from model import UNet, NoiseScheduler
import os
import torchvision.utils as vutils
from PIL import Image
import numpy as np

# Hyperparameters
image_size = 64
num_images = 2
model_path = 'final_unet_model.pth'
output_dir = 'generated_images'

# Check if model exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file '{model_path}' not found. Please train the model first.")

# Initialize model and noise scheduler
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=3, out_channels=3, init_features=32).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

noise_scheduler = NoiseScheduler(num_timesteps=200, device=device)

# Generate random noise
random_noise = torch.randn(num_images, 3, image_size, image_size, device=device)
random_noise = (random_noise - random_noise.mean(dim=[1,2,3], keepdim=True)) / random_noise.std(dim=[1,2,3], keepdim=True)

# Denoise the random noise using iterative denoising
def stable_diffusion_denoise(noise, model, noise_scheduler):
    x = noise
    for t in reversed(range(noise_scheduler.num_timesteps)):
        t_tensor = torch.tensor([t], device=device).long()
        with torch.no_grad():
            predicted_noise = model(x, t_tensor)
        alpha_t = noise_scheduler.alphas[t]
        alpha_t_prev = noise_scheduler.alphas[t - 1] if t > 0 else torch.tensor(1.0, device=device)
        beta_t = noise_scheduler.betas[t]
        noise_scale = torch.sqrt(1 - alpha_t_prev)
        x = (x - beta_t * predicted_noise) / torch.sqrt(alpha_t) + noise_scale * torch.randn_like(x)
    return x

with torch.no_grad():
    denoised_images = stable_diffusion_denoise(random_noise, model, noise_scheduler)

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Save generated images
for i in range(num_images):
    denoised_image = denoised_images[i].permute(1, 2, 0).cpu().numpy()
    denoised_image = (denoised_image * 0.5 + 0.5) * 255
    denoised_image = np.clip(denoised_image, 0, 255).astype(np.uint8)
    img = Image.fromarray(denoised_image)
    img.save(os.path.join(output_dir, f'generated_image_{i+1}.png'))

print(f"Generated images saved to '{output_dir}'")
