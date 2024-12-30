import torch
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
from PIL import Image

class NoiseScheduler:
    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02, device="cuda"):
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        # Sinusoidal noise schedule
        betas = self._get_sinusoidal_beta_schedule().to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def _get_sinusoidal_beta_schedule(self):
        """Creates sinusoidal beta schedule"""
        t = torch.linspace(0, self.timesteps - 1, self.timesteps)
        w = 2.0 * np.pi / self.timesteps
        betas = torch.cos(w * t) * (self.beta_end - self.beta_start) / 2 + (self.beta_end + self.beta_start) / 2
        return torch.clip(betas, self.beta_start, self.beta_end)

    def add_noise(self, x, t):
        """Add noise to input at timestep t"""
        x = x.to(self.device)
        noise = torch.randn_like(x, device=self.device)
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        return sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * noise, noise

    def remove_noise(self, x, t, predicted_noise):
        """Remove noise from input at timestep t using model's predicted noise"""
        x = x.to(self.device)
        predicted_noise = predicted_noise.to(self.device)
        alpha_cumprod = self.alphas_cumprod[t].reshape(-1, 1, 1, 1)
        beta = self.betas[t].reshape(-1, 1, 1, 1)
        
        if t > 0:
            noise = torch.randn_like(x, device=self.device)
        else:
            noise = torch.zeros_like(x, device=self.device)
            
        mean = (1 / torch.sqrt(self.alphas[t])) * (x - (beta * predicted_noise) / 
               torch.sqrt(1 - self.alphas_cumprod[t]))
        
        if t > 0:
            variance = beta * noise
        else:
            variance = 0
            
        x = mean + variance
        return x

    @torch.no_grad()
    def sample(self, model, batch_size=1, img_size=64, channels=3):
        model.eval()
        x = torch.randn(batch_size, channels, img_size, img_size, device=self.device)
        
        for t in reversed(range(self.timesteps)):
            timesteps = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            # Convert to float and expand dimensions for time embedding
            timesteps = timesteps.float().unsqueeze(-1).expand(-1, model.time_dim)
            predicted_noise = model(x, timesteps)
            x = self.remove_noise(x, t, predicted_noise)
            x = torch.clamp(x, -1.0, 1.0)
            
        model.train()
        return x

    def save_samples(self, samples, save_path="samples.png", denormalize=True):
        """Save generated samples as images"""
        if denormalize:
            samples = (samples + 1) / 2  # [-1, 1] -> [0, 1]
        save_image(samples.cpu(), save_path, nrow=int(np.sqrt(len(samples))))

def test_scheduler():
    # Test parameters
    batch_size = 2
    img_size = 64
    channels = 3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        scheduler = NoiseScheduler(device=device)
        
        # Test noise addition
        x = torch.randn(batch_size, channels, img_size, img_size, device=device)
        t = torch.randint(0, scheduler.timesteps, (batch_size,), device=device)
        noisy_x, noise = scheduler.add_noise(x, t)
        
        assert noisy_x.shape == x.shape, f"Shape mismatch: {noisy_x.shape} vs {x.shape}"
        assert not torch.isnan(noisy_x).any(), "NaN values in noisy output"
        
        # Test beta schedule
        assert torch.all(scheduler.betas >= scheduler.beta_start)
        assert torch.all(scheduler.betas <= scheduler.beta_end)
        
        print("Scheduler validation passed!")
        return True
        
    except Exception as e:
        print(f"Scheduler validation failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_scheduler()