import torch
from torch.optim import Adam
from dataset import get_dataloader
from model import UNet
from scheduler import NoiseScheduler
import torch.nn.functional as F
import wandb
from tqdm import tqdm
import os

def train_diffusion(
    image_dir="images",
    epochs=1,
    batch_size=4,
    lr=1e-4,
    img_size=64,
    device=None,
    model_save_path="best_model.pth"
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(project="stablediffusion", name="diffusion-training")
    
    dataloader = get_dataloader(image_dir, batch_size=batch_size, img_size=img_size, train=True)
    model = UNet(in_channels=3, out_channels=3, time_dim=256).to(device)
    
    # Load the best model if it exists
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        print(f"Loaded model from {model_save_path}")

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = NoiseScheduler(device=device)
    wandb.watch(model, log="all")
    
    best_loss = float('inf')
    
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        epoch_loss = 0
        for i, real_images in enumerate(tqdm(dataloader, desc="Training Batches", leave=False)):
            # if i >= 2000:
            #     break
            real_images = real_images.to(device)
            t = torch.randint(0, 256, (real_images.size(0),), device=device)
            noisy, noise = scheduler.add_noise(real_images, t)
            predicted_noise = model(noisy, t.float())
            loss = F.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            wandb.log({"batch_loss": loss.item()})

        epoch_loss /= len(dataloader)
        wandb.log({"epoch": epoch + 1, "epoch_loss": epoch_loss})
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")

        # Save the model if it has the best loss so far
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model to {model_save_path}")

        # Infer a sample batch of random noises
        with torch.no_grad():
            samples = scheduler.sample(model, batch_size=1, img_size=img_size, channels=3)
            scheduler.save_samples(samples, save_path=f"sample_epoch_{epoch+1}.png")
            wandb.log({"generated_image": [wandb.Image(s, caption=f"Sample Epoch {epoch+1}") for s in samples]})

if __name__ == "__main__":
    train_diffusion()