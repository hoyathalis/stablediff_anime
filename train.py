import torch
from torch.optim import Adam
from dataset import get_dataloaders
from model import UNet
from scheduler import NoiseScheduler
import torch.nn.functional as F
import wandb
from tqdm import tqdm
import os

def train_diffusion(
    image_dir="images",
    epochs=500,
    batch_size=32,
    lr=1e-4,
    img_size=64,
    device=None,
    model_save_path="best_model.pth",
    patience=5
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb.init(project="stablediffusion", name="diffusion-training")
    
    train_loader, val_loader = get_dataloaders(image_dir, batch_size=batch_size, img_size=img_size,validation_split=0.1)
    model = UNet(in_channels=3, out_channels=3, time_dim=256).to(device)
    
    # Load the best model if it exists
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        print(f"Loaded model from {model_save_path}")

    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = NoiseScheduler(device=device)
    wandb.watch(model, log="all")
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        epoch_loss = 0
        for i, real_images in enumerate(tqdm(train_loader, desc="Training Batches", leave=False)):
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

        epoch_loss /= len(train_loader)
        wandb.log({"epoch": epoch + 1, "epoch_loss": epoch_loss})
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for real_images in tqdm(val_loader, desc="Validation Batches", leave=False):
                real_images = real_images.to(device)
                t = torch.randint(0, 256, (real_images.size(0),), device=device)
                noisy, noise = scheduler.add_noise(real_images, t)
                predicted_noise = model(noisy, t.float())
                loss = F.mse_loss(predicted_noise, noise)
                val_loss += loss.item()
                wandb.log({"val_batch_loss": loss.item()})

        val_loss /= len(val_loader)
        wandb.log({"epoch": epoch + 1, "val_epoch_loss": val_loss})
        print(f"Epoch {epoch+1}/{epochs} | Validation Loss: {val_loss:.4f}")

        # Early stopping and model saving
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model to {model_save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        # Infer a sample batch of random noises
        with torch.no_grad():
            samples = scheduler.sample(model, batch_size=1, img_size=img_size, channels=3)
            scheduler.save_samples(samples, save_path=f"sample_epoch_{epoch+1}.png")
            wandb.log({"generated_image": [wandb.Image(s, caption=f"Sample Epoch {epoch+1}") for s in samples]})

if __name__ == "__main__":
    train_diffusion()