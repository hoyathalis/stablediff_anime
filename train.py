import os
import torch
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model import UNet, NoiseScheduler  # Ensure your UNet model and NoiseScheduler are correctly implemented
from dataset import AnimeDataset
import wandb
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import subprocess
import atexit
import optuna

# Hyperparameters
log_dir = 'C:/Users/PC/OneDrive/Desktop/animediff/logs'
image_dir = './images'
val_split = 0.1
patience = 5

def tensor_to_image_grid(tensor_grid):
    grid = tensor_grid.permute(1, 2, 0)
    grid = (grid * 0.5 + 0.5) * 255
    grid = grid.cpu().numpy()
    grid = np.clip(grid, 0, 255).astype(np.uint8)
    return Image.fromarray(grid)

def objective(trial):
    # Initialize a new wandb run for each trial
    wandb.init(project="StableDiffusion", entity="hoyathaliaezakmi", reinit=True, name=f"lr_{trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)}_bs_{trial.suggest_categorical('batch_size', [32, 64, 128])}_epochs_{trial.suggest_int('num_epochs', 100, 150)}_is_{trial.suggest_categorical('image_size', [64])}_if_{trial.suggest_categorical('init_features', [16, 32, 64])}_dp_{trial.suggest_uniform('dropout_prob', 0.0, 0.5)}_act_{trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu'])}")

    # Hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    num_epochs = trial.suggest_int('num_epochs', 100, 150)
    image_size = trial.suggest_categorical('image_size', [64])
    init_features = trial.suggest_categorical('init_features', [16, 32, 64])
    dropout_prob = trial.suggest_uniform('dropout_prob', 0.0, 0.5)
    activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu'])

    # Initialize noise scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    noise_scheduler = NoiseScheduler(num_timesteps=200, device=device)

    # Create the dataset and dataloaders
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = AnimeDataset(image_dir=image_dir, transform=transform)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize model, loss function, optimizer, and TensorBoard writer
    model = UNet(in_channels=3, out_channels=3, init_features=init_features, dropout_prob=dropout_prob, activation=activation).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter(log_dir=log_dir)

    # Training loop with early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device)
            t = torch.randint(0, noise_scheduler.num_timesteps, (images.size(0),), device=device).long()
            noisy_images, noise = noise_scheduler.add_noise(images, t)
            optimizer.zero_grad()
            outputs = model(noisy_images, t)
            loss = criterion(outputs, noise)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Log training loss to wandb more often
            wandb.log({"Train loss (batch)": loss.item()})

        train_loss /= len(train_loader)
        writer.add_scalar('Loss/train', train_loss, epoch)
        wandb.log({"Train loss (epoch)": train_loss})

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                t = torch.randint(0, noise_scheduler.num_timesteps, (images.size(0),), device=device).long()
                noisy_images, noise = noise_scheduler.add_noise(images, t)
                outputs = model(noisy_images, t)
                loss = criterion(outputs, noise)
                val_loss += loss.item()

                # Log validation loss to wandb more often
                wandb.log({"Val loss (batch)": loss.item()})

        val_loss /= len(val_loader)
        writer.add_scalar('Loss/val', val_loss, epoch)
        wandb.log({"Val loss (epoch)": val_loss})

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_unet_model.pth')
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # Inference on random noise after each epoch
        model.eval()
        with torch.no_grad():
            # Generate random noise
            random_noise = torch.randn(16, 3, image_size, image_size, device=device)
            # Normalize the noise to match the training data normalization
            random_noise = (random_noise - random_noise.mean(dim=[1,2,3], keepdim=True)) / random_noise.std(dim=[1,2,3], keepdim=True)
            # Denoise the random noise using iterative denoising
            denoised_random = noise_scheduler.iterative_denoise(random_noise, model, steps=noise_scheduler.num_timesteps)
            #print(denoised_random)
            # Create grids for visualization
            noise_grid = vutils.make_grid(random_noise, nrow=4, normalize=True, scale_each=True)
            denoised_grid = vutils.make_grid(denoised_random, nrow=4, normalize=True, scale_each=True)

            # Convert tensor grids to human-readable images
            noise_image = tensor_to_image_grid(noise_grid)
            denoised_image = tensor_to_image_grid(denoised_grid)

            # Log the image grids to wandb
            wandb.log({
                "Random Noise Grid": wandb.Image(noise_image, caption="Random Noise"),
                "Denoised Random Images Grid": wandb.Image(denoised_image, caption="Denoised Random Images")
            })

        # Inference and visualization on validation set after each epoch
        with torch.no_grad():
            # Select a batch of images from the validation set
            val_batch = next(iter(val_loader))[:16].to(device)
            # Add noise to the validation images
            t_val = torch.randint(0, noise_scheduler.num_timesteps, (val_batch.size(0),), device=device).long()
            noisy_val_images, noise = noise_scheduler.add_noise(val_batch, t_val)
            # Predict the noise using the model
            predicted_noise = model(noisy_val_images, t_val)
            # Remove the predicted noise to get the denoised images
            denoised_val = noisy_val_images - predicted_noise

            # Create grids for visualization
            original_grid = vutils.make_grid(val_batch, nrow=4, normalize=True, scale_each=True)
            noisy_val_grid = vutils.make_grid(noisy_val_images, nrow=4, normalize=True, scale_each=True)
            denoised_val_grid = vutils.make_grid(denoised_val, nrow=4, normalize=True, scale_each=True)

            # Convert tensor grids to human-readable images
            original_image = tensor_to_image_grid(original_grid)
            noisy_val_image = tensor_to_image_grid(noisy_val_grid)
            denoised_val_image = tensor_to_image_grid(denoised_val_grid)

            # Log the image grids to wandb
            wandb.log({
            "Validation Noisy Images Grid": wandb.Image(noisy_val_image, caption="Validation Noisy Images"),
            "Validation Denoised Images Grid": wandb.Image(denoised_val_image, caption="Validation Denoised Images")
            })

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    # Save the final model checkpoint
    torch.save(model.state_dict(), 'final_unet_model.pth')
    writer.close()

    # Return the best validation loss for optuna
    return best_val_loss

if __name__ == '__main__':
    # Create a lock file to prevent multiple instances
    lock_file = 'train.lock'
    if os.path.exists(lock_file):
        print("Another instance of the script is already running.")
        exit(1)
    else:
        open(lock_file, 'w').close()

    # Ensure the lock file is removed when the script exits
    atexit.register(lambda: os.remove(lock_file))

    # Run hyperparameter optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    print("Best hyperparameters: ", study.best_params)
    print("Best validation loss: ", study.best_value)

    # Launch TensorBoard in a non-blocking way
    subprocess.Popen(['tensorboard', '--logdir', log_dir])