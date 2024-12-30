import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.linear_1 = nn.Linear(dim, dim * 4)
        self.linear_2 = nn.Linear(dim * 4, dim)
        
    def forward(self, x):
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_channels)
        self.transform = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
    def forward(self, x, t):
        residual = self.transform(x)
        
        x = self.conv1(x)
        x = self.norm1(x)
        x += self.time_mlp(t)[..., None, None]
        x = F.silu(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.silu(x)
        
        return x + residual

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_dim=256):
        super().__init__()
        
        # Time embedding
        self.time_embed = TimeEmbedding(time_dim)
        
        self.time_dim=time_dim
        # Initial conv
        self.init_conv = nn.Conv2d(in_channels, 64, 3, padding=1)
        
        # Encoder
        self.down1 = ResidualBlock(64, 128, time_dim)
        self.down2 = ResidualBlock(128, 256, time_dim)
        self.down3 = ResidualBlock(256, 512, time_dim)
        
        # Bottleneck
        self.bottleneck1 = ResidualBlock(512, 512, time_dim)
        self.bottleneck2 = ResidualBlock(512, 512, time_dim)
        
        # Decoder
        self.up1 = ResidualBlock(1024, 256, time_dim)
        self.up2 = ResidualBlock(512, 128, time_dim)
        self.up3 = ResidualBlock(256, 64, time_dim)
        
        # Final conv
        self.final_conv = nn.Conv2d(64, out_channels, 1)
        
        self.pools = nn.ModuleList([
            nn.MaxPool2d(2) for _ in range(3)
        ])
        
        self.upsamples = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) for _ in range(3)
        ])
        
    def forward(self, x, timestep):
        # Time embedding
        if timestep.dim() == 1:
            timestep = timestep.unsqueeze(-1).expand(-1, self.time_dim)
        
        # Time embedding
        t = self.time_embed(timestep)        
        # Initial conv
        x0 = self.init_conv(x)
        
        # Encoder
        x1 = self.down1(x0, t)
        x1_pool = self.pools[0](x1)
        
        x2 = self.down2(x1_pool, t)
        x2_pool = self.pools[1](x2)
        
        x3 = self.down3(x2_pool, t)
        x3_pool = self.pools[2](x3)
        
        # Bottleneck
        x3_pool = self.bottleneck1(x3_pool, t)
        x3_pool = self.bottleneck2(x3_pool, t)
        
        # Decoder with skip connections
        x = self.upsamples[0](x3_pool)
        x = torch.cat([x, x3], dim=1)
        x = self.up1(x, t)
        
        x = self.upsamples[1](x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x, t)
        
        x = self.upsamples[2](x)
        x = torch.cat([x, x1], dim=1)
        x = self.up3(x, t)
        
        return self.final_conv(x)

def test_unet():
    # Test parameters
    batch_size = 4
    channels = 3
    height = 64
    width = 64
    time_dim = 256
    
    # Create model
    model = UNet(in_channels=channels, out_channels=channels, time_dim=time_dim)
    
    # Create dummy inputs
    x = torch.randn(batch_size, channels, height, width)
    t = torch.randn(batch_size, time_dim)
    
    # Forward pass
    try:
        output = model(x, t)
        
        # Validate output shape
        expected_shape = (batch_size, channels, height, width)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        
        # Validate no NaN values
        assert not torch.isnan(output).any(), "Output contains NaN values"
        
        print("Model validation passed!")
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        
        # Calculate number of parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"Model validation failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_unet()