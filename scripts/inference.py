import torch
import numpy as np
import os

from scripts.sample_diffusion import load_model
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from einops import rearrange

# Update the import to use the custom dataset
from ldm.data.drsk import DrskSegEval


def ldm_cond_sample(config_path, ckpt_path, dataset, batch_size, save_dir='outputs'):
    """
    Generate samples from a conditional latent diffusion model
    """
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load config and model
    config = OmegaConf.load(config_path)
    model, _ = load_model(config, ckpt_path, None, None)
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Get a batch of data
    x = next(iter(dataloader))
    
    # Extract segmentation
    seg = x['segmentation']
    
    print(f"Segmentation shape: {seg.shape}")
    
    with torch.no_grad():
        # Rearrange segmentation from [B, H, W, C] to [B, C, H, W]
        seg = rearrange(seg, 'b h w c -> b c h w')
        
        # Visualize condition
        if hasattr(model, 'to_rgb'):
            condition = model.to_rgb(seg)
            save_image(condition, os.path.join(save_dir, 'condition.png'))
        
        # Move to GPU and convert to float
        seg = seg.to('cuda').float()
        
        # Get learned conditioning
        cond = model.get_learned_conditioning(seg)
        
        # Sample from the model
        samples, _ = model.sample_log(
            cond=cond, 
            batch_size=batch_size, 
            ddim=True,
            ddim_steps=200, 
            eta=1.0
        )
        
        # Decode the samples
        samples = model.decode_first_stage(samples)
    
    # Save original segmentation visualization
    if hasattr(model, 'to_rgb'):
        save_image(condition, os.path.join(save_dir, 'segmentation.png'))
    
    # Save generated samples
    save_image(samples, os.path.join(save_dir, 'samples.png'))
    
    # If multiple samples, save them individually
    if batch_size > 1:
        for i, sample in enumerate(samples):
            save_image(sample, os.path.join(save_dir, f'sample_{i}.png'))


if __name__ == '__main__':
    # Update paths to match your configuration
    config_path = 'models/ldm/drsk/config-512-with-vq-f4.yaml'  # Path to your yaml file with model config
    
    # Use the checkpoint from your config
    ckpt_path = 'logs/2025-03-21T08-22-52_config-512-with-vq-f4/checkpoints/epoch=000099.ckpt'
    
    # Create your custom dataset with the appropriate size from your config
    dataset = DrskSegEval(size=512)  # Using size from your data config
    
    # Run the sampling function
    ldm_cond_sample(config_path, ckpt_path, dataset, batch_size=4)