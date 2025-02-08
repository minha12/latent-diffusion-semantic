# Practical Guide: Training LDM with Segmentation Mask Conditioning

## Overview
This guide provides practical instructions for training a Latent Diffusion Model (LDM) with segmentation mask conditioning, based on successful implementation experience.

## 1. Dataset Preparation

### Using the SFLCKR Dataset
1. Collect Flickr data according to the taming-transformers repo instructions
2. Use the provided `sflckr.py` as the base dataset implementation

### Dataset Class Structure
```python
# Example from sflckr.py
class SegmentationBase(Dataset):
    def __init__(self, data_csv, data_root, segmentation_root, size=None, ...):
        # ... initialization code ...
        
    def __getitem__(self, i):
        # Returns dictionary with:
        # - "image": normalized image tensor [-1,1]
        # - "segmentation": segmentation mask tensor
        # ... implementation ...
```

## 2. Configuration

### Essential Config File (config.yaml)
```yaml
model:
  base_learning_rate: 1.0e-06
  target: ldm.models.diffusion.ddpm.LatentDiffusion
  params:
    linear_start: 0.0015
    linear_end: 0.0205
    log_every_t: 100
    timesteps: 1000
    loss_type: l1
    first_stage_key: image
    cond_stage_key: segmentation
    image_size: 64
    channels: 3
    concat_mode: true
    cond_stage_trainable: true

    scheduler_config:
      target: ldm.lr_scheduler.LambdaLinearScheduler
      params:
        warm_up_steps: [10000]
        cycle_lengths: [10000000000000]
        f_start: [1.e-6]
        f_max: [1.]
        f_min: [1.]

    unet_config:
      target: ldm.modules.diffusionmodules.openaimodel.UNetModel
      params:
        image_size: 64
        in_channels: 6  # 3 (image) + 3 (condition)
        out_channels: 3
        model_channels: 128
        attention_resolutions: [32, 16, 8]
        num_res_blocks: 2
        channel_mult: [1, 4, 8]
        num_heads: 8

    first_stage_config:
      target: ldm.models.autoencoder.VQModelInterface
      params:
        embed_dim: 3
        n_embed: 8192
        ckpt_path: models/first_stage_models/vq-f4/model.ckpt
        ddconfig:
          double_z: false
          z_channels: 3
          resolution: 256
          in_channels: 3
          out_ch: 3
          ch: 128
          ch_mult: [1, 2, 4]
          num_res_blocks: 2
          attn_resolutions: []
          dropout: 0.0
        lossconfig:
          target: torch.nn.Identity

    cond_stage_config:
      target: ldm.modules.encoders.modules.SpatialRescaler
      params:
        n_stages: 2
        in_channels: 182  # Number of segmentation classes
        out_channels: 3

data:
  target: main.DataModuleFromConfig
  params:
    batch_size: 12
    num_workers: 5
    wrap: False
    train:
      target: ldm.data.flickr.FlickrSegTrain
      params:
        size: 256
    validation:
      target: ldm.data.flickr.FlickrSegEval
      params:
        size: 256

lightning:
  callbacks:
    image_logger:
      target: main.ImageLogger
      params:
        batch_frequency: 5000
        max_images: 8
        increase_log_steps: False

  trainer:
    benchmark: True
```

## 3. Training

Run training using:
```bash
python main.py --base config.yaml -t --gpus 0,
```

## 4. Inference

Here's a complete inference script:

```python
import torch
import numpy as np
from scripts.sample_diffusion import load_model
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from einops import rearrange
from ldm.data.flickr import FlickrSegEval

def ldm_cond_sample(config_path, ckpt_path, dataset, batch_size):
    # Load model
    config = OmegaConf.load(config_path)
    model, _ = load_model(config, ckpt_path, None, None)

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Get batch of data
    x = next(iter(dataloader))
    seg = x['segmentation']

    with torch.no_grad():
        # Process segmentation mask
        seg = rearrange(seg, 'b h w c -> b c h w')
        condition = model.to_rgb(seg)

        seg = seg.to('cuda').float()
        seg = model.get_learned_conditioning(seg)

        # Generate samples
        samples, _ = model.sample_log(
            cond=seg,
            batch_size=batch_size,
            ddim=True,
            ddim_steps=200,
            eta=1.
        )

        # Decode samples
        samples = model.decode_first_stage(samples)

    # Save results
    save_image(condition, 'cond.png')
    save_image(samples, 'sample.png')

if __name__ == '__main__':
    config_path = 'models/ldm/semantic_synthesis256/config.yaml'
    ckpt_path = 'models/ldm/semantic_synthesis256/model.ckpt'
    dataset = FlickrSegEval(size=256)
    ldm_cond_sample(config_path, ckpt_path, dataset, 4)
```

## Key Points

1. **Model Architecture**
   - Uses VQModelInterface as the first stage model
   - SpatialRescaler for condition processing
   - UNet with concatenative conditioning

2. **Important Parameters**
   - `in_channels`: 6 (3 for image + 3 for condition)
   - `image_size`: 64 (latent space size)
   - `cond_stage_trainable`: true
   - `n_stages`: 2 (in SpatialRescaler)

3. **Inference**
   - Use DDIM sampling for faster generation
   - Default 200 steps for good quality
   - Proper mask reshaping is crucial

## Common Issues

1. **Segmentation Mask Format**
   - Ensure masks are properly one-hot encoded
   - Check channel ordering (use rearrange if needed)

2. **Memory Issues**
   - Adjust batch size based on GPU memory
   - Consider gradient accumulation for larger effective batches

3. **Training Stability**
   - Start with lower learning rate (1e-6)
   - Use warmup steps (10000 recommended)
   - Monitor loss curves for convergence
```
