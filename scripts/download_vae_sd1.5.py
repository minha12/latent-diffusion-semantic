import os
import torch
from diffusers import StableDiffusionPipeline

def download_and_convert_vae():
    print("Downloading and converting the VAE from sd-legacy/stable-diffusion-v1-5...")
    
    # Create directory if it doesn't exist
    save_dir = "models/first_stage_models/kl-f8"
    os.makedirs(save_dir, exist_ok=True)
    
    # Download VAE through diffusers
    pipe = StableDiffusionPipeline.from_pretrained(
        "sd-legacy/stable-diffusion-v1-5",
        #torch_dtype=torch.float32,
        components=["vae"]
    )
    vae = pipe.vae
    
    # Get the state dictionary and save in format needed for LDM
    diffusers_state_dict = vae.state_dict()
    ckpt_path = os.path.join(save_dir, "model.ckpt")
    torch.save({"state_dict": diffusers_state_dict}, ckpt_path)
    
    print("You can now use this VAE in your configuration at: models/first_stage_models/kl-f8/model.ckpt")

if __name__ == "__main__":
    download_and_convert_vae()