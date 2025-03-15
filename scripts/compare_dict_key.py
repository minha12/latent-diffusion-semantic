import torch
from pathlib import Path

def compare_checkpoint_keys(ckpt1_path, ckpt2_path):
    # Load checkpoints
    print(f"Loading checkpoint 1: {ckpt1_path}")
    ckpt1 = torch.load(ckpt1_path, map_location='cpu')
    print(f"Loading checkpoint 2: {ckpt2_path}")
    ckpt2 = torch.load(ckpt2_path, map_location='cpu')

    # Extract state dictionaries
    if 'state_dict' in ckpt1:
        ckpt1_keys = set(ckpt1['state_dict'].keys())
    else:
        ckpt1_keys = set(ckpt1.keys())
    
    if 'state_dict' in ckpt2:
        ckpt2_keys = set(ckpt2['state_dict'].keys())
    else:
        ckpt2_keys = set(ckpt2.keys())

    # Compare keys
    print("\nKeys in kl-f8 but not in vq-f4:")
    for key in sorted(ckpt1_keys - ckpt2_keys):
        print(f"- {key}")

    print("\nKeys in vq-f4 but not in kl-f8:")
    for key in sorted(ckpt2_keys - ckpt1_keys):
        print(f"- {key}")

    print("\nCommon keys:")
    common_keys = sorted(ckpt1_keys & ckpt2_keys)
    print(f"Total common keys: {len(common_keys)}")

if __name__ == "__main__":
    ckpt1_path = "models/first_stage_models/kl-f8/model.ckpt"
    ckpt2_path = "models/first_stage_models/vq-f4/model.ckpt"

    if not Path(ckpt1_path).exists():
        print(f"Error: {ckpt1_path} does not exist")
    elif not Path(ckpt2_path).exists():
        print(f"Error: {ckpt2_path} does not exist")
    else:
        compare_checkpoint_keys(ckpt1_path, ckpt2_path)