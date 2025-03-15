# Latent Diffusion Models for Semantic Segmentation
Source: [arXiv](https://arxiv.org/abs/2112.10752) | [BibTeX](#bibtex)

This repository contains code for training Latent Diffusion Models (LDM) using segmentation masks from the DRSK dataset.

<p align="center">
<img src=assets/modelfigure.png />
</p>

## Requirements
A suitable [conda](https://conda.io/) environment named `ldm` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate ldm
```

## Troubleshooting

### Environment Setup Issues

If you encounter dependency issues after setting up the conda environment, try the following fixes:

```bash
# Fix for torch metrics compatibility
pip install torchmetrics==0.5.0

# Install required repositories
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip
pip install -e .

# Fix for kornia
pip install kornia==0.5.1
```

### TensorBoard Issues

If you encounter problems with TensorBoard, try reinstalling with specific versions:

```bash
pip uninstall -y tensorboard protobuf
pip install tensorboard==2.11.0 protobuf==3.19.4
```

### Monitoring Training with TensorBoard

To monitor your training progress, run TensorBoard with:

```bash
nohup tensorboard --logdir ~/latent-diffusion-semantic/logs/ --host 0.0.0.0 --port 8088 &
```

Then access the TensorBoard interface in your browser at `http://<your-server-ip>:8088`

## DRSK Dataset Preparation

This repository is configured to work with the DRSK dataset available at:
[https://datahub.aida.scilifelab.se/10.23698/aida/drsk](https://datahub.aida.scilifelab.se/10.23698/aida/drsk)

### Dataset Structure

After downloading the DRSK dataset, organize it into the following structure:

```
~/datasets/drsk/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── segmentation/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── image_names.txt
```

Where:
- `images/` contains the original images
- `segmentation/` contains segmentation masks (PNG files)
- `image_names.txt` contains the list of all image filenames (one filename per line)

### Creating Train/Test Split

Run the provided script to split the dataset into training and evaluation sets:

```bash
bash scripts/split_dataset.sh
```

This will create:
- `image_names_train.txt`: Contains 90% of the images for training
- `image_names_eval.txt`: Contains 10% of the images for evaluation

## Training Configuration

The configuration for training the Latent Diffusion Model on DRSK dataset is provided in:
`models/ldm/drsk/config.yaml`

The model is set up with:
- Conditional generation using segmentation masks
- A VQ-regularized autoencoder as first stage (f=4, d=3)
- A UNet-based diffusion model in the latent space

### Training the Model

To start training the model:

```shell script
CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base models/ldm/drsk/config.yaml -t --gpus 0,
```

Logs and checkpoints for trained models are saved to `logs/<START_DATE_AND_TIME>_<config_spec>`.

### Sampling from the Model

After training, you can generate images conditioned on segmentation masks using:

```shell script
python scripts/sample_diffusion.py -r logs/<your_run>/checkpoints/last.ckpt -c 50 -n 4 --scale 1.0
```

## BibTeX

```
@misc{rombach2021highresolution,
      title={High-Resolution Image Synthesis with Latent Diffusion Models}, 
      author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
      year={2021},
      eprint={2112.10752},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@misc{https://doi.org/10.23698/aida/drsk,
  doi={10.23698/aida/drsk},
  url={https://datahub.aida.scilifelab.se/10.23698/aida/drsk},
  author={Nathan Löfmark and Viktor Holmgren and Hanna Dahlstrand and Martin Dahlö and Johan Christenson and Matías Araya and Martin Lindvall and Sven Nelander and Maria Häggman and Christian Lundberg and Ingrid Lönnstedt and Kevin Smith and Petter Ranefall and Carolina Wählby},
  title={Digital Pathology of Renal Cell Carcinoma with Survival Data},
  publisher={AIDA},
  year={2023}
}


