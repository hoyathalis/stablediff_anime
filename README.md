# Anime Face Diffusion Model

A lightweight implementation of a Stable Diffusion model trained from scratch on 64x64 anime face images. This project provides a complete pipeline for training and generating anime-style face images using diffusion models.

## Features

* Custom stable diffusion implementation from scratch
* Training pipeline optimized for 64x64 anime faces
* Configurable noise scheduling and diffusion steps
* Basic inference script for image generation
* Dataset handling for anime face images

## Installation

```bash
git clone https://github.com/yourusername/anime-face-diffusion.git
cd anime-face-diffusion
pip install -r requirements.txt
```

## Project Structure

```
├── dataset.py          # Dataset loading and preprocessing
├── dataset_test.py     # Dataset validation tests
├── model.py            # Core diffusion model architecture
├── scheduler.py        # Noise scheduling implementation
├── train.py           # Training pipeline
├── infer.py           # Image generation script
└── requirements.txt    # Project dependencies
```
