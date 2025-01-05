# Anime Face Diffusion Model

A lightweight implementation of a Stable Diffusion model trained from scratch on 64x64 anime face images. This project provides a complete pipeline for training and generating anime-style face images using diffusion models.

> Built as a learning exploration into diffusion models, this project focuses on understanding core diffusion concepts including training dynamics, scheduler behavior, and inference strategies. While the UNet architecture can be enhanced for higher quality outputs, the current implementation is designed to balance learning objectives with consumer GPU constraints. This hands-on approach provided valuable insights into the intricacies of diffusion model development.

## Generated Images Sample
![2](https://github.com/user-attachments/assets/d06aa732-79a4-427d-9cf1-00e9efcc0996)
![11](https://github.com/user-attachments/assets/d0cb7297-027d-437e-b06c-905f41d4a7d5)
![3](https://github.com/user-attachments/assets/a06dfed3-f5b3-49d5-9f9c-38837f3382e5)


![image](https://github.com/user-attachments/assets/8c081e0c-82f5-467c-8f49-271539a6d856)

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
