# Observation-Conditioned Reachability (OCR) (code release in-progress)

This repository hosts the code used in [this project](https://sia-lab-git.github.io/One_Filter_to_Deploy_Them_All/).

## TODO:
1. ~~upload OCR-VN training and evaluation code~~
2. upload simulation experiment code (TODO July)
3. upload hardware experiment code (TODO July)

## Setup

Install packages in a Python3.12 virtual environment.

`python3.12 -m venv env`

`source env/bin/activate`

`pip install --upgrade pip`

`pip install -U "jax[cuda12]"`

`pip install --upgrade hj-reachability`

`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`

`pip install -r requirements.txt`

## Generate Data for OCR-VN Training

`python scripts/generate_value_network_data.py`

Generating the training and validation dataset should take ~30 minutes on the latest NVIDIA GPU models. If you encounter `RuntimeError: Unable to load cuSPARSE. Is it installed?` from the Jax library, you may need to first run `unset LD_LIBRARY_PATH`

## Train the OCR-VN

`python scripts/train_value_network.py`

Training the OCR-VN on the latest NVIDIA GPU models should take ~8 hours.

## Inspect the OCR-VN Prediction In a Validation Environment

`python scripts/visualize_value_network_single_lidar.py` - predictions across the x-y space using the centroidal LiDAR scan

`python scripts/visualize_value_network_multi_lidar.py` - predictions across the x-y space using the local LiDAR scan