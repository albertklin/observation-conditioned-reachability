# Observation-Conditioned Reachability (OCR) (code release in-progress)

This repository hosts the code used in [this project](https://sia-lab-git.github.io/One_Filter_to_Deploy_Them_All/).

## TODO:
1. ~~upload OCR-VN training and evaluation code~~
2. ~~upload simulation experiment code~~
3. upload hardware experiment code (TODO July)

## Setup

Install packages in a Python3.8 virtual environment.

`python3.8 -m venv env`

`source env/bin/activate`

`pip install --upgrade pip`

`pip install "jax[cuda11_pip]==0.4.13" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`

`pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118`

`pip install --upgrade hj-reachability`

`pip install -e libraries/walk-these-ways`

`pip install -e libraries/BreezySLAM/python`

`pip install -r requirements.txt`

Install IsaacGym

`wget https://developer.nvidia.com/isaac-gym-preview-4`

`tar -xf isaac-gym-preview-4 -C libraries`

`rm isaac-gym-preview-4`

`pip install -e libraries/isaacgym/python`

Test your isaacgym package installation with:

`cd libraries/isaacgym/python/examples`

`python joint_monkey.py`

For systems with both Intel and NVIDIA GPUs, you might need to run:

`sudo prime-select nvidia`

`export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`

## Test Your Python Environment

`python scripts/test_python_environment.py`

## Generate Data for OCR-VN Training

`python scripts/generate_value_network_data.py`

Generating the training and validation dataset should take ~30 minutes on the latest NVIDIA GPU models. If you encounter `RuntimeError: Unable to load cuSPARSE. Is it installed?` from the Jax library, you may need to first run `unset LD_LIBRARY_PATH`

If you get a JAX CUDA error on a GPU with <20GB, it could be due to memory issues. Try reducing the `state_grid_shape` on Line 43.

## Train the OCR-VN

`python scripts/train_value_network.py`

Training the OCR-VN on the latest NVIDIA GPU models should take ~8 hours.

## Download a Trained Checkpoint Model from Hugging Face

`huggingface-cli login` OR set environment variable `HUGGINGFACE_TOKEN`

Get access to the project here: https://huggingface.co/datasets/albertkuilin/observation-conditioned-reachability

`python scripts/sync_data.py pull results results`

## Inspect the OCR-VN Prediction In a Validation Environment

`python scripts/visualize_value_network_single_lidar.py` - predictions across the x-y space using the centroidal LiDAR scan

`python scripts/visualize_value_network_multi_lidar.py` - predictions across the x-y space using the local LiDAR scan

## Generate Simulation Environments

`python scripts/generate_sims.py`

You can also pull simulation environments with `python scripts/sync_data.py pull data data`

These correspond to what is described in the paper as "hard" environments, where large dynamical uncertainty is introduced by large variations in the floor friction.

## Run the OCR Safety Filter In Simulation

`python scripts/run_sims.py`

Analyze the results with `python scripts/analyze_sims.py`