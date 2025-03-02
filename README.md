# Offical Code Implementation for MedGRAF
MedGRAF: Sparse View X-ray Generative Radiance Field with Multi-scale Sampling and Medical Augmentation
## Table of Contents
- [Overview](#overview)
- [Environment Setup](#environment-setup)
  - [Create and Activate a Conda Virtual Environment](#create-and-activate-a-conda-virtual-environment)
  - [Install Dependencies](#install-dependencies)
  - [Install PyTorch-GPU](#install-pytorch-gpu)
    - [Check CUDA Version](#check-cuda-version)
    - [Install CUDA](#install-cuda)
    - [Install PyTorch-GPU](#install-pytorch-gpu-1)
  - [Verify PyTorch-GPU Installation](#verify-pytorch-gpu-installation)

---

## Overview
This section provides a detailed guide on how to install the dependencies required for the project, as well as PyTorch-GPU. including creating a virtual environment, installing dependencies, and configuring GPU support. Follow each step carefully to ensure all components are installed and configured correctly.

---

## Environment Setup

### Create and Activate a Conda Virtual Environment

1. Clone the `mednerf` project from GitHub:
   ```bash
   git clone https://github.com/abrilcf/mednerf.git

2. Create a virtual environment (you can customize the environment name; here, we use `graf`):
   ```bash
   conda create --name graf python=3.10
   ```

3. Activate the virtual environment:
   ```bash
   conda activate graf
   ```

---

### Install Dependencies

Run the view reconstruction script and install the required dependencies based on any errors encountered:

1. Install `ignite`:
   ```bash
   pip install ignite
   ```

2. Install `pytorch-ignite`:
   ```bash
   pip install pytorch-ignite
   ```

3. Install `torchvision`:
   ```bash
   pip install torchvision
   ```

4. Install `tqdm`:
   ```bash
   pip install tqdm
   ```

5. Install `opencv-python`:
   ```bash
   pip install opencv-python
