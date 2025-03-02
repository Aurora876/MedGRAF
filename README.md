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
## Environment Setup
### Install Anaconda
Before installing PyTorch-GPU, you need to install Anaconda. Anaconda is a popular Python distribution that includes a wide range of libraries for scientific computing and data analysis.

1. Visit the Anaconda website: [Anaconda Download](https://www.anaconda.com/download).
2. Download and install Anaconda.
3. After installation, restart your computer.
4. Open the command line and run the following command to check the Anaconda version and confirm successful installation:
   ```bash
   conda -V
3. If the version number is displayed, Anaconda has been installed successfully.

### Install PyTorch-GPU

#### Check CUDA Version
Before installing PyTorch-GPU, confirm the CUDA version supported by your computer.

1. Open the command line (cmd) and run:
   ```bash
   nvidia-smi
   ```

2. Check the output to find the maximum supported CUDA version. For example, if it shows CUDA 12.8, you need to install a CUDA version less than or equal to 12.8.

#### Install CUDA
Download and install the appropriate CUDA toolkit version from the NVIDIA website based on your computer's supported CUDA version.

1. Visit the CUDA Toolkit Archive: [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive).
2. Download and install the CUDA version compatible with your system.
3. After installation, CUDA will be installed by default in the `C:\Program Files\NVIDIA GPU Computing Toolkit` directory.

#### Install PyTorch-GPU
You have now successfully installed PyTorch-GPU and its dependencies.
