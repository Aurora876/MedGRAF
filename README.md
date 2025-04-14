# Offical Code Implementation for MedGRAF
MedGRAF: Sparse-view X-ray Generative Radiance Field via Multi-scale Sampling and Medical Augmentation
#![GitHub Logo](https://github.com/user-attachments/assets/01ea7238-873b-4048-b209-bf0e5a1c0dbe)

# Project Setup and Usage Guide

This guide provides instructions for setting up the environment, obtaining data and models, and using the project for training, evaluation, and view reconstruction.The code will be uploaded progressively after modification and organization.

---

## Table of Contents
- [Installation](#installation)
- [Data Acquisition](#data-acquisition)
- [Model Acquisition](#model-acquisition)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Model Evaluation](#model-evaluation)
  - [View Reconstruction](#view-reconstruction)

---

## Installation

Create and activate a virtual environment, then install PyTorch-GPU and its dependencies.

1. Create a virtual environment named `xxx`:
   ```bash
   conda create --name xxx python=3.10


2. Activate the `xxx` environment:
   ```bash
   conda activate xxx
   ```

---

## Data Acquisition

You can obtain all the data through [this link](https://drive.google.com/drive/folders/12l-HJ6vH4xFLtd9z6WFqwSwCt4gZvlJd?usp=sharing). The dataset includes:
- **Chest Dataset**: Contains 1440 instances of images.
- **Knee Dataset**: Contains 360 instances of images.

---

## Model Acquisition

You can obtain the pre-trained models through [this link](https://drive.google.com/drive/folders/12l-HJ6vH4xFLtd9z6WFqwSwCt4gZvlJd?usp=sharing). The models include:
- **Chest Model**
- **Knee Model**

---

## Usage

Once the project environment is set up, you can train the model, evaluate the model, and perform view reconstruction.

### Training the Model

To train the model, run the following command:
```bash
python train.py configs/knee.yaml
```
- The `train.py` script is used to train the model.
- The configuration parameters for the model are defined in the `configs/knee.yaml` file.

### Model Evaluation

To evaluate the model, run the following command:
```bash
python eval.py configs/knee.yaml --pretrained --fid_kid --rotation_elevation --shape_appearance --reconstruction
```

### View Reconstruction

After training the model with the data, you can generate a complete 360-degree X-ray result by providing one or multiple X-ray images generated using the DRR method.

Run the following command for view reconstruction:
```bash
python render_xray_G_mod.py configs/experiment.yaml /
    --xray_img_path path_to_xray_folder /
    --save_dir ./renderings /
    --model path_to_trained_model/model_best.pt /
```

---

## Acknowledgments

Our code is heavily inspired by the codebase from [this repository](https://github.com/autonomousvision/graf).
```
