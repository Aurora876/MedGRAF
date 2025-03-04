# Offical Code Implementation for MedGRAF
MedGRAF: Sparse View X-ray Generative Radiance Field with Multi-scale Sampling and Medical Augmentation

# Project Setup and Usage Guide

This guide provides instructions for setting up the environment, obtaining data and models, and using the project for training, evaluation, and view reconstruction.

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
Here is the content formatted as a GitHub README in English:

```markdown
# Project Setup and Usage Guide

This guide provides instructions for setting up the environment, obtaining data and models, and using the project for training, evaluation, and view reconstruction.

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
   ```

2. Activate the `xxx` environment:
   ```bash
   conda activate xxx
   ```

---

## Data Acquisition

You can obtain all the data through [this link](#). The dataset includes:
- **Chest Dataset**: Contains X instances of images.
- **Knee Dataset**: Contains Y instances of images.

---

## Model Acquisition

You can obtain the pre-trained models through [this link](#). The models include:
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
python render_xray_G_mod.py configs/knee.yaml --xray_img_path /home/zd/jzd/new/graf/data/render1_360 --save_dir /home/zd/jzd/new/graf/results/baseline/test1 --model /home/zd/jzd/new/graf/results/baseline/ckpt/model.pt
```

---

## Acknowledgments

Our code is heavily inspired by the codebase from [this repository](https://github.com/autonomousvision/graf).
```

---

### Key Features of the README:
1. **Clear Structure**: Divided into sections for easy navigation.
2. **Code Formatting**: Commands are formatted as code blocks for clarity.
3. **Links**: Placeholder links are included for data and model acquisition.
4. **Acknowledgments**: Proper credit is given to the original codebase.

Let me know if you need further adjustments! 😊
