# Pointcept Integration Requirements

This document outlines the hardware, software, and CUDA requirements for running the Pointcept library and its dependencies, particularly for GPU-accelerated operations like `pointops`.

## Hardware Requirements

- **GPU:** A powerful NVIDIA GPU is highly recommended for training and inference. For specific models like PTv2, a GPU with at least 24GB of memory is advisable.
- **RAM:** A sufficient amount of RAM is needed, especially for handling large point cloud datasets. 16GB or more is a good starting point.

## Software Requirements

- **Operating System:** Ubuntu 18.04 or a later version.
- **Python:** While a specific version is not mentioned in the official documentation, using Python 3.8 or later is recommended to ensure compatibility with all dependencies.
- **NVIDIA CUDA:**
    - **CUDA Version:** 11.3 or higher.
    - **FlashAttention:** For using the PTv3 model with FlashAttention, CUDA 11.6 or newer is required.

## Dependencies

The following dependencies are required for a full installation of Pointcept. It is recommended to use a virtual environment (like conda or venv) to manage these packages.

### Core Dependencies

- **PyTorch:** Version 1.10.0 or newer. The official documentation provides an example with version 1.12.1.
- **Ninja:** Required for building custom PyTorch extensions.

### Python Packages

The following packages can be installed via `pip` or `conda`.

**Using Conda:**

It is recommended to install PyTorch and its related packages from the `pytorch` channel in conda for better compatibility.

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

Other packages:

```bash
conda install h5py pyyaml sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor -c anaconda
conda install timm -c conda-forge
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg
```

**Using Pip:**

```bash
pip install torch-geometric spconv-cu113 ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

### Additional Libraries

For specific models or features, you might need to install extra libraries:

- **Torch-Points3D:** For the Stratified Transformer model.
- **MinkowskiEngine:** For SparseUNet models.

Please refer to the official Pointcept repository for the most up-to-date and detailed installation instructions.
