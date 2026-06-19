# Prestack-Probabilistic-Inversion-JCMPC

Code for prestack multi-parameter probabilistic inversion integrating joint covariance modeling and physics constraints.

## Overview

This repository provides the source code and example workflow for prestack multi-parameter probabilistic inversion based on joint covariance modeling and physics constraints. The method is designed for the joint prediction of P-wave velocity, S-wave velocity, and density from prestack seismic data, while characterizing the statistical dependencies and joint uncertainty among elastic parameters.

The framework includes data preparation for the Marmousi2 model, prestack synthetic seismic-data generation, probabilistic inversion, prediction-result visualization, residual analysis, and noise robustness testing.

## Repository Structure

```text
Prestack-Probabilistic-Inversion-JCMPC/
│
├── marmousi2_crop.py
├── marmousi_test.py
├── marmousi_noise_test.py
├── README.md
│
├── figure/
│   ├── 图1.png
│   ├── 图2.png
│   ├── ...
│   └── 图9.png
│
└── model data/
    ├── data_crop/
    │   ├── marmousi2_crop_aux.mat
    │   ├── marmousi2_crop_mod4d.mat
    │   └── marmousi2_crop_stack4d_35hz.mat
    │
    ├── vp_marmousi-ii.segy
    ├── vs_marmousi-ii.segy
    └── density_marmousi-ii.segy
```

## Code Description

The main scripts are listed below.

```text
marmousi2_crop.py
```

This script is used to crop and prepare the Marmousi2 elastic-parameter models and generate the auxiliary model files required for the inversion experiment.

```text
marmousi_test.py
```

This script is used to perform the prestack multi-parameter probabilistic inversion experiment on the Marmousi2 model and generate prediction results, residual maps, validation plots, and related output files.

```text
marmousi_noise_test.py
```

This script is used to conduct noise robustness tests under different signal-to-noise ratio conditions.

## Data Description

The folder `model data/data_crop/` contains the processed Marmousi2 data used by the scripts, including:

```text
marmousi2_crop_aux.mat
marmousi2_crop_mod4d.mat
marmousi2_crop_stack4d_35hz.mat
```

These files contain the cropped Marmousi2 elastic-parameter models, time-domain models, prestack synthetic seismic data, angle information, wavelet information, and related sampling parameters.

The three large original SEG-Y files are:

```text
vp_marmousi-ii.segy
vs_marmousi-ii.segy
density_marmousi-ii.segy
```

Because these files are large, they may exceed the standard GitHub file-size limit and are not recommended for direct upload to the repository. These original Marmousi2 model files can be downloaded from the Allied Geophysical Laboratories website:

```text
http://www.agl.uh.edu/downloads/downloads.htm
```

After downloading, place the three SEG-Y files in the following directory:

```text
model data/
```

The expected file paths are:

```text
model data/vp_marmousi-ii.segy
model data/vs_marmousi-ii.segy
model data/density_marmousi-ii.segy
```

## Requirements

The experiments in this study were conducted on a workstation equipped with dual Intel Xeon Platinum 8173M CPUs, with 56 physical cores and 112 logical threads, and an NVIDIA GeForce RTX 3090 GPU with 24 GB VRAM.

The software environment used in this study was:

```text
Python 3.10
PyTorch
CUDA 12.2
NVIDIA GPU driver 535.54.03
NumPy
SciPy
Matplotlib
scikit-learn
h5py
```

An NVIDIA GPU is recommended for model training.

## Installation

Clone the repository:

```bash
git clone https://github.com/zhub4826-code/Prestack-Probabilistic-Inversion-JCMPC.git
cd Prestack-Probabilistic-Inversion-JCMPC
```

Create a Python environment:

```bash
conda create -n prestack_jcmpc python=3.10
conda activate prestack_jcmpc
```

Install the required packages:

```bash
pip install numpy scipy matplotlib scikit-learn h5py
pip install torch torchvision torchaudio
```

Please install the PyTorch version that matches your CUDA environment.

## Usage

### 1. Data preparation

If the processed `.mat` files are already available in `model data/data_crop/`, this step can be skipped.

To prepare the cropped Marmousi2 data from the original SEG-Y files, run:

```bash
python marmousi2_crop.py
```

### 2. Main inversion experiment

Run the main prestack probabilistic inversion experiment:

```bash
python marmousi_test.py
```

This script performs model training, prediction, validation, result visualization, and residual analysis.

### 3. Noise robustness test

Run the noise robustness experiment:

```bash
python marmousi_noise_test.py
```

This script evaluates the performance of the proposed method under different noise levels.

## Output Files

The scripts generate prediction figures, residual figures, validation-well profiles, joint-distribution plots, training curves, and saved result files.

Typical outputs include:

```text
prediction profiles
residual profiles
validation-well profiles
training curves
noise robustness results
npz result files
json metric files
```

The generated figures can be used for manuscript visualization and result analysis.

## Computer Code Availability

Name of code/library: Prestack-Probabilistic-Inversion-JCMPC

Developer: Bingbing Zhu

Contact: [18921790664@163.com](mailto:18921790664@163.com)

Hardware requirements: The experiments were conducted on a workstation equipped with dual Intel Xeon Platinum 8173M CPUs, with 56 physical cores and 112 logical threads, and an NVIDIA GeForce RTX 3090 GPU with 24 GB VRAM.

Software requirements: Python 3.10, PyTorch, CUDA 12.2, and NVIDIA GPU driver version 535.54.03.

Program language: Python

Program size: 370 KB

Availability: The source code is available at https://github.com/zhub4826-code/Prestack-Probabilistic-Inversion-JCMPC.git

## Citation

If you use this code in your research, please cite the associated paper:

```text
Bingbing Zhu, Peng Wang, Xiaoyang Wang, Qinghui Mao, and Ke Pan.
Prestack Multi-parameter Probabilistic Inversion Integrating Joint Covariance Modeling and Physics Constraints.
Computers & Geosciences.
```

The full citation information will be updated after publication.

## Notes

This repository is intended for academic research and reproducibility. Users may need to modify local file paths, data paths, and GPU settings according to their own computing environment.

Large SEG-Y model files should be downloaded separately from the Allied Geophysical Laboratories website and placed in the `model data/` directory before running the data-preparation script.

## Contact

For questions or further information, please contact:

```text
Bingbing Zhu
Email: 18921790664@163.com
```
# Prestack-Probabilistic-Inversion-JCMPC
