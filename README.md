# Sudoku AI - Neural Network

[![license](https://img.shields.io/pypi/l/ansicolortags.svg)]()
[![GitHub issues](https://img.shields.io/github/issues/paul-buechner/sudoku-ai)]()
[![GitHub pull requests](https://img.shields.io/github/issues-pr/paul-buechner/sudoku-ai)]()

Sudoku AI is an sudoku solving algorithm recognizing real images using neural-network trained on MNIST dataset with tensorflow.

It also can perform a live digit recognition via camera feed.

<div align="center" style="padding:5px;">
<img src="https://media.giphy.com/media/ASAPIID1mWIUABjMYB/giphy.gif" align="left" style="margin: 10px;" width="300" height="225"/>
<img src="https://media.giphy.com/media/nOuBaamMmtVH2E60op/giphy.gif" align="right" style="margin: 10px;" width="300" height="225"/>
</div>

# Installation

Sudoku-AI runs on Python 3.7 or higher version. The featured version of Tensorflow is 2.4. Read more about the setup [here](#Tensorflow).

## Directly

If you want to hack on yourself, clone this repository and in that directory execute:

```bash
# Install python requirements
pip install -r requirements.txt
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.

## Anaconda

If you are using Anaconda Environment execute the following steps:

- Create a conda environment using `conda env create -f environment.yml`
- Activate the created environment `conda activate sudoku-ai`

# Tensorflow

The following NVIDIA® software must be installed on your system:

- [NVIDIA® GPU drivers](https://www.nvidia.com/download/index.aspx?lang=en-us) — CUDA® 11.0 requires 450.x or higher.
- [CUDA® Toolkit](https://developer.nvidia.com/cuda-11.0-download-archive) — TensorFlow supports CUDA® 11 (TensorFlow >= 2.4.0)
- [CUPTI](https://docs.nvidia.com/cuda/cupti/) ships with the CUDA® Toolkit.
- [cuDNN SDK 8.0.5](https://developer.nvidia.com/rdp/cudnn-archive#a-collapse805-110) for CUDA 11.0 [cuDNN versions](https://developer.nvidia.com/rdp/cudnn-archive).

See more information [here](https://www.tensorflow.org/install/gpu).

## Installation

For more information installing the required drivers look up:

- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)
- [cuDNN Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows)

# Usage

After installing the required packages run the sudoku solver:

```bash
python main.py
```

The image on which the solver is applied can be configured in `main() -> path="path/to/image"` function of `main.py`

Running the live camera feed:

```bash
cd src
python live_model.py
```

# Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

# License

This project falls under the [MIT](https://choosealicense.com/licenses/mit/) license.
