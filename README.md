# Instrument-Activity-Detection

This repository contains the implementation for Homework 1 of the CommE5070 Deep Learning for Music Analysis and Generation course, Fall 2024, at National Taiwan University. For a detailed report, please refer to this [slides](https://docs.google.com/presentation/d/1-gXmFUJInCMMKa3ISmMDKD37zjx9JIyQBKthhEseU7c/edit?usp=sharing).


## Setup
To set up the virtual environment and install the required packages, use the following commands:
```
virtualenv --python=python3.10 deepmir_hw1
source deepmir_hw1/bin/activate
pip install -r requirements.txt
```

## Data and Checkpoint Download

### Dataset
To download the dataset, run the following script:
```
bash scripts/download_data.sh
```

### Checkpoint
To download the pre-trained model checkpoints, use the command:
```
bash scripts/download_ckpt.sh
```


## Dataset Preparation
To prepare the training, validation, and test datasets, execute the following command:
```
bash scripts/prepare_data.sh
```


## Training
To train the model, run the command:
```
bash scripts/train.sh
```


## Reproducing
To reproduce the inference results, run the command:
```
bash scripts/run_reproduce.sh
```


## Environment
We implemented the code on an environment running Ubuntu 22.04.1, utilizing a 12th Generation Intel(R) Core(TM) i7-12700 CPU, along with a single NVIDIA GeForce RTX 4090 GPU equipped with 24 GB of dedicated memory.


## Citation
If you use this code, please cite the following:
```bibtex
@misc{instrument_activity_detection_2024,
    title  = {Instrument-Activity-Detection},
    author = {Jia-Wei Liao},
    url    = {https://github.com/jwliao1209/Instrument-Activity-Detection},
    year   = {2024}
}
```
