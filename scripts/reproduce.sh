#!/bin/bash

# Prepare the dataset
python prepare_dataset.py --data_path hw1/slakh/test_labels.json --data_folder hw1/slakh/test --save_path dataset/test.json

# Inference
python test.py

# Plot the pianoroll
python plot_pianoroll.py
