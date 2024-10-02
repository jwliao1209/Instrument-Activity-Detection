#!/bin/bash

# # Prepare the dataset
# python prepare_dataset.py --data_path hw1/slakh/test_labels.json --data_folder hw1/slakh/test --save_path dataset/test.json

# # Inference
# python test.py --ckpt_dir checkpoints/10-02-06-52-56

# Plot the pianoroll
python plot_pianoroll.py \
    --ckpt_dir checkpoints/10-02-06-52-56 \
    --thresholds 0.87 0.0866 0.1632 0.8797 0.9467 0.4441 0.2394 0.4611 0.2807
