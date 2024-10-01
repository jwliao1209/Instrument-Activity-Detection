#!/bin/bash

python prepare_dataset.py --data_path hw1/slakh/train_labels.json --data_folder hw1/slakh/train --save_path dataset/train.json
python prepare_dataset.py --data_path hw1/slakh/validation_labels.json --data_folder hw1/slakh/validation --save_path dataset/valid.json
python prepare_dataset.py --data_path hw1/slakh/test_labels.json --data_folder hw1/slakh/test --save_path dataset/test.json
