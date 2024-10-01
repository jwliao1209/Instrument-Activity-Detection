#!/bin/bash

# python train.py --model_name m-a-p/MERT-v1-95M --hidden_states first --fine_tune_method full --epochs 10
python train.py --model_name m-a-p/MERT-v1-95M --hidden_states first --fine_tune_method lora --epochs 10


