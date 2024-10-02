#!/bin/bash

python train.py --model_name m-a-p/MERT-v1-95M --hidden_states first --fine_tune_method last_layer
python train.py --model_name m-a-p/MERT-v1-95M --hidden_states first --fine_tune_method full
python train.py --model_name m-a-p/MERT-v1-95M --hidden_states first --fine_tune_method lora
python train.py --model_name m-a-p/MERT-v1-95M --hidden_states all --fine_tune_method last_layer

python train.py --model_name m-a-p/MERT-v1-330M --hidden_states first --fine_tune_method last_layer
python train.py --model_name m-a-p/MERT-v1-330M --hidden_states first --fine_tune_method full
python train.py --model_name m-a-p/MERT-v1-330M --hidden_states first --fine_tune_method lora
python train.py --model_name m-a-p/MERT-v1-330M --hidden_states all --fine_tune_method last_layer
