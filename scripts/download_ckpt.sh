#!/bin/bash

# Download checkpoints
if [ ! -d "checkpoints" ]; then
    gdown 10_6mqJ3L5OHC8YpIoUX7hnGJjfMS2uL- -O checkpoints.zip
    unzip -n checkpoints.zip
    rm checkpoints.zip
fi
