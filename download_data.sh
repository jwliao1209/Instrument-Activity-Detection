#!/bin/bash

# Download training data
if [ ! -d "hw1" ]; then
    gdown 1gRykfOOmKJsxppBo3DT7Nm7gMiPGVUAa -O hw1.zip
    unzip -n hw1.zip
    rm hw1.zip
fi
