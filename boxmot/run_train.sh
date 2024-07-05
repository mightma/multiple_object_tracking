#!/bin/bash

# install dependencies
poetry install --with yolo  # installed boxmot + yolo dependencies
poetry shell  # activates the newly created environment with the installed dependencies

# Run training
python3 train.py

# Example of using command line instruction
# PYTHON3_PATH=$(which python3)
# export PATH=$PATH:${PYTHON3_PATH%python3}
# yolo detect train data=litter_1cls_dataset.yaml \
#     model=yolov8s.pt epochs=300 imgsz=640 batch=-1 name="litter_1cls_yolov8s" \
#     devices=0,1,2,3,4,5,6,7 \
#     project="/opt/ml/model"
