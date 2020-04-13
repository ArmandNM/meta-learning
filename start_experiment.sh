#!/bin/bash

# First argument represents the name of the experiment. If not given, use default
if [ -z "$1" ]
  then
    EXPERIMENT_NAME="grifon_v1"
  else
    EXPERIMENT_NAME=$1
fi

CURRENT_DATE=$(date '+%d_%B_%Y__%H_%M_%S')
EXPERIMENT_NAME="${EXPERIMENT_NAME}__${CURRENT_DATE}"
EXPERIMENT_PATH="experiments/$EXPERIMENT_NAME"

# Create experiment folder
mkdir "experiments" 2> /dev/null
mkdir "experiments/${EXPERIMENT_NAME}"

# Clone project directory
echo "Cloning project to experiment directory"
mkdir "$EXPERIMENT_PATH/project/" 2> /dev/null
#shopt -s extglob
#cp -r !("experiments", "datasets") "$EXPERIMENT_PATH/project/"
rsync -a ./ "$EXPERIMENT_PATH/project/" --exclude "experiments" --exclude "datasets"
mkdir "$EXPERIMENT_PATH/project/experiments"
ln -s "$(pwd)/datasets" "$(pwd)/$EXPERIMENT_PATH/project/datasets"
echo "Starting experiment: $EXPERIMENT_NAME"

# Start experiment in the background
CUDA_VISIBLE_DEVICES=0 nohup python main.py --experiment_name="${EXPERIMENT_NAME}" > "$EXPERIMENT_PATH/${EXPERIMENT_NAME}.log" 2>&1 &
tail -f "$EXPERIMENT_PATH/${EXPERIMENT_NAME}.log"