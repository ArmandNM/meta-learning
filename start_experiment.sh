#!/bin/bash

# First argument represents the name of the experiment. If not given, use default
if [ -z "$1" ]
  then
    EXPERIMENT_NAME="tmp"
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
rsync -a ./ "$EXPERIMENT_PATH/project/" --exclude "experiments" --exclude "datasets" --exclude ".git" --exclude ".idea"
ln -s "$(pwd)/datasets" "$(pwd)/$EXPERIMENT_PATH/project/datasets"
echo "Starting experiment: $EXPERIMENT_NAME"

# Start experiment in the background
CUDA_VISIBLE_DEVICES=0 nohup python "$EXPERIMENT_PATH/project/main.py" > "$EXPERIMENT_PATH/logs" 2>&1 &
tail -f "$EXPERIMENT_PATH/logs"
