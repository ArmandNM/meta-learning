#!/bin/bash

# First argument represents the name of the experiment. If not given, use default
if [ -z "$1" ]
  then
    EXPERIMENT_NAME="maml_vanilla"
  else
    EXPERIMENT_NAME=$1
fi

CURRENT_DATE=$(date '+%d_%B_%Y__%H_%M_%S')
EXPERIMENT_NAME="${EXPERIMENT_NAME}__${CURRENT_DATE}"
EXPERIMENT_PATH="experiments/$EXPERIMENT_NAME"

mkdir "experiments" 2> /dev/null
mkdir "experiments/${EXPERIMENT_NAME}"
echo "Starting experiment: $EXPERIMENT_NAME"

nohup python main.py --experiment_name="${EXPERIMENT_NAME}" > "$EXPERIMENT_PATH/${EXPERIMENT_NAME}.log" 2>&1 &