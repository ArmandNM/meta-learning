#!/bin/bash

# You can use either an iteration checkpoint or a named checkpoint.
# Examples:
  # --checkpoint_name="checkpoint_1500"
  # --checkpoint_name="best_checkpoint"
  # --checkpoint_name="last_checkpoint'

# Run final testing using model from given checkpoint
# CUDA_VISIBLE_DEVICES=0 nohup python main.py --test --checkpoint_name="checkpoint_1500" >> "../logs" 2>&1 &

# Continue training from given checkpoint
CUDA_VISIBLE_DEVICES=0 nohup python main.py --checkpoint_name="last_checkpoint" >> "../logs" 2>&1 &

# Watch logs
tail -f "../logs"