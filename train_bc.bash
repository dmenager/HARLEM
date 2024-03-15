#!/bin/bash
conda activate atari

# Train CliffWalking-v0
python behavior_cloning.py --train-expert --env CliffWalking-v0 --eval
python behavior_cloning.py --train-hems --env CliffWalking-v0 --eval
python behavior_cloning.py --train-expert-hems --env CliffWalking-v0 --eval
python behavior_cloning.py --train-hems-expert --env CliffWalking-v0 --eval
python behavior_cloning.py --train-both --env CliffWalking-v0 --eval

# Train FrozenLake-v1
python behavior_cloning.py --train-expert --env FrozenLake-v1 --eval
python behavior_cloning.py --train-hems --env FrozenLake-v1 --eval
python behavior_cloning.py --train-expert-hems --env FrozenLake-v1 --eval
python behavior_cloning.py --train-hems-expert --env FrozenLake-v1 --eval
python behavior_cloning.py --train-both --env FrozenLake-v1 --eval

# Train Taxi-v3
python behavior_cloning.py --train-expert --env Taxi-v3 --eval
python behavior_cloning.py --train-hems --env Taxi-v3 --eval
python behavior_cloning.py --train-expert-hems --env Taxi-v3 --eval
python behavior_cloning.py --train-hems-expert --env Taxi-v3 --eval
python behavior_cloning.py --train-both --env Taxi-v3 --eval

# Evaluate something in particular, swap out the env and load
python behavior_cloning.py --env CliffWalking-v0 --load expert_trained_CliffWalking-v0.pkl --eval