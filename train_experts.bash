#!/bin/bash
conda activate atari

# Train Blackjack experts (the action space for this environment is not currently supported for sb3)
# python -m rl_zoo3.train --algo a2c --env Blackjack-v1 --conf ./hyperparameters/a2c.yml -f rl_experts --seed 8 --eval-freq -1
# python -m rl_zoo3.train --algo ars --env Blackjack-v1 --conf ./hyperparameters/ars.yml -f rl_experts --seed 8 --eval-freq -1
# python -m rl_zoo3.train --algo dqn --env Blackjack-v1 --conf ./hyperparameters/dqn.yml -f rl_experts --seed 8 --eval-freq -1
# python -m rl_zoo3.train --algo ppo --env Blackjack-v1 --conf ./hyperparameters/ppo.yml -f rl_experts --seed 8 --eval-freq -1
# python -m rl_zoo3.train --algo qrdqn --env Blackjack-v1 --conf ./hyperparameters/qrdqn.yml -f rl_experts --seed 8 --eval-freq -1

# Train CliffWalking experts
python -m rl_zoo3.train --algo a2c --env CliffWalking-v0 --conf ./hyperparameters/a2c.yml -f rl_experts --seed 8 --eval-freq -1
python -m rl_zoo3.train --algo ars --env CliffWalking-v0 --conf ./hyperparameters/ars.yml -f rl_experts --seed 8 --eval-freq -1
python -m rl_zoo3.train --algo dqn --env CliffWalking-v0 --conf ./hyperparameters/dqn.yml -f rl_experts --seed 8 --eval-freq -1
python -m rl_zoo3.train --algo ppo --env CliffWalking-v0 --conf ./hyperparameters/ppo.yml -f rl_experts --seed 8 --eval-freq -1
python -m rl_zoo3.train --algo qrdqn --env CliffWalking-v0 --conf ./hyperparameters/qrdqn.yml -f rl_experts --seed 8 --eval-freq -1

# Train Taxi experts
python -m rl_zoo3.train --algo a2c --env Taxi-v3 --conf ./hyperparameters/a2c.yml -f rl_experts --seed 8 --eval-freq -1
python -m rl_zoo3.train --algo ars --env Taxi-v3 --conf ./hyperparameters/ars.yml -f rl_experts --seed 8 --eval-freq -1
python -m rl_zoo3.train --algo dqn --env Taxi-v3 --conf ./hyperparameters/dqn.yml -f rl_experts --seed 8 --eval-freq -1
python -m rl_zoo3.train --algo ppo --env Taxi-v3 --conf ./hyperparameters/ppo.yml -f rl_experts --seed 8 --eval-freq -1
python -m rl_zoo3.train --algo qrdqn --env Taxi-v3 --conf ./hyperparameters/qrdqn.yml -f rl_experts --seed 8 --eval-freq -1

# Train FrozenLake experts
python -m rl_zoo3.train --algo a2c --env FrozenLake-v1 --conf ./hyperparameters/a2c.yml -f rl_experts --seed 8 --eval-freq -1
python -m rl_zoo3.train --algo ars --env FrozenLake-v1 --conf ./hyperparameters/ars.yml -f rl_experts --seed 8 --eval-freq -1
python -m rl_zoo3.train --algo dqn --env FrozenLake-v1 --conf ./hyperparameters/dqn.yml -f rl_experts --seed 8 --eval-freq -1
python -m rl_zoo3.train --algo ppo --env FrozenLake-v1 --conf ./hyperparameters/ppo.yml -f rl_experts --seed 8 --eval-freq -1
python -m rl_zoo3.train --algo qrdqn --env FrozenLake-v1 --conf ./hyperparameters/qrdqn.yml -f rl_experts --seed 8 --eval-freq -1
