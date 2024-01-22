# Instructions for creating episode traces from RL agents

There are a few steps to setting up your system to generate new data. If you want to use the existing data, the following steps are not necessary.

## 1 Use conda to setup the environment

```bash
conda create -n atari python=3.9
conda activate atari
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install stable-baselines3[extra]
pip install rl_zoo3
pip install gymnasium[accept-rom-license]
pip install gymnasium[atari]
```

## 2 Dowload all the trained agents

Because we don't need to keep it, hugging face

```bash
# Space Invaders
python -m rl_zoo3.load_from_hub --algo a2c --env SpaceInvadersNoFrameskip-v4 -orga sb3 -f rl_trained_agents/
python -m rl_zoo3.load_from_hub --algo dqn --env SpaceInvadersNoFrameskip-v4 -orga sb3 -f rl_trained_agents/
python -m rl_zoo3.load_from_hub --algo qrdqn --env SpaceInvadersNoFrameskip-v4 -orga sb3 -f rl_trained_agents/
python -m rl_zoo3.load_from_hub --algo ppo --env SpaceInvadersNoFrameskip-v4 -orga sb3 -f rl_trained_agents/
# SeaQuest
python -m rl_zoo3.load_from_hub --algo a2c --env SeaquestNoFrameskip-v4 -orga sb3 -f rl_trained_agents/
python -m rl_zoo3.load_from_hub --algo dqn --env SeaquestNoFrameskip-v4 -orga sb3 -f rl_trained_agents/
python -m rl_zoo3.load_from_hub --algo qrdqn --env SeaquestNoFrameskip-v4 -orga sb3 -f rl_trained_agents/
python -m rl_zoo3.load_from_hub --algo ppo --env SeaquestNoFrameskip-v4 -orga sb3 -f rl_trained_agents/
# RoadRunner
python -m rl_zoo3.load_from_hub --algo a2c --env RoadRunnerNoFrameskip-v4 -orga sb3 -f rl_trained_agents/
python -m rl_zoo3.load_from_hub --algo dqn --env RoadRunnerNoFrameskip-v4 -orga sb3 -f rl_trained_agents/
python -m rl_zoo3.load_from_hub --algo qrdqn --env RoadRunnerNoFrameskip-v4 -orga sb3 -f rl_trained_agents/
python -m rl_zoo3.load_from_hub --algo ppo --env RoadRunnerNoFrameskip-v4 -orga sb3 -f rl_trained_agents/
# Asteroids
python -m rl_zoo3.load_from_hub --algo a2c --env AsteroidsNoFrameskip-v4 -orga sb3 -f rl_trained_agents/
python -m rl_zoo3.load_from_hub --algo dqn --env AsteroidsNoFrameskip-v4 -orga sb3 -f rl_trained_agents/
python -m rl_zoo3.load_from_hub --algo qrdqn --env AsteroidsNoFrameskip-v4 -orga sb3 -f rl_trained_agents/
python -m rl_zoo3.load_from_hub --algo ppo --env AsteroidsNoFrameskip-v4 -orga sb3 -f rl_trained_agents/
```

## 3 Gennerate the data

`python generate_ep_data.py --all`
