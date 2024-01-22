"""
most code is copied over from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/enjoy.py
"""

import argparse
import importlib
import os
import sys

import gymnasium as gym
import numpy as np
import torch as th
import yaml
from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.callbacks import tqdm
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import AtariWrapper

import rl_zoo3.import_envs
from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.load_from_hub import download_from_hub
from rl_zoo3.utils import StoreDict, get_model_path


if __name__ == "__main__":
    # Collect arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=EnvironmentName,
                        default="AsteroidsNoFrameskip-v4")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl_trained_agents")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo",
                        type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-timesteps", help="number of timesteps", default=1000, type=int)
    parser.add_argument(
        "--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument("--n-envs", help="number of environments", default=1, type=int)
    parser.add_argument(
        "--exp-id", help="Experiment ID (default: 0: latest, -1: no exp folder)", default=0, type=int)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)",
                        default=1, type=int)
    parser.add_argument(
        "--render", action="store_true", default=False, help="Render the environment (default is to not render)"
    )
    parser.add_argument("--deterministic", action="store_true",
                        default=False, help="Use deterministic actions")
    parser.add_argument(
        "--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="auto", type=str)
    parser.add_argument(
        "--load-best", action="store_true", default=False, help="Load best model instead of last model if available"
    )
    parser.add_argument(
        "--load-checkpoint",
        type=int,
        help="Load checkpoint instead of last model if available, "
        "you must pass the number of timesteps corresponding to it",
    )
    parser.add_argument(
        "--load-last-checkpoint",
        action="store_true",
        default=False,
        help="Load last checkpoint instead of last model if available",
    )
    parser.add_argument("--stochastic", action="store_true",
                        default=False, help="Use stochastic actions")
    parser.add_argument(
        "--norm-reward", action="store_true", default=False, help="Normalize reward if applicable (trained with VecNormalize)"
    )
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    parser.add_argument("--reward-log", help="Where to log reward", default="", type=str)
    parser.add_argument(
        "--gym-packages",
        type=str,
        nargs="+",
        default=[],
        help="Additional external Gym environment package modules to import",
    )
    parser.add_argument(
        "--env-kwargs", type=str, nargs="+", action=StoreDict, help="Optional keyword argument to pass to the env constructor"
    )
    parser.add_argument(
        "--custom-objects", action="store_true", default=False, help="Use custom objects to solve loading issues"
    )
    parser.add_argument(
        "-P",
        "--progress",
        action="store_true",
        default=False,
        help="if toggled, display a progress bar using tqdm and rich",
    )
    args = parser.parse_args()

    #
    env_name: EnvironmentName = args.env
    algo = args.algo
    folder = args.folder

    _, model_path, log_path = get_model_path(
        args.exp_id,
        folder,
        algo,
        env_name,
        args.load_best,
        args.load_checkpoint,
        args.load_last_checkpoint,
    )

    set_random_seed(args.seed)

    stats_path = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(
        stats_path, norm_reward=args.norm_reward, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path) as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    # overwrite with command line arguments
    if args.env_kwargs is not None:
        env_kwargs.update(args.env_kwargs)

    log_dir = args.reward_log if args.reward_log != "" else None

    env = create_test_env(
        env_name.gym_id,
        n_envs=args.n_envs,
        stats_path=maybe_stats_path,
        seed=args.seed,
        log_dir=log_dir,
        should_render=args.render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    model = ALGOS[algo].load(model_path, device=args.device)  # , **kwargs)

    done = False
    obs = env.reset()

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        ram_val = env.get_attr('ale')[0].getRAM()
        print(f'{ram_val}')

    # Episode #, timestep (opt), ROM state, action,
    # [0, 1, 254, 67, 98, ...]
