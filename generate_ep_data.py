"""
most code is copied over from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/enjoy.py
"""

import argparse
import os
import sys

import pandas as pd
import yaml
from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.utils import set_random_seed

from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.utils import StoreDict, get_model_path


def run_eval_episodes(str_env_name, algo, folder, num_episodes, log_dir, r_seed=0, deterministic=True, render=False, norm_reward=False, device="auto"):
    """
    """
    # Build the environment
    env_name = EnvironmentName(str_env_name)

    _, model_path, log_path = get_model_path(
        exp_id=0,
        folder=folder,
        algo=algo,
        env_name=env_name,
    )

    set_random_seed(r_seed)

    stats_path = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(
        stats_path, norm_reward=norm_reward, test_mode=True)

    # load env_kwargs if existing
    env_kwargs = {}
    args_path = os.path.join(log_path, env_name, "args.yml")
    if os.path.isfile(args_path):
        with open(args_path, encoding='UTF-8') as f:
            loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)
            if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]

    env = create_test_env(
        env_name.gym_id,
        n_envs=1,
        stats_path=maybe_stats_path,
        seed=r_seed,
        should_render=render,
        hyperparams=hyperparams,
        env_kwargs=env_kwargs,
    )

    # Load the RL model
    kwargs = dict(seed=r_seed)
    off_policy_algos = ["qrdqn", "dqn", "ddpg", "sac", "her", "td3", "tqc"]
    if algo in off_policy_algos:
        # Dummy buffer size as we don't need memory to enjoy the trained agent
        kwargs.update(dict(buffer_size=1))
        # Hack due to breaking change in v1.6
        # handle_timeout_termination cannot be at the same time
        # with optimize_memory_usage
        if "optimize_memory_usage" in hyperparams:
            kwargs.update(optimize_memory_usage=False)

    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }

    if "HerReplayBuffer" in hyperparams.get("replay_buffer_class", ""):
        kwargs["env"] = env

    model = ALGOS[algo].load(model_path, custom_objects=custom_objects, device=device, **kwargs)

    # Run through all the episodes
    full_df = pd.DataFrame()

    for ep in range(num_episodes):
        done = False
        obs = env.reset()
        states = []
        actions = []
        rewards = []
        total_reward = 0

        while not done:
            # Get the current RAM state and save it as a string to remove newlines. It will be saved
            # in the csv as a string anyway, so pre-empting that conversion allows us to clean it up.
            ram_val = env.get_attr('ale')[0].getRAM()
            states.append(str(ram_val).replace("\n", ""))

            # Get the action the agent wil take
            action, _ = model.predict(obs, deterministic=deterministic)
            actions.append(action[0])

            # Execute the action
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            rewards.append(reward[0])

        # Get terminal state and add a terminal action
        ram_val = env.get_attr('ale')[0].getRAM()
        states.append(str(ram_val).replace("\n", ""))
        actions.append('terminal')
        rewards.append(total_reward)

        # Create the dataframe and save it
        temp_df = pd.DataFrame(
            {
                "Episode Number":   [ep] * len(states),
                "Timestep":         range(len(states)),
                "RAM State":        states,
                "Action":           actions,
                "Rewards":          rewards
            }
        )

        full_df = pd.concat([full_df, temp_df], ignore_index=True)

        # Print status update
        print(f"Completed Ep: {ep+1}/{num_episodes} for {algo}_{str_env_name}")

    # Save the results to the desired folder
    filename = f"{algo}_{str_env_name}_data.csv"
    data_path = os.path.join(log_dir, filename)
    full_df.to_csv(data_path, index=False)


if __name__ == "__main__":
    # Collect arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", default=False,
                        help="Run all algorithms in all environments as described in docs.")
    parser.add_argument("--env", help="environment ID", type=EnvironmentName,
                        default="AsteroidsNoFrameskip-v4")
    parser.add_argument("-f", "--folder", help="Log folder", type=str, default="rl_trained_agents")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo",
                        type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-episodes",
                        help="number of episodes to evaluate", default=1000, type=int)
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
    parser.add_argument(
        "--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="auto", type=str)
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

    if args.all:
        ALGORITHMS = ["a2c", "dqn", "ppo", "qrdqn"]
        ENVIRONMENTS = ["AsteroidsNoFrameskip-v4", "RoadRunnerNoFrameskip-v4",
                        "SeaquestNoFrameskip-v4", "SpaceInvadersNoFrameskip-v4"]

        for algorithm in ALGORITHMS:
            for environment in ENVIRONMENTS:
                run_eval_episodes(
                    str_env_name=environment,
                    algo=algorithm,
                    folder=args.folder,
                    num_episodes=args.n_episodes,
                    log_dir="./logs",
                    r_seed=args.seed,
                    deterministic=not args.stochastic,
                    render=args.render,
                    norm_reward=args.norm_reward,
                    device=args.device
                )

    else:
        run_eval_episodes(
            str_env_name=args.env,
            algo=args.algo,
            folder=args.folder,
            num_episodes=args.n_episodes,
            log_dir="./logs",
            r_seed=args.seed,
            deterministic=not args.stochastic,
            render=args.render,
            norm_reward=args.norm_reward,
            device=args.device
        )
