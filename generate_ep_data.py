"""
most code is copied over from https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/enjoy.py
"""

import argparse
import os
import sys

import pandas as pd
import numpy as np
import yaml
from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.utils import set_random_seed

from rl_zoo3 import ALGOS, create_test_env, get_saved_hyperparams
from rl_zoo3.utils import get_model_path


def decode(env_name, environment, observation):
    """
    decoder

    return:
        Taxi-v3:
            [taxi_row, taxi_col, pass_loc, dest_idx]
        FrozenLake-v1:
            [row, col]
        CliffWalking-v0:
            [row, col]
    """
    val = observation[0]
    if env_name == "Taxi-v3":
        out = []
        out.append(val % 4)
        val = val // 4
        out.append(val % 5)
        val = val // 5
        out.append(val % 5)
        val = val // 5
        out.append(val)
        assert 0 <= val < 5
        out.reverse()

    elif env_name == "FrozenLake-v1":
        n_columns = environment.get_attr('ncol')[0]
        col = val % n_columns
        row = val // n_columns
        # def to_s(row, col):
        #     return row * ncol + col
        out = [row, col]

    elif env_name == "CliffWalking-v0":
        position = np.unravel_index(val, environment.get_attr('shape')[0])
        out = list(position)

    else:
        print(f"Environment {env_name} does not have a decoder yet.")
        out = [val]

    return out


def run_atari_eval_episodes(
        str_env_name: str,
        algo: str,
        folder: str,
        num_episodes: int,
        log_dir: str,
        r_seed: int = 0,
        deterministic: bool = True,
        render: bool = False,
        norm_reward: bool = False,
        device: str = "auto"
):
    """
    This function runs the specified number of episodes in the singular atari environment. 

    params:
        str_env_name : str
            The environment name provided as a string. Ex. "AsteroidsNoFrameskip-v4"
        algo : str
            The name of the algorithm used to train the agent being evaluated. This is needed to
            properly load the trained agent. Ex. "ppo"
        folder : str
            The folder where the trained agent is located. Just the top level because the 
            algorithm name and environment name will be used to complete the path.
        num_episodes : int
            The number of episodes to evaluate in the environment with the trained agent.
        log_dir : str
            Where the episode data will be saved.
        r_seed : int = 0
            Random seed for seeding the environment.
        deterministic : bool = True
            Whether to run the trained agent as a deterministic policy or not.
        render : bool = False
            Render the environment while running through the evaluation?
        norm_reward : bool = False
            Normalize the reward? This will scale things accross environments so all final scores 
            are in the range 0-1.
        device : str = "auto"
            Run on GPU, CPU, or let the system decide based on what it can find?
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
    np.set_printoptions(threshold=sys.maxsize)

    if "HerReplayBuffer" in hyperparams.get("replay_buffer_class", ""):
        kwargs["env"] = env

    model = ALGOS[algo].load(model_path, custom_objects=custom_objects, device=device, **kwargs)

    # Run through all the episodes
    full_df = pd.DataFrame()

    for ep in range(num_episodes):
        done = False
        obs = env.reset()
        states = []
        observations = []
        actions = []
        rewards = []
        total_reward = 0

        while not done:
            # Get the current RAM state and save it as a string to remove newlines. It will be
            # saved in the csv as a string anyway, so pre-empting that conversion allows us
            # to clean it up.
            ram_val = env.get_attr('ale')[0].getRAM()
            states.append(str(ram_val).replace("\n", ""))
            gray_screen = env.get_attr('ale')[0].getScreenGrayscale()
            observations.append(str(gray_screen).replace("\n", "").replace(
                "] [", "").replace("[[", "[").replace("]]", "]"))

            # Get the action the agent wil take
            action, _ = model.predict(obs, deterministic=deterministic)
            actions.append(action[0])

            # Execute the action
            obs, reward, done, _ = env.step(action)
            total_reward += reward[0]
            rewards.append(reward[0])

        # Get terminal state and add a terminal action
        ram_val = env.get_attr('ale')[0].getRAM()
        states.append(str(ram_val).replace("\n", ""))
        gray_screen = env.get_attr('ale')[0].getScreenGrayscale()
        observations.append(str(gray_screen).replace("\n", ""))
        actions.append('terminal')
        rewards.append(total_reward)

        # Create the dataframe and save it
        temp_df = pd.DataFrame(
            {
                "Episode Number":   [ep] * len(states),
                "Timestep":         range(len(states)),
                "RAM State":        states,
                "Gray Scale Image": observations,
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
    # full_df.to_pickle(data_path.replace(".csv", ".pkl"))  # Will save a pkl file if desired


def run_eval_episodes(
        str_env_name: str,
        algo: str,
        folder: str,
        num_episodes: int,
        log_dir: str,
        r_seed: int = 0,
        deterministic: bool = True,
        render: bool = False,
        norm_reward: bool = False,
        device: str = "auto"
):
    """
    This function runs the specified number of episodes in the singular, non-atari environment. 

    params:
        str_env_name : str
            The environment name provided as a string. Ex. "CliffWalker-v0"
        algo : str
            The name of the algorithm used to train the agent being evaluated. This is needed to
            properly load the trained agent. Ex. "ppo"
        folder : str
            The folder where the trained agent is located. Just the top level because the 
            algorithm name and environment name will be used to complete the path.
        num_episodes : int
            The number of episodes to evaluate in the environment with the trained agent.
        log_dir : str
            Where the episode data will be saved.
        r_seed : int = 0
            Random seed for seeding the environment.
        deterministic : bool = True
            Whether to run the trained agent as a deterministic policy or not.
        render : bool = False
            Render the environment while running through the evaluation?
        norm_reward : bool = False
            Normalize the reward? This will scale things accross environments so all final scores 
            are in the range 0-1.
        device : str = "auto"
            Run on GPU, CPU, or let the system decide based on what it can find?
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
    np.set_printoptions(threshold=sys.maxsize)

    if "HerReplayBuffer" in hyperparams.get("replay_buffer_class", ""):
        kwargs["env"] = env

    model = ALGOS[algo].load(model_path, custom_objects=custom_objects, device=device, **kwargs)

    # Run through all the episodes
    full_df = pd.DataFrame()

    for ep in range(num_episodes):
        done = False
        obs = env.reset()
        states = []         # Extracted information about the environment converted from the obs
        observations = []   # obs returned from the environment
        actions = []        # Action selected by the expert
        rewards = []
        total_reward = 0
        max_steps = 200
        steps = 0

        while not done and steps < max_steps:
            # Get the state and observation
            hidden_state = decode(str_env_name, env, obs)
            states.append(str(hidden_state).replace(',', ''))
            observations.append(obs)

            # Get the action the agent wil take
            action, _ = model.predict(obs, deterministic=deterministic)
            actions.append(action[0])

            # Execute the action
            obs, reward, done, _ = env.step(action)
            total_reward += reward[0]
            rewards.append(reward[0])
            steps += 1

        # Get terminal state and add a terminal action
        hidden_state = decode(str_env_name, env, obs)
        states.append(str(hidden_state).replace(',', ''))
        observations.append(obs)
        actions.append('terminal')
        rewards.append(total_reward)

        # Create the dataframe and save it
        temp_df = pd.DataFrame(
            {
                "Episode Number":   [ep] * len(states),
                "Timestep":         range(len(states)),
                "Hidden State":     states,
                "Observation":      observations,
                "Action":           actions,
                "Rewards":          rewards
            }
        )

        full_df = pd.concat([full_df, temp_df], ignore_index=True)

        # Print status update
        print(f"Completed Ep: {ep+1}/{num_episodes} for {algo}_{str_env_name}")

    # Save the results to the desired folder
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    filename = f"{algo}_{str_env_name}_data.csv"
    data_path = os.path.join(log_dir, filename)
    full_df.to_csv(data_path, index=False)
    # full_df.to_pickle(data_path.replace(".csv", ".pkl"))  # Will save a pkl file if desired


if __name__ == "__main__":
    # Collect arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", default=False,
                        help="Run all algorithms in all environments as described in docs.")
    parser.add_argument("--atari", action="store_true", default=False,
                        help="Run all algorithms in all Atari environments as described in docs.")
    parser.add_argument("--text", action="store_true", default=False,
                        help="Run all algorithms in all Toy Text environments as described in \
                            docs.")
    parser.add_argument("--env", help="environment ID", type=EnvironmentName,
                        default="AsteroidsNoFrameskip-v4")
    parser.add_argument("-f", "--folder", help="Log folder", type=str,
                        default="rl_experts")
    parser.add_argument("--algo", help="RL Algorithm", default="ppo",
                        type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument("-n", "--n-episodes",
                        help="number of episodes to evaluate", default=1, type=int)
    parser.add_argument(
        "--render", action="store_true", default=False,
        help="Render the environment (default is to not render)")
    parser.add_argument(
        "--device", help="PyTorch device to be use (ex: cpu, cuda...)", default="auto", type=str)
    parser.add_argument("--stochastic", action="store_true",
                        default=False, help="Use stochastic actions")
    parser.add_argument(
        "--norm-reward", action="store_true", default=False,
        help="Normalize reward if applicable (trained with VecNormalize)")
    parser.add_argument("--seed", help="Random generator seed", type=int, default=0)
    args = parser.parse_args()

    ALGORITHMS = [
        "a2c",
        "ars",
        "dqn",
        "ppo",
        "qrdqn"
    ]
    ATARI_ENVIRONMENTS = [
        "AsteroidsNoFrameskip-v4",
        "RoadRunnerNoFrameskip-v4",
        "SeaquestNoFrameskip-v4",
        "SpaceInvadersNoFrameskip-v4"
    ]
    TEXT_ENVIRONMENTS = [
        # "Blackjack-v1",
        "CliffWalking-v0",
        "Taxi-v3",
        "FrozenLake-v1"
    ]

    if args.all:
        args.atari = True
        args.text = True

    if args.atari:
        for algorithm in ALGORITHMS:
            for environment in ATARI_ENVIRONMENTS:
                run_atari_eval_episodes(
                    str_env_name=environment,
                    algo=algorithm,
                    folder=args.folder,
                    num_episodes=args.n_episodes,
                    log_dir=f"./logs_{args.n_episodes}",
                    r_seed=args.seed,
                    deterministic=not args.stochastic,
                    render=args.render,
                    norm_reward=args.norm_reward,
                    device=args.device
                )

    elif args.text:
        for algorithm in ALGORITHMS:
            for environment in TEXT_ENVIRONMENTS:
                if (algorithm == "ars") and (environment == "CliffWalking-v0"):
                    continue
                run_eval_episodes(
                    str_env_name=environment,
                    algo=algorithm,
                    folder=args.folder,
                    num_episodes=args.n_episodes,
                    log_dir=f"./ep_data_{args.n_episodes}",
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
