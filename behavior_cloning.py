"""
This code is utilizes stuff from https://github.com/driptaRC/BCO-PyTorch 
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.distributions import Normal
from torch.distributions.categorical import Categorical
import pickle
import tempfile
import pandas as pd
import gymnasium as gym
import numpy as np
import cl4py
from cl4py import Cons, List, DottedList
import os
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.model_zoo as model_zoo
import torch.onnx
from scipy import stats
from functools import partial
from random import randint
import inflect
import time
import sys

from huggingface_sb3 import EnvironmentName
from rl_zoo3 import ALGOS, get_saved_hyperparams
from rl_zoo3.utils import get_model_path

class NNPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(NNPolicy, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.linear_1 = nn.Linear(state_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        #self.linear_3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.linear_1(x)
        # x = F.leaky_relu(x, 0.001)
        x = F.tanh(x)
        x = self.linear_2(x)
        # x = F.leaky_relu(x, 0.001)
        x = F.tanh(x)
        '''
        x = self.linear_3(x)
        # x = F.leaky_relu(x, 0.001)
        x = F.tanh(x)
        '''
        logits = self.linear_out(x)
        #return Categorical(logits=logits)
        return logits

    def _get_constructor_parameters(self):
        """
        Get data that need to be saved in order to re-create the model when loading it from disk.

        :return: The dictionary to pass to the as kwargs constructor when reconstruction this model.
        """
        return dict(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            action_dim=self.action_dim,
        )

    def save(self, path: str):
        """
        Save model to a given location.

        :param path: string path to file. Must end with .pkl
        """
        torch.save({"state_dict": self.state_dict(),
                   "data": self._get_constructor_parameters()}, path)

    @classmethod
    def load(cls, path: str):
        """
        Load model from path.

        :param path:
        :param device: Device on which the policy should be loaded.
        :return:
        """
        # Note(antonin): we cannot use `weights_only=True` here because we need to allow
        # gymnasium imports for the policy to be loaded successfully  map_location="auto"
        saved_variables = torch.load(path, weights_only=False)

        # Create policy object
        model = cls(**saved_variables["data"])
        # Load weights
        model.load_state_dict(saved_variables["state_dict"])
        # model.to("auto")
        return model


class ImitationDataset(Dataset):
    def __init__(self):
        self.data = []

    def build_from_atari(self, demos):
        for idx, state in enumerate(demos['RAM State']):
            action = demos['Action'][idx]  # [0]
            if action != 'terminal':
                torch_state = torch.tensor(np.fromstring(state.replace(
                    '[', '').replace(']', ''), sep=' ', dtype='float32'), dtype=torch.float32)
                torch_action = torch.tensor(np.array(action, dtype='uint8'), dtype=torch.uint8)
                self.data.append((torch_state, torch_action))

    def build_from_toy_text(self, demos):
        for idx, state in enumerate(demos['Observation']):
            action = demos['Action'][idx]
            if action != 'terminal':
                torch_state = torch.tensor(np.fromstring(state.replace(
                    '[', '').replace(']', ''), sep=' ', dtype='float32'), dtype=torch.float32)
                torch_action = torch.tensor(int(action), dtype=torch.uint8)
                self.data.append((torch_state, torch_action))

    def build_from_hems(self, list_s, list_a):
        for idx, state in enumerate(list_s):
            action = list_a[idx]
            torch_state = torch.tensor([state], dtype=torch.float32)
            torch_action = torch.tensor(action, dtype=torch.uint8)
            self.data.append((torch_state, torch_action))

    def build_from_hems_csv(self, demos):
        for hems_sample in demos['sample']:
            print(type(hems_sample))
            _, o, a = dissect_hems_sample(hems_sample)
            if o is None or a is None:
                continue
            torch_state = torch.tensor([o], dtype=torch.float32)
            torch_action = torch.tensor(a, dtype=torch.uint8)
            print(f'state: {o}, action: {a}')
            self.data.append((torch_state, torch_action))

    def merge_with(self, dataset):
        self.data += dataset.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, action = self.data[idx]
        return state, action


def dissect_hems_sample(sample):
    state = None
    state_dict = {}
    observation = None
    obs_dict = {}
    action = None

    if not isinstance(sample, Cons):
        return state, observation, action

    # sort out state into the state dict
    obs_state_act = sample.car
    if not isinstance(obs_state_act.cdr[0], str):
        # state is the car
        state_list = obs_state_act.car
        i = 0
        while True:
            try:
                item = state_list[i]
            except:
                break
            else:
                if repr(item).startswith("DottedList("):
                    state_dict[state_list[i].car] = state_list[i].cdr
                i += 1
        obs_act = obs_state_act.cdr
    else:
        obs_act = obs_state_act

    # Sort off the action
    if obs_act.cdr[0] != 'terminal':
        action = int(obs_act.cdr[0])

    # Build the observation dictionary
    obs_list = obs_act.car
    i = 0
    while True:
        try:
            item = obs_list[i]
        except:
            break
        else:
            if repr(item).startswith("DottedList("):
                obs_dict[obs_list[i].car] = obs_list[i].cdr
            i += 1

    # TODO: Convert state
    # if len(state_dict) > 0:
    #     for key, value in state_dict:

    # Convert observation
    if len(obs_dict) > 0:
        obs_num = None
        if len(obs_dict.items()) == 1:
            for key, value in obs_dict.items():
                if value == "NA":
                    continue
                elif obs_num == None:
                    obs_num = int(value)
                elif obs_num != -1:
                    obs_num = None
        '''
        for key, value in obs_dict.items():
            if value == "NA":
                return state, observation, action
            if "VAR1" in key:
                obs_num += int(value)
            elif "VAR2" in key:
                obs_num += 10 * int(value)
            elif "VAR3" in key:
                obs_num += 100 * int(value)
            else:
                raise f"{key} is too large and unsupported"
        '''
        # if obs_num == 0.:
        #print(f'obs_dict: {obs_dict}, obs_num: {obs_num}')
        observation = obs_num
    print(f"Sampled state: {state}, observation: {observation}, action: {action}")
    return state, observation, action

def sample_obs(hems_inst, obs, n_samples=1):
    with tempfile.NamedTemporaryFile() as fp:
        p = inflect.engine()
        observed = p.number_to_words(obs).replace('-', '_').replace(' ', '_').upper()
        fp.write(bytes(f"c1 = (percept-node {observed}_1 :value \"{obs}\")\n", 'utf-8'))
        fp.write(bytes(f"c2 = (relation-node NUMBER_1 :value \"{obs}\")\n", 'utf-8'))
        fp.write(bytes(f"c2 -> c1\n", 'utf-8'))
        fp.seek(0)
        observation_bn = hems_inst.compile_program_from_file(fp.name)
    observations = []
    actions = []
    action_counts = dict()
    obs_counts = dict()
    failures = 0
    count = 0
    while (len(observations) < n_samples):
        (evidence_bn, backlinks) = hems.make_temporal_episode_retrieval_cue(hems.get_eltm(), observation=observation_bn)
        hems_sample = hems_inst.py_conditional_sample(hems_inst.get_eltm(
        ), evidence_bn, "state-transitions", hiddenstatep=True, outputperceptsp=True, backlinks=backlinks)

        # convert
        _, obs, act = dissect_hems_sample(hems_sample)
        if (obs is None) or (act is None):
            continue

        observations.append(obs)
        actions.append(act)

        if act in action_counts:
            action_counts[act] = action_counts[act] + 1
        else:
            action_counts[act] = 1

        if obs in obs_counts:
            obs_counts[obs] = obs_counts[obs] + 1
        else:
            obs_counts[obs] = 1

    return observations, actions, action_counts, obs_counts

def sample_obs_from_action(hems_inst, action_name, n_samples=1):
    with tempfile.NamedTemporaryFile() as fp:
        fp.write(bytes(f"c1 = (percept-node action :value \"{str(action_name)}\")\n", 'utf-8'))
        fp.seek(0)
        evidence_bn = hems_inst.compile_program_from_file(fp.name)

    observations = []
    actions = []
    action_counts = dict()
    obs_counts = dict()
    failures = 0
    count = 0
    while (len(observations) < n_samples):
        hems_sample = hems_inst.py_conditional_sample(hems_inst.get_eltm(
        ), evidence_bn, "state-transitions", hiddenstatep=True, outputperceptsp=True)

        # convert
        _, obs, act = dissect_hems_sample(hems_sample)
        if (obs is None) or (act is None):
            continue

        observations.append(obs)
        actions.append(act)

        if act in action_counts:
            action_counts[act] = action_counts[act] + 1
        else:
            action_counts[act] = 1

        if obs in obs_counts:
            obs_counts[obs] = obs_counts[obs] + 1
        else:
            obs_counts[obs] = 1

    return observations, actions, action_counts, obs_counts


def sample_from_hems(hems_inst, n_samples):
    print("Generating HEMS samples")
    observations = []
    actions = []
    action_counts = dict()
    obs_counts = dict()
    obs_act_map = dict()
    obs_act_map[0] = 0
    obs_act_map[1] = 3
    obs_act_map[2] = 1
    obs_act_map[3] = 3
    obs_act_map[4] = 0
    obs_act_map[6] = 0
    obs_act_map[8] = 3
    obs_act_map[9] = 1
    obs_act_map[10] = 0
    obs_act_map[13] = 2
    obs_act_map[14] = 1
    while (len(observations) < n_samples):
        hems_sample = hems_inst.py_sample(hems_inst._car(hems_inst.get_eltm()),
                                          hiddenstatep=True, outputperceptsp=True)
        _, obs, act = dissect_hems_sample(hems_sample)
        if (obs is None) or (act is None):
            continue
        #if int(act) != obs_act_map[int(obs)]:
        #    print(f"reference observation: {int(obs)}")
        #    print(f"reference action: {obs_act_map[int(obs)]}")
        #    print(f"inferred action: {int(act)}")
        #    print("failure")
        #    breakpoint()
        observations.append(obs)
        actions.append(act)

        if act in action_counts:
            action_counts[act] = action_counts[act] + 1
        else:
            action_counts[act] = 1

        if obs in obs_counts:
            obs_counts[obs] = obs_counts[obs] + 1
        else:
            obs_counts[obs] = 1

    return observations, actions, action_counts, obs_counts

'''
Balancing Observation Samples
sampled state distribution
{4: 1135, 0: 969, 8: 762, 14: 205, 13: 218, 9: 361, 2: 87, 10: 107, 3: 78, 6: 40, 1: 38}
'''
def balance_observation_samples (hems_inst, observations, actions, obs_counts, training_obs):
    print("Balancing Observation Samples")
    max_act = -1
    max_obs = -1
    new_observations = observations
    new_actions = actions
    for obs, count in obs_counts.items():
        if count > max_obs:
            max_obs = count

    obs_act_map = dict()
    obs_act_map[0] = 0
    obs_act_map[1] = 3
    obs_act_map[2] = 1
    obs_act_map[3] = 3
    obs_act_map[4] = 0
    obs_act_map[6] = 0
    obs_act_map[8] = 3
    obs_act_map[9] = 1
    obs_act_map[10] = 0
    obs_act_map[13] = 2
    obs_act_map[14] = 1
    repeat = True
    while repeat:
        for obs in training_obs:
            obs = int(obs)
            if obs not in obs_counts:
                obs_counts[obs] = 0
            count = obs_counts[obs]
            obs_dif = (max_obs - count)
            if obs_dif > 0:
                new_obs, new_acts, act_c, obs_c = sample_obs(hems_inst, obs, 1)
                for (new_ob, new_act) in zip(new_obs, new_acts):
                    print(f"Upsampling {new_ob} observation.")
                    print(f"Action: {new_act}")
                    print()
                    #if int(new_act) != obs_act_map[int(new_ob)]:
                    #    print(f"reference observation: {int(new_ob)}")
                    #    print(f"reference action: {obs_act_map[int(new_ob)]}")
                    #    print(f"inferred action: {int(new_act)}")
                    #    print("failure")
                    #    breakpoint()
                    new_observations.append(new_ob)
                    new_actions.append(new_act)
                    obs_counts[new_ob] = obs_counts[new_ob] + 1
        for o, c in obs_counts.items():
            if c < max_obs:
                repeat = True
                break
            else:
                repeat = False
    print(obs_counts)
    return new_observations, new_actions

def balance_action_samples(hems_inst, observations, actions, action_counts, obs_counts):
    print("Balancing Action Samples")
    max_act = -1
    max_obs = -1
    new_observations = observations
    new_actions = actions
    for act, count in action_counts.items():
        if count > max_act:
            max_act = count
    for act, count in action_counts.items():
        print(f"Upsampling {act} action.")
        diff = max_act - count
        while diff > 0:
            new_obs, new_acts, _ = sample_obs_from_action(hems_inst, act, diff)
            new_observations += new_obs
            new_actions += new_acts
            diff -= len(new_acts)
            action_counts[act] = action_counts[act] + len(new_obs)
    print(action_counts)
    return new_observations, new_actions

def get_oracle(
        str_env_name: str,
        algo: str,
        folder: str,
        r_seed: int = 0,
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
        r_seed : int = 0
            Random seed for seeding the environment.
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
    
    stats_path = os.path.join(log_path, env_name)
    hyperparams, maybe_stats_path = get_saved_hyperparams(
        stats_path, norm_reward=norm_reward, test_mode=True)

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

    return ALGOS[algo].load(model_path, custom_objects=custom_objects, device=device, **kwargs)

def train_with_bc(policy: NNPolicy, dataset: ImitationDataset, num_epochs: int):
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
    #loader = DataLoader(dataset, batch_size=None, shuffle=True, num_workers=4)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)  # , weight_decay=0.001)
    criterion = nn.CrossEntropyLoss()

    # TRAIN POLICY
    print("Epoch,Batch,Loss")
    epoch_losses = []
    times = []
    for epoch in range(num_epochs):
        running_loss = 0
        epoch_loss = 0
        start = time.time()
        for i, data in enumerate(loader):
            s, a = data
            policy_dist = policy(s)
            #loss = criterion(policy_dist.probs, a)
            loss = criterion(policy_dist, a)
            running_loss += loss.item()
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            if (epoch % 20) == 0 and (i % 100 == 0):
                #print(f'Epoch:{epoch} Batch:{i+1} Loss:{running_loss/20}')
                print(f'{epoch},{i+1},{running_loss/20}')
                running_loss = 0
            optimizer.step()
        end = time.time()
        times.append(end - start)
        epoch_losses.append(epoch_loss)

    epoch_training_loss = pd.DataFrame({"Epoch": range(num_epochs), "Loss": epoch_losses, "Elapsed Time": times})

    return policy, epoch_training_loss

def randints(count, *randint_args):
    ri = partial(randint, *randint_args)
    return [ri() for _ in range(count)]


if __name__ == "__main__":
    # Collect arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-expert",
                        action="store_true", default=False,
                        help="Train policy on only expert data.")
    parser.add_argument("--train-hems",
                        action="store_true", default=False,
                        help="Train policy on only HEMS-generated data.")
    parser.add_argument("--train-expert-hems",
                        action="store_true", default=False,
                        help="Train policy on expert data and then continue training with HEMS \
                            data.")
    parser.add_argument("--evaluate-oracle",
                        action="store_true", default=False,
                        help="Train policy on expert data and then continue training with HEMS \
                            data.")
    parser.add_argument("--train-hems-expert",
                        action="store_true", default=False,
                        help="Train policy on HEMS data and then continue training with expert \
                            data.")
    parser.add_argument("--train-both",
                        action="store_true", default=False,
                        help="Train policy on combined expert and HEMS data.")
    parser.add_argument("--train-sampled-hems",
                        action="store_true", default=False,
                        help="Train policy on combined expert and HEMS data.")
    parser.add_argument("--algo",
                        default="ppo", type=str,
                        help="RL algorithm used to train the expert.")
    parser.add_argument("--env",
                        default="CliffWalking-v0",  # "Taxi-v3",  # "FrozenLake-v1",
                        type=str,
                        help="Target environment.")
    parser.add_argument("--n-epochs",
                        default=300, type=int,
                        help="The number of epochs for training. If training with a sequential \
                            option, the number of epochs will be evenly split between the two.")
    parser.add_argument("--random-seed",
                        default=8, type=int,
                        help="The random seed to use for the training.")
    parser.add_argument("--load",
                        default=None, type=str,
                        help="Run all algorithms in all environments as described in docs.")
    parser.add_argument("--eval",
                        action="store_true", default=False,
                        help="Run all algorithms in all environments as described in docs.")
    parser.add_argument("--test",
                        action="store_true", default=False,
                        help="Run all algorithms in all environments as described in docs.")
    parser.add_argument("--render",
                        action="store_true", default=False,
                        help="Render the evaluation.")
    args = parser.parse_args()

    ENV_NAME = args.env
    ALGO = args.algo
    HEMS_DIR = os.path.join('./hems_samples', 'samples 1.csv')
    HEMS_MODEL_DIR = os.path.join("./HEMS_model", ALGO+'_'+ENV_NAME+"/","eltm.txt")
    RENDER = args.render
    N_EPOCHS = args.n_epochs
    TOY_TEXT_BOOL = False
    NUM_HEMS_SAMPLES = 5000
    MODEL_SAVE_LOC = "./bc_trained_agents/"
    LOG_SAV_LOC = "./bc_training_logs/"
    performance_name = None

    # SETUP ENV
    TOY_TEXT_ENV_NAMES = ["Blackjack-v1", "CliffWalking-v0", "FrozenLake-v1", "Taxi-v3"]
    
    # SORT ARGS AsteroidsNoFrameskip-v4
    all_seeds=[]
    all_agent_types=[]
    all_returns=[]
    all_lengths=[]
    all_mode_action=[]
    all_states = []
    all_counts = []
    all_sds = []
    all_eps = []
    for seed in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]:#randints(5, 1, 100):
        args.random_seed = seed
        for agent in ['Expert']:#['HEMS', 'Baseline', 'Expert']:
            if agent == 'Baseline':
                args.train_expert = True
                args.train_hems = False
                args.train_expert_hems = False
                args.evaluate_oracle = False
                args.train_hems_expert = False
                args.train_both = False
                args.train_sampled_hems = False
            elif agent == 'HEMS':
                args.train_expert = False
                args.train_hems = True
                args.train_expert_hems = False
                args.evaluate_oracle = False
                args.train_hems_expert = False
                args.train_both = False
                args.train_sampled_hems = False
            elif agent == 'Expert':
                args.train_expert = False
                args.train_hems = False
                args.train_expert_hems = False
                args.evaluate_oracle = True
                args.train_hems_expert = False
                args.train_both = False
                args.train_sampled_hems = False
            for ep_data in ['./ep_data_1', './ep_data_2', './ep_data_3', './ep_data_4', './ep_data_5', './ep_data_6', './ep_data_7', './ep_data_8', './ep_data_9', './ep_data_10']:#['./ep_data_100', './ep_data_200', './ep_data_300', './ep_data_400', './ep_data_500', './ep_data_600', './ep_data_700', './ep_data_800', './ep_data_900', './ep_data_1000']:
                DEMO_DIR = os.path.join(ep_data, ALGO+'_'+ENV_NAME+'_data.csv')
                # Set random seeds
                torch.manual_seed(args.random_seed)
                np.random.seed(args.random_seed)
                
                # SETUP HEMS
                # get a handle to the lisp subprocess with quicklisp loaded.
                lisp = cl4py.Lisp(cmd=('sbcl', '--dynamic-space-size', '30000',
                                       '--script'), quicklisp=True, backtrace=True)
                
                # Start quicklisp and import HEMS package
                lisp.find_package('QL').quickload('HEMS')
                
                # load hems and retain reference.
                hems = lisp.find_package("HEMS")
                
                if ENV_NAME in TOY_TEXT_ENV_NAMES:
                    # Toy Text
                    TOY_TEXT_BOOL = True
                    env = gym.make(ENV_NAME)
                    pi = NNPolicy(1, 32, env.action_space.n)
                else:
                    # Atari
                    env = gym.make(ENV_NAME, obs_type="ram", render_mode='human')
                    pi = NNPolicy(env.observation_space.shape[0], 32, env.action_space.n)

                # TRAINING POLICY: Expert data only, no HEMS
                if args.train_expert:
                    # Load expert data
                    demos = pd.read_csv(DEMO_DIR)
                    
                    # Convert to database
                    expert_dataset = ImitationDataset()
                    if TOY_TEXT_BOOL:
                        expert_dataset.build_from_toy_text(demos)
                    else:
                        expert_dataset.build_from_atari(demos)
                    stts = []
                    cnts = []
                    sds = []
                    ep_datas = []
                    # print(expert_dataset.data)
                    # Train on expert database
                    trained_pi, training_data = train_with_bc(pi, expert_dataset, N_EPOCHS)

                    # Save model
                    performance_name = f"expert_trained_{ENV_NAME}"
                    save_path = os.path.join(MODEL_SAVE_LOC, f"{ep_data}_{performance_name}_{seed}.pkl")
                    trained_pi.save(save_path)

                    # Save training data
                    log_path = os.path.join(LOG_SAV_LOC, f"{ep_data}_{performance_name}_{seed}.csv")
                    training_data.to_csv(log_path)
                if args.evaluate_oracle:
                    stts = []
                    cnts = []
                    sds = []
                    ep_datas = []
                # TRAINING POLICY: continued training with HEMS
                if args.train_hems:
                    # Run HEMS model
                    df = pd.read_csv(DEMO_DIR)
                    observations = df["Observation"].unique()
                    for i, obs in enumerate(observations):
                        observations[i]=obs[1:-1]
                    hems.init_eltm()
                    hems.run_execution_trace(DEMO_DIR)

                    # Load HEMS model
                    #print("Loading ELTM")
                    #hems.load_eltm_from_file(HEMS_MODEL_DIR)
                    #print("Done")
                    # Sample from HEMS model
                    obs, acts, act_counts, obs_counts = sample_from_hems(hems, NUM_HEMS_SAMPLES)
                    stts = []
                    cnts = []
                    sds = []
                    ep_datas = []
                    for (obss, count) in obs_counts.items():
                        stts.append(obss)
                        cnts.append(count)
                        sds.append(seed)
                        ep_datas.append(int(ep_data.split('_')[-1]))
                    all_states.extend(stts)
                    all_counts.extend(cnts)
                    all_sds.extend(sds)
                    all_eps.extend(ep_datas)
                    observations, actions = balance_observation_samples(hems, obs, acts, obs_counts, observations)
                    #observations, actions = balance_action_samples(hems, obs, acts, act_counts, obs_counts)
                    
                    # Convert to database
                    hems_dataset = ImitationDataset()
                    hems_dataset.build_from_hems(observations, actions)
                    # print(hems_dataset.data)
                    
                    # Train on database
                    trained_pi, training_data = train_with_bc(pi, hems_dataset, N_EPOCHS)
                    
                    # Save model
                    performance_name = f"hems_trained_{ENV_NAME}"
                    save_path = os.path.join(MODEL_SAVE_LOC, f"{ep_data}_{performance_name}_{seed}.pkl")
                    trained_pi.save(save_path)

                    # Save training data
                    log_path = os.path.join(LOG_SAV_LOC, f"{ep_data}_{performance_name}_{seed}.csv")
                    training_data.to_csv(log_path)

                # TRAINING POLICY: Expert data then HEMS
                if args.train_expert_hems:
                    # Load expert data
                    demos = pd.read_csv(DEMO_DIR)
                    
                    # Convert to database
                    expert_dataset = ImitationDataset()
                    if TOY_TEXT_BOOL:
                        expert_dataset.build_from_toy_text(demos)
                    else:
                        expert_dataset.build_from_atari(demos)

                    # Train on expert database
                    trained_expert_pi, expert_training_data = train_with_bc(pi, expert_dataset, N_EPOCHS/2)
                    
                    # Load HEMS model
                    # hems_model = hems.load_eltm_from_file("filename")
                    hems.run_execution_trace(DEMO_DIR)

                    # Sample from HEMS model
                    obs, acts, act_counts = sample_from_hems(hems, NUM_HEMS_SAMPLES)
                    observations, actions = balance_action_samples(hems, obs, acts, act_counts)

                    # Convert to database
                    hems_dataset = ImitationDataset()
                    hems_dataset.build_from_hems(observations, actions)
                    # print(hems_dataset.data)
                    
                    # Train on HEMS database
                    trained_pi, hems_training_data = train_with_bc(trained_expert_pi, hems_dataset, N_EPOCHS/2)
                    
                    # Save model
                    performance_name = f"expert_then_hems_trained_{ENV_NAME}"
                    save_path = os.path.join(MODEL_SAVE_LOC, f"{ep_data}_{performance_name}_{seed}.pkl")
                    trained_pi.save(save_path)

                    # Save training data
                    log_path = os.path.join(LOG_SAV_LOC, f"{ep_data}_{performance_name}_{seed}.csv")
                    training_data = pd.concat([expert_training_data, hems_training_data])
                    training_data.to_csv(log_path)

                # TRAINING POLICY: Training with HEMS then expert data
                if args.train_hems_expert:
                    # Load HEMS model
                    # hems_model = hems.load_eltm_from_file("filename")
                    hems.run_execution_trace(DEMO_DIR)
                    
                    # Sample from HEMS model
                    obs, acts, act_counts = sample_from_hems(hems, NUM_HEMS_SAMPLES)
                    observations, actions = balance_action_samples(hems, obs, acts, act_counts)
                    
                    # Convert to database
                    hems_dataset = ImitationDataset()
                    hems_dataset.build_from_hems(observations, actions)
                    # print(hems_dataset.data)
                    
                    # Train on HEMS database
                    trained_hems_pi, hems_training_data = train_with_bc(
                        trained_expert_pi, hems_dataset, N_EPOCHS/2)

                    # Load expert data
                    demos = pd.read_csv(DEMO_DIR)
                    
                    # Convert to database
                    expert_dataset = ImitationDataset()
                    if TOY_TEXT_BOOL:
                        expert_dataset.build_from_toy_text(demos)
                    else:
                        expert_dataset.build_from_atari(demos)
                        
                    # Train on expert database
                    trained_pi, expert_training_data = train_with_bc(
                        trained_hems_pi, expert_dataset, N_EPOCHS/2)
                    
                    # Save model
                    performance_name = f"hems_then_expert_trained_{ENV_NAME}"
                    save_path = os.path.join(MODEL_SAVE_LOC, f"{ep_data}_{performance_name}_{seed}.pkl")
                    trained_pi.save(save_path)

                    # Save training data
                    log_path = os.path.join(LOG_SAV_LOC, f"{ep_data}_{performance_name}_{seed}.csv")
                    training_data = pd.concat([hems_training_data, expert_training_data])
                    training_data.to_csv(log_path)
                    
                # TRAINING POLICY: Expert and  HEMS data merged
                if args.train_both:
                    # Load expert data
                    demos = pd.read_csv(DEMO_DIR)
                    
                    # Convert to database
                    expert_dataset = ImitationDataset()
                    if TOY_TEXT_BOOL:
                        expert_dataset.build_from_toy_text(demos)
                    else:
                        expert_dataset.build_from_atari(demos)
                        
                    # Load HEMS model
                    # hems_model = hems.load_eltm_from_file("filename")
                    hems.run_execution_trace(DEMO_DIR)
                    
                    # Sample from HEMS model
                    obs, acts, act_counts = sample_from_hems(hems, NUM_HEMS_SAMPLES)
                    observations, actions = balance_action_samples(hems, obs, acts, act_counts)
                    
                    # Convert to database
                    hems_dataset = ImitationDataset()
                    hems_dataset.build_from_hems(observations, actions)
                    
                    # Merge databases
                    expert_dataset.merge_with(hems_dataset)
                    
                    # Train on database
                    trained_pi, training_data = train_with_bc(pi, expert_dataset, N_EPOCHS)

                    # Save model
                    performance_name = f"expert_and_hems_trained_{ENV_NAME}"
                    save_path = os.path.join(MODEL_SAVE_LOC, f"{ep_data}_{performance_name}_{seed}.pkl")
                    trained_pi.save(save_path)

                    # Save training data
                    log_path = os.path.join(LOG_SAV_LOC, f"{ep_data}_{performance_name}_{seed}.csv")
                    training_data.to_csv(log_path)

                # TRAINING POLICY: Expert data only, no HEMS
                if args.train_sampled_hems:
                    # Load expert data
                    demos = pd.read_csv(HEMS_DIR, index_col=False, names=['sample'])
                    
                    # Convert to database
                    sampled_dataset = ImitationDataset()
                    sampled_dataset.build_from_hems_csv(demos)
                    
                    # print(expert_dataset.data)
                    # Train on expert database
                    trained_pi, training_data = train_with_bc(pi, sampled_dataset, N_EPOCHS)
                    
                    # Save model
                    performance_name = f"sampled_hems_trained_{ENV_NAME}"
                    save_path = os.path.join(MODEL_SAVE_LOC, f"{ep_data}_{performance_name}_{seed}.pkl")
                    trained_pi.save(save_path)

                hems = None
                lisp = None
                # EVALUATE POLICY
                if args.eval:
                    if args.load is not None or args.evaluate_oracle == True:
                        if args.load is not None:
                            load_path = os.path.join(MODEL_SAVE_LOC, args.load)
                            trained_pi = NNPolicy.load(load_path)
                            performance_name = args.load.replace(".pkl", "")
                        elif args.evaluate_oracle == True:
                            # get the expert
                            trained_pi = get_oracle(ENV_NAME, 'ppo', 'rl_experts')
                            performance_name = f"expert_eval_{ENV_NAME}"
                    max_steps = 1000  # env.spec.timestep_limit
                    returns = []
                    seeds = []
                    agent_types = []
                    mode_action = []
                    lengths = []
                    for i in range(1000):
                        print('iter', i)
                        reset_obs = env.reset()
                        obs = reset_obs[0]
                        done = term = False
                        totalr = 0.
                        steps = 0
                        actions = []
                        while (not (done or term)) and steps < max_steps:
                            if args.evaluate_oracle == False:
                                pi_dist = trained_pi(torch.tensor([obs], dtype=torch.float32))
                            # print(f'obs: {obs}, dist: {pi_dist.probs}, mode: {pi_dist.mode.item()}')
                            if ENV_NAME in TOY_TEXT_ENV_NAMES:
                                #a = pi_dist.mode.item()
                                if args.evaluate_oracle == True:
                                    action, _ = trained_pi.predict(obs, deterministic=True)
                                    a = action.item()
                                else:
                                    a = pi_dist.argmax().item()
                            else:
                                a = pi_dist.mode.numpy()[0]
                            actions.append(a)
                            obs, r, done, term, _ = env.step(a)
                            if RENDER:
                                env.render()
                            totalr += r
                            steps += 1
                            # if steps % 100 == 0:
                            #     print("%i/%i" % (steps, max_steps))
                            # if steps >= max_steps:
                            #     break
                        returns.append(totalr)
                        lengths.append(steps)
                        seeds.append(seed)
                        agent_types.append(agent)
                        np_actions = np.array(actions)
                        mode_action.append(stats.mode(np_actions)[0])

                    print('returns', returns)
                    print('mean return', np.mean(returns))
                    print('std of return', np.std(returns))

                    # Save return data
                    all_seeds.extend(seeds)
                    all_agent_types.extend(agent_types)
                    all_returns.extend(returns)
                    all_lengths.extend(lengths)
                    all_mode_action.extend(mode_action)
                    current_df = pd.DataFrame(
                        {"Seed": seeds, "Agent": agent_types, "Return": returns, "Length": lengths, "Most Common Action": mode_action}
                    )
                    dist_df = pd.DataFrame(
                       {"Seed": sds, "Num_Examples": ep_datas, "State": stts, "Count": cnts}
                    )
                    log_path = os.path.join(LOG_SAV_LOC, f"{ep_data}_{performance_name}_{seed}_final_policy_eval.csv")
                    log_st_path = os.path.join(LOG_SAV_LOC, f"{ep_data}_{performance_name}_{seed}_final_state_distribution.csv")
                    current_df.to_csv(log_path)
                    dist_df.to_csv(log_st_path)
    #log_path = os.path.join(LOG_SAV_LOC, f"{performance_name}_final_policy_eval.csv")
    #log_st_path = os.path.join(LOG_SAV_LOC, f"{performance_name}_final_state_distribution.csv")
    #returns_data = pd.DataFrame(
    #    {"Seed": all_seeds,"Agent": all_agent_types, "Return": all_returns, "Length": all_lengths, "Most Common Action": all_mode_action})
    #returns_data.to_csv(log_path)
    #state_dist_data = pd.DataFrame(
    #    {"Seed": all_sds, "Num_Examples": all_eps, "State": all_states, "Count": all_counts})
    #state_dist_data.to_csv(log_st_path)
    # JUST TESTING STUFF
    if args.test:
        hems.run_execution_trace(DEMO_DIR)
        
        # Sample from HEMS model
        observations, actions, action_counts = sample_obs_from_action(hems, '1', NUM_HEMS_SAMPLES)
