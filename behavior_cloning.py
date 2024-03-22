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


class NNPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(NNPolicy, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.linear_1 = nn.Linear(state_dim, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = self.linear_1(x)
        x = F.leaky_relu(x, 0.001)
        x = self.linear_2(x)
        x = F.leaky_relu(x, 0.001)
        logits = self.linear_out(x)
        return Categorical(logits=logits)

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
    obs_num = 0
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
    observation = obs_num

    return state, observation, action


def sample_obs_from_action(hems_inst, action_name, n_samples=1000):
    with tempfile.NamedTemporaryFile() as fp:
        fp.write(bytes(f"c1 = (percept-node action :value \"{str(action_name)}\")\n", 'utf-8'))
        fp.seek(0)
        evidence_bn = hems_inst.compile_program_from_file(fp.name)

    observations = []
    actions = []
    action_counts = dict()
    failures = 0
    count = 0
    while (len(observations) < n_samples) and (failures < n_samples):
        hems_sample = hems_inst.py_conditional_sample(hems_inst.get_eltm(
        ), evidence_bn, "state-transitions", hiddenstatep=True, outputperceptsp=True)

        # convert
        _, obs, act = dissect_hems_sample(hems_sample)
        if (obs is None) or (act is None):
            failures += 1
            continue

        observations.append(obs)
        actions.append(act)

        if act in action_counts:
            action_counts[act] = action_counts[act] + 1
        else:
            action_counts[act] = 1

    return observations, actions, action_counts


def sample_from_hems(hems_inst, n_samples):
    observations = []
    actions = []
    action_counts = dict()
    failures = 0
    while (len(observations) < n_samples) and (failures < n_samples):
        hems_sample = hems_inst.py_sample(hems_inst._car(hems_inst.get_eltm()),
                                          hiddenstatep=True, outputperceptsp=True)
        _, obs, act = dissect_hems_sample(hems_sample)
        if (obs is None) or (act is None):
            failures += 1
            continue

        observations.append(obs)
        actions.append(act)
        
        if act in action_counts:
            action_counts[act] = action_counts[act] + 1
        else:
            action_counts[act] = 1

    return observations, actions, action_counts

def balance_action_samples(hems_inst, observations, actions, action_counts):
    max_act = -1
    new_observations = observations
    new_actions = actions
    for act, count in action_counts.items():
        if count > max_act:
            max_act = count
    for act, count in action_counts.items():
        diff = max_act - count
        if diff > 0:
            new_obs, new_acts, _ = sample_obs_from_action(hems_inst, act, diff)
            new_observations = new_observations + new_obs
            new_actions = new_actions + new_acts
            action_counts[act] = action_counts[act] + diff
    return new_observations, new_actions

def train_with_bc(policy: NNPolicy, dataset: ImitationDataset, num_epochs: int):
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3, weight_decay=0.0001)
    criterion = nn.CrossEntropyLoss()

    # TRAIN POLICY
    for epoch in range(num_epochs):
        running_loss = 0
        for i, data in enumerate(loader):
            s, a = data
            optimizer.zero_grad()
            policy_dist = policy(s)
            loss = criterion(policy_dist.probs, a)
            running_loss += loss.item()
            loss.backward()
            if (epoch % 20) == 0 and (i % 100 == 0):
                print(f'Epoch:{epoch} Batch:{i+1} Loss:{running_loss/20}')
                running_loss = 0
            optimizer.step()
    return policy


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
    parser.add_argument("--train-hems-expert",
                        action="store_true", default=False,
                        help="Train policy on HEMS data and then continue training with expert \
                            data.")
    parser.add_argument("--train-both",
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
    # SORT ARGS AsteroidsNoFrameskip-v4
    ENV_NAME = args.env
    ALGO = args.algo
    DEMO_DIR = os.path.join('./ep_data_1', ALGO+'_'+ENV_NAME+'_data.csv')
    RENDER = args.render
    N_EPOCHS = args.n_epochs
    TOY_TEXT_BOOL = False
    NUM_HEMS_SAMPLES = 2000
    MODEL_SAVE_LOC = "./bc_trained_agents/"
    performance_name = None

    # SETUP HEMS
    # get a handle to the lisp subprocess with quicklisp loaded.
    lisp = cl4py.Lisp(quicklisp=True, backtrace=True)

    # Start quicklisp and import HEMS package
    lisp.find_package('QL').quickload('HEMS')

    # load hems and retain reference.
    hems = lisp.find_package("HEMS")

    # SETUP ENV
    TOY_TEXT_ENV_NAMES = ["Blackjack-v1", "CliffWalking-v0", "FrozenLake-v1", "Taxi-v3"]
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

        # print(expert_dataset.data)
        # Train on expert database
        trained_pi = train_with_bc(pi, expert_dataset, N_EPOCHS)

        # Save model
        performance_name = f"expert_trained_{ENV_NAME}"
        save_path = os.path.join(MODEL_SAVE_LOC, f"{performance_name}.pkl")
        trained_pi.save(save_path)

    # TRAINING POLICY: continued training with HEMS
    if args.train_hems:
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

        # Train on database
        trained_pi = train_with_bc(pi, hems_dataset, N_EPOCHS)

        # Save model
        performance_name = f"hems_trained_{ENV_NAME}"
        save_path = os.path.join(MODEL_SAVE_LOC, f"{performance_name}.pkl")
        trained_pi.save(save_path)

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
        trained_expert_pi = train_with_bc(pi, expert_dataset, N_EPOCHS/2)

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
        trained_pi = train_with_bc(trained_expert_pi, hems_dataset, N_EPOCHS/2)

        # Save model
        performance_name = f"expert_then_hems_trained_{ENV_NAME}"
        save_path = os.path.join(MODEL_SAVE_LOC, f"{performance_name}.pkl")
        trained_pi.save(save_path)

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
        trained_hems_pi = train_with_bc(trained_expert_pi, hems_dataset, N_EPOCHS/2)

        # Load expert data
        demos = pd.read_csv(DEMO_DIR)

        # Convert to database
        expert_dataset = ImitationDataset()
        if TOY_TEXT_BOOL:
            expert_dataset.build_from_toy_text(demos)
        else:
            expert_dataset.build_from_atari(demos)

        # Train on expert database
        trained_pi = train_with_bc(trained_hems_pi, expert_dataset, N_EPOCHS/2)

        # Save model
        performance_name = f"hems_then_expert_trained_{ENV_NAME}"
        save_path = os.path.join(MODEL_SAVE_LOC, f"{performance_name}.pkl")
        trained_pi.save(save_path)

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
        trained_pi = train_with_bc(pi, expert_dataset, N_EPOCHS)

        # Save model
        performance_name = f"expert_and_hems_trained_{ENV_NAME}"
        save_path = os.path.join(MODEL_SAVE_LOC, f"{performance_name}.pkl")
        trained_pi.save(save_path)

    # EVALUATE POLICY
    if args.eval:
        if args.load is not None:
            load_path = os.path.join(MODEL_SAVE_LOC, args.load)
            trained_pi = NNPolicy.load(load_path)
            performance_name = args.load.replace(".pkl", "")
        max_steps = 1000  # env.spec.timestep_limit
        returns = []
        for i in range(10):
            print('iter', i)
            reset_obs = env.reset()
            obs = reset_obs[0]
            done = term = False
            totalr = 0.
            steps = 0
            while (not (done or term)) and steps < max_steps:
                pi_dist = trained_pi(torch.tensor([obs], dtype=torch.float32))
                if ENV_NAME in TOY_TEXT_ENV_NAMES:
                    a = pi_dist.mode.item()
                else:
                    a = pi_dist.mode.numpy()[0]
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

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

    # JUST TESTING STUFF
    if args.test:
        hems.run_execution_trace(DEMO_DIR)

        # Sample from HEMS model
        observations, actions, action_counts = sample_obs_from_action(hems, '1', NUM_HEMS_SAMPLES)
