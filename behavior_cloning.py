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
import pandas as pd
import gymnasium as gym
import numpy as np
import os
import torch.optim as optim
from torch.utils.data import DataLoader


class NNPolicy(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(NNPolicy, self).__init__()
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


class ImitationDataset(Dataset):
    def __init__(self, demos):
        self.data = []
        for idx, state in enumerate(demos['RAM State']):
            action = demos['Action'][idx]  # [0]
            if action != 'terminal':
                np_state = torch.tensor(np.fromstring(state.replace(
                    '[', '').replace(']', ''), sep=' ', dtype='float32'), dtype=torch.float32)
                np_action = torch.tensor(np.array(action, dtype='uint8'), dtype=torch.uint8)
                self.data.append((np_state, np_action))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state, action = self.data[idx]
        return state, action


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
    parser.add_argument("--train-expert", action="store_true", default=False,
                        help="Run all algorithms in all environments as described in docs.")
    parser.add_argument("--eval", action="store_true", default=False,
                        help="Run all algorithms in all environments as described in docs.")
    args = parser.parse_args()
    # SORT ARGS
    ENV_NAME = 'AsteroidsNoFrameskip-v4'
    ALGO = 'ppo'
    DEMO_DIR = os.path.join('./logs_100', ALGO+'_'+ENV_NAME+'_data.csv')
    RENDER = False

    # PREPARE DATA
    demos = pd.read_csv(DEMO_DIR)

    env = gym.make(ENV_NAME, obs_type="ram")  # , render_mode='human')

    pi = NNPolicy(env.observation_space.shape[0], 32, env.action_space.n)

    # TRAINING POLICY: Expert data, no HEMS
    if args.train_expert:
        expert_dataset = ImitationDataset(demos)
        trained_pi = train_with_bc(pi, expert_dataset, 300)

    # EVALUATE POLICY
    if args.eval:
        max_steps = 1000  # env.spec.timestep_limit
        returns = []
        for i in range(10):
            print('iter', i)
            reset_obs = env.reset()
            obs = reset_obs[0]
            done = term = False
            totalr = 0.
            steps = 0
            while not (done or term):
                pi_dist = trained_pi(torch.tensor([obs], dtype=torch.float32))
                a = pi_dist.mode
                obs, r, done, term, _ = env.step(a.numpy()[0])
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
