import gym
import numpy as np
import torch
import torch.nn as nn


def build_model(dim_in, H, dim_out):
    return nn.Sequential(
        nn.Linear(dim_in, H),
        nn.ReLU(),
        nn.Linear(H, H),
        nn.ReLU(),
        nn.Linear(H, H),
        nn.ReLU(),
        nn.Linear(H, dim_out)
    )

env = gym.make('LunarLander-v2')

dim_in = env.observation_space.shape[0]
dim_out = env.action_space.n
H = 100

Qfunc = build_model(dim_in, H, dim_out)
Qfunc.load_state_dict(torch.load('./weights'))

done = False
obs = env.reset()
while not done:
    env.render()
    act = Qfunc(torch.tensor(np.asarray(obs, dtype=np.float32)))
    greedy_act = torch.argmax(act).item()
    obs, reward, done, _ = env.step(greedy_act)
