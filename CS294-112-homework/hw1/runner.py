import pickle
import numpy as np 
import gym 
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('envname', type=str)

args = parser.parse_args()
envname = args.envname

with open(os.path.join('expert_data', envname + '.pkl'), 'rb') as f:
    expert_data = pickle.load(f)
# data for behavioural cloning
X = expert_data['observations']
Y = expert_data['actions']

from model import get_trained_model
model, metrics = get_trained_model(X, Y)

print(metrics.history['loss'][-1], metrics.history['val_loss'][-1])

env = gym.make(envname)

num_rollouts = 3
max_steps = 1000
render = True

returns = []

for i in range(num_rollouts):
    obs = env.reset()
    done = False
    totalr = 0
    steps = 0
    while (not done) and steps <= max_steps:
        action = model.predict(obs[None, :])
        obs, r, done, _ = env.step(action)
        totalr += r
        steps += 1
        if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
        env.render()
    
    returns.append(totalr)

print(envname + " Mean and Std Returns: ")
print(np.mean(returns), np.std(returns))

runner_data = {
    'mean_returns': np.mean(returns),
    'std_returns': np.std(returns),
    'loss': metrics.history['loss'][-1], 
    'val_loss': metrics.history['val_loss'][-1]
}

with open(os.path.join('bc_data', args.envname + '.pkl'), 'wb') as f:
    pickle.dump(runner_data, f, pickle.HIGHEST_PROTOCOL)
