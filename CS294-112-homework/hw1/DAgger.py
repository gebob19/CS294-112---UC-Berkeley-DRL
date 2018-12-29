#!/usr/bin/env python
# coding: utf-8

import pickle
import numpy as np 
import gym 
import os
import load_policy
from model import get_trained_model
from sklearn.model_selection import train_test_split
import tensorflow as tf

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('envname', type=str)

args = parser.parse_args()
envname = args.envname

# ## Train Model for agent policy
with open(os.path.join('expert_data', envname + '.pkl'), 'rb') as f:
    expert_data = pickle.load(f)
# data for behavioural cloning
X = expert_data['observations']
Y = expert_data['actions']

model, metrics = get_trained_model(X, Y)

# ## Perform rollout and record agents observations
env = gym.make(envname)

def rollout(n, max_steps):
    observations = []
    returns = []

    for i in range(n):
        obs = env.reset()
        done = False
        totalr = 0
        steps = 0
        while (not done) and steps <= max_steps:
            action = model.predict(obs[None, :])
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            # record observation
            observations.append(obs)
            env.render()
        returns.append(totalr)
    return np.mean(returns), np.std(returns), observations

_, _, observations = rollout(2, 1000)

# ## Label the Agents Observations to the Experts Action
policy_fn = load_policy.load_policy('experts/'+ envname + '.pkl')

actions = []
with tf.Session():
    tf.global_variables_initializer()
    
    for obs in observations:
        action = policy_fn(obs[None, :])
        actions.append(action)

X_train, X_val, Y_train, Y_val = train_test_split(np.asarray(observations),
                                                  np.asarray(actions), test_size=0.1)

metrics = model.fit(X_train, 
            Y_train,
            epochs=50,
            batch_size=64,
            verbose=0,
            validation_data=(X_val, Y_val))

# # Rollouts
rmean, rstd, _ = rollout(2, 1000)

runner_data = {
    'mean_returns': rmean,
    'std_returns': rstd,
    'loss': metrics.history['loss'][-1], 
    'val_loss': metrics.history['val_loss'][-1]
}

with open(os.path.join('dagger_data', args.envname + '.pkl'), 'wb') as f:
    pickle.dump(runner_data, f, pickle.HIGHEST_PROTOCOL)







