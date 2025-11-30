import os
import random
import gymnasium as gym

from collections import deque
from itertools import count
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import load_config
from test import EvaluateDQN

from agent.memory import ReplayMemory
from agent.agent import build_agent
from agent.env import build_env

# -------------------------
# SETUP
# -------------------------
cfg = load_config()

# log_env, env = build_env()

logs_envs = []
num_envs = 4

def make_env():
    log_env, env = build_env()
    logs_envs.append(log_env)
    return env

env_fns = [make_env for _ in range(num_envs)]
env = gym.vector.AsyncVectorEnv(
    env_fns = env_fns,
    shared_memory = True # Images are big so more efficient
    # copy = False,
)

# -------------------------
# REPRODUCTABILITY
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

memory = ReplayMemory(cfg["env"]["memory_capacity"])
model = build_agent(memory, env.action_space.nvec[0], device)
evaluation = EvaluateDQN(model)

# -------------------------
# Q Values states
# -------------------------
eval_states = []

for log_env in logs_envs:
    log_env.set_logging(False)

_, _ = env.reset()
for i in range(cfg["eval"]["q_eval_states_count"]):
    act = env.action_space.sample()
    obs, _, _, _, _ = env.step(act)
    eval_states.append(torch.tensor(obs))

eval_states = torch.cat(eval_states).to(device)

for log_env in logs_envs:
    log_env.set_model(model, eval_states)

# -------------------------
# Main Loop
# -------------------------
done = [True for _ in range(4)]
end_ep = [False for _ in range(4)]
stuck = np.zeros(4)
total_reward = np.zeros(4)
best = 0

for i_step in range(cfg["train"]["num_steps"]):

    # Avoid useless logging
    if i_step > cfg["train"]["replay_start"]:
        for log_env in logs_envs:
            log_env.set_logging(True)

    for i in range(4):
        if done[i]:
            done = False
            total_reward[i] = 0

            stuck[i] = 0

    print(obs.shape)
    
    action = model.action(i_step, obs)

    print(action, action.shape)

    next_obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    if cfg["env"]["clip_reward"]:
        reward = np.sign(reward)

    print(info)

    for i in range(4):

        # Bool test
        loss_life = info[i]["lose_life"]
        done[i] = terminated[i] or truncated[i]
        end_ep[i] = done[i] or (loss_life and cfg["env"]["episodic_life"])

        if reward[i] != 0:
            stuck[i] = 0
        else:
            stuck[i] += 1

        if cfg["env"]["looping_penalty"] and stuck[i] > 150:
            reward[i] = -1

        if loss_life and cfg["env"]["lose_life_penalty"]:
            reward[i] = -1

    next_obs = torch.tensor(next_obs)
    memory.push(obs, action, torch.tensor(reward, dtype=torch.float32), next_obs, end_ep)
    obs = next_obs

    if max(total_reward) > best:
        # save model
        best = max(total_reward)

        if best > 400:
            path = cfg["env"]["mod_name"] + "/checkpoints"
            os.makedirs(path, exist_ok=True)
            torch.save(model.policy_net.state_dict(), f"{path}/best-train-{best}.pth")

    if i_step % cfg["train"]["optimize_freq"] == 0:
        model.optimize()

    if i_step % cfg["train"]["target_update_freq"] == 0:
        model.update_target_network()

    if len(memory) >= cfg["train"]["replay_start"] and i_step % cfg["eval"]["eval_freq"] == 0:
        evaluation.evaluate(eval_states)

env.close()
