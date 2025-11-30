import os
import random
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

log_env, env = build_env()

# -------------------------
# REPRODUCTABILITY
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

memory = ReplayMemory(cfg["env"]["memory_capacity"])
model = build_agent(memory, env.action_space.n, device)
evaluation = EvaluateDQN(model)

# -------------------------
# Q Values states
# -------------------------
eval_states = []

log_env.set_logging(False)

_, _ = env.reset()
for i in range(cfg["eval"]["q_eval_states_count"]):
    act = random.randint(0, 3)
    obs, _, _, _, _ = env.step(act)
    eval_states.append(torch.tensor(obs).unsqueeze(0))

eval_states = torch.cat(eval_states).to(device)

log_env.set_model(model, eval_states)

# -------------------------
# Main Loop
# -------------------------
done = True
end_ep = False
stuck = 0
total_reward = 0
best = 0

for i_step in range(cfg["train"]["num_steps"]):

    # Avoid useless logging
    if i_step > cfg["train"]["replay_start"]:
        log_env.set_logging(True)

    if done:
        obs, _ = env.reset()
        obs = torch.tensor(obs)
        done = False
        total_reward = 0

        stuck = 0
    
    action = model.action(i_step, obs)

    next_obs, reward, terminated, truncated, info = env.step(action.item())

    total_reward += reward

    if cfg["env"]["clip_reward"]:
        reward = np.sign(reward) # Clip for learning

    # Bool test
    loss_life = info["lose_life"]
    done = terminated or truncated
    end_ep = done or (loss_life and cfg["env"]["episodic_life"])

    if reward != 0:
        stuck = 0
    else:
        stuck += 1

    if cfg["env"]["looping_penalty"] and stuck > 150:
        reward = -1

    if loss_life and cfg["env"]["lose_life_penalty"]:
        reward = -1

    next_obs = torch.tensor(next_obs)
    memory.push(obs, action, torch.tensor(reward, dtype=torch.float32), next_obs, end_ep)
    obs = next_obs

    if total_reward > best:
        # save model
        best = total_reward

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
