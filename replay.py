import os
import gymnasium as gym
import torch
import random

from utils import *
from model import *

from agent.env import build_env
from agent.agent import build_agent

class ReplayDQN():

    def __init__(self, model):
        self.model = model

        self.cfg = load_config()


    def run(self, best, start_eps=0.05, glitch=False):
        self.env = build_env(
            use_seed=False,
            render_mode="rgb_array", #"human"
            train = False
        )

        self.env = gym.wrappers.RecordVideo(
            self.env,
            episode_trigger=lambda num: True,
            video_folder="best-videos",
            name_prefix="eval-"
        )

        self.model.policy_net.eval()

        obs, _ = self.env.reset()
        obs = torch.tensor(obs)
        steps = 0
        total_reward = 0
        counter = 0
        bricks = 18 * 6
        eps = start_eps

        while True:
            if (not glitch or total_reward >= 775 or counter >= 100) and random.random() <= eps:
                action = random.randint(0, 3)
            else:
                action = self.model.predict(obs).item()
            
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated

            if reward != 0:
                counter = 0
                eps = start_eps
                bricks -= 1

            if bricks == 0:
                print("= Second wall =")
                eps = start_eps
                bricks = 18 * 6

            next_obs = torch.tensor(next_obs)
            obs = next_obs
            steps += 1
            counter += 1

            if glitch and counter % 200 == 0:
                eps += 0.01
            if counter > 5000:
                print("possibly looping")
                break
    

            if terminated or truncated:
                break

        print("R:", total_reward, "steps:", steps, "best:", best, "eps:", eps)
        self.env.close()
        return total_reward

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device, use_cuda)

# Load model
model = build_agent(None, 4, device)
#model.policy_net.load_state_dict(torch.load("model/D3QN4/checkpoints/best-train-806.0.pth"))
model.policy_net.load_state_dict(torch.load("model/D3QN4/checkpoints/best.pth", map_location=device))
model.policy_net.eval()

replay = ReplayDQN(model)

import os

N = 20
avg = []
best = 0

for _ in range(N):
    rr = replay.run(best, 0.0, True)

    if rr > best or (rr > 432 and rr < 500):
        best = rr
        print(f"======== BEST: {best} =========")
        os.rename("best-videos/eval--episode-0.mp4", "best.mp4")
    else:
        os.remove("best-videos/eval--episode-0.mp4")
    avg.append(rr)

print("Average reward:", sum(avg) / len(avg))
print("== BEST:", best)
#print(sum(avg) / N)