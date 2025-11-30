import os
import gymnasium as gym
import torch

from utils import *

from agent.env import build_env

class EvaluateDQN():

    def __init__(self, model):
        self.model = model

        self.cfg = load_config()
        self.log = get_logger("eval")
        
        self.csv_log = CSVLogger(f"{self.cfg['env']['mod_name']}/logs/eval.csv", [
            "Id",
            "Q",
            "Avg R",
            "Best R"
        ], 1)

        self.max_eval = 0
        self.eval_count = 0
        self.episode_eval = self.cfg["eval"]["ep_per_eval"]

        self.env = build_env(
            use_seed = False,
            render_mode = "rgb_array",
            train = False
        )

    def evaluate(self, eval_states):
        self.eval_count += 1

        env_video = gym.wrappers.RecordVideo(
            self.env,
            episode_trigger=lambda num: num % (self.episode_eval - 1) == 0,
            video_folder=self.cfg["env"]["mod_name"] + "/videos",
            name_prefix="eval-" + str(self.eval_count)
        )

        self.model.policy_net.eval()

        avgReward = []
        avgQ = []

        for i in range(self.episode_eval):
            if i == self.episode_eval - 1:
                envv = env_video
            else:
                envv = self.env

            obs, _ = envv.reset()
            obs = torch.tensor(obs)
            total_reward = 0
            stuck = 0

            while True:
                action = self.model.predict(obs)

                next_obs, reward, terminated, truncated, info = envv.step(action.item())
        
                total_reward += reward
                done = terminated or truncated

                if reward != 0:
                    stuck = 0

                next_obs = torch.tensor(next_obs)
                obs = next_obs
                stuck += 1

                if stuck > 250:
                    self.log.warning("[red]Probably looping during eval[/]")
                    break

                if terminated or truncated:
                    break

            avgReward.append(total_reward)

        q = self.model.evaluate_q(eval_states)
        avgR = sum(avgReward) / self.episode_eval

        if avgR > self.max_eval:
            self.max_eval = avgR

            path = self.cfg["env"]["mod_name"] + "/checkpoints"
            os.makedirs(path, exist_ok=True)
            torch.save(self.model.policy_net.state_dict(), f"{path}/best.pth")

        self.log.info(f"[yellow]== EVAL[/] {self.eval_count} Q: {q}, reward: {avgR}, best: {self.max_eval}")
        self.csv_log.add_row([self.eval_count, q, avgR, self.max_eval])

        self.model.policy_net.train()
        env_video.close()
