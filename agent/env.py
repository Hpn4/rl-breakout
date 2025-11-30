import gymnasium as gym
import ale_py
import torch
import random

from collections import deque

from utils import *

class AutoFireWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        self.lives = info["lives"]
        obs, _, _, _, info = self.env.step(1) # Auto fire
        info["lose_life"] = False

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        lives = info["lives"]
        if self.lives != lives:
            self.lives = lives

            obs, reward, terminated, truncated, info = self.env.step(1) # Auto fire

            info["lose_life"] = True
        else:
            info["lose_life"] = False

        return obs, reward, terminated, truncated, info

class LoggingWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

        self.cfg = load_config()
        self.log = get_logger("train")
        self.logging = False

        self.model = None
        self.eval_states = None

        header = [
            "Id",
            "Steps",
            "Reward",
            "Ep steps",
            "Ep reward",
            "Lives",
            "Avg Q",
            "Avg R",
            "Stucked"
        ]
        self.csv_log = CSVLogger(
            f"{self.cfg['env']['mod_name']}/logs/train.csv",
            header
        )

        self.avgReward = deque(maxlen=self.cfg["eval"]["avg_window"])
        self.avgQ = deque(maxlen=self.cfg["eval"]["avg_window"])

        self.data = {
            "steps": 0,
            "episode": 0,
            "lives": 5,
            "life_steps": 0,
            "life_reward": 0,
            "ep_steps": 0,
            "ep_reward": 0,
            "stuck": 0,
            "stucked": False
        }

    def set_model(self, model, eval_states):
        self.model = model
        self.eval_states = eval_states

    def set_logging(self, logging):
        self.logging = logging

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if not self.logging:
            return obs, info

        self.data["lives"] = info["lives"]
        self.data["ep_steps"] = 0
        self.data["stuck"] = 0
        self.data["ep_reward"] = 0
        self.data["stucked"] = False

        return obs, info

    def _end_ep(self):
        if self.data["lives"] == 0:
            self.avgReward.append(self.data["ep_reward"])
        
        self.avgQ.append(self.model.evaluate_q(self.eval_states))

        aQ = sum(self.avgQ)
        aR = sum(self.avgReward)
        
        if len(self.avgQ) > 0:
            aQ /= len(self.avgQ)

        if len(self.avgReward) > 0:
            aR /= len(self.avgReward)

        if self.data["lives"] == 0:
            self.log.info(f"[green]==== EPISODE {self.data['episode']} ====[/]")
            self.log.info(f"Total R: {self.data['ep_reward']}, Steps: {self.data['ep_steps']}")
            self.log.info(f"Avg Q: {aQ}, R: {aR}")
            self.log.info(f"Eps: {self.model.eps}, Tot steps: {self.data['steps']}, Got stucked: {self.data['stucked']}")

        self.csv_log.add_row([
            self.data["episode"],
            self.data["life_steps"],
            self.data["life_reward"],
            self.data["ep_steps"],
            self.data["ep_reward"],
            self.data["lives"],
            aQ,
            aR,
            self.data["stucked"]
        ])
        
        self.data["episode"] += 1
        self.data["life_steps"] = 0
        self.data["life_reward"] = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if not self.logging:
            return obs, reward, terminated, truncated, info

        if reward != 0:
            self.data["stuck"] = 0
        else:
            self.data["stuck"] += 1

        if self.data["stuck"] > 200:
            self.data["stucked"] = True

        self.data["ep_reward"] += reward
        self.data["life_reward"] += reward
        self.data["steps"] += 1
        self.data["life_steps"] += 1
        self.data["ep_steps"] += 1

        if info["lose_life"]:
            self.data["lives"] = info["lives"]
            self._end_ep()

        return obs, reward, terminated, truncated, info

def build_env(
    use_seed=True,
    render_mode=None,
    train=True):
    gym.register_envs(ale_py)

    cfg = load_config()

    env = gym.make(
        cfg["env"]["game"],
        render_mode = render_mode,
        frameskip = 1,
        repeat_action_probability = cfg["env"]["repeat_action_probability"]
    )

    if use_seed:
        seed = cfg["env"]["seed"]
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # Preprocessing observations
    env = gym.wrappers.AtariPreprocessing(
        env,
        terminal_on_life_loss = cfg["env"]["terminal_on_life_loss"],
        grayscale_obs = cfg["env"]["grayscale_obs"],
        screen_size = cfg["env"]["screen_size"],
        frame_skip = cfg["env"]["frame_skip"],
        scale_obs = cfg["env"]["scale_obs"],
        noop_max = cfg["env"]["noop_max"] # random number of no ops between 0-30 to avoid same start so reduce overfit
    )

    # Stack observations
    env = gym.wrappers.FrameStackObservation(
        env,
        stack_size = 4,
        padding_type = "reset"
    )

    env = AutoFireWrapper(env)

    if train:
        env_log = LoggingWrapper(env)

        return env_log, env_log

    return env
