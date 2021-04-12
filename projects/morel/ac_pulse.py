import gym
import numpy as np
from attrdict import AttrDict


class ACPulse(gym.Env):
    def __init__(self, env_config):
        self.action_space = gym.spaces.Discrete(2)
        self.action_space.shape = (1,)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(env_config["obs_shape"],), dtype=np.float32
        )
        self.spec = AttrDict(
            {
                "id": "ac-pulse-v0",
                "max_episode_steps": 60,
                "observation_dim": (env_config["obs_shape"],),
                "action_dim": (1,),
            }
        )
