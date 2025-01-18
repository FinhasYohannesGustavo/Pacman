# pacman_Reinforcement_Learning/__init__.py

import gymnasium as gym
from pacman_Reinforcement_Learning.pacman_env import PacManEnv

# Register the environment with Gymnasium
gym.envs.registration.register(
    id='PacMan-v0',
    entry_point='pacman_Reinforcement_Learning.pacman_env:PacManEnv',
)

# Optional: Define what gets imported when using 'from pacman_Reinforcement_Learning import *'
__all__ = ['PacManEnv', 'Ghost']
