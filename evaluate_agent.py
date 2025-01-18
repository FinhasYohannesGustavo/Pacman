# evaluate_agent.py

import gymnasium as gym
from stable_baselines3 import PPO
import pygame
import pacman_Reinforcement_Learning  # triggers registration


def main():
    # Load the trained model
    model = PPO.load("ppo_pacman_final")
    
    # Create the environment with rendering enabled
    env = gym.make('PacMan-v0', render_enabled=True)
    
    # Reset the environment
    observation, info = env.reset()
    done = False
    truncated = False
    total_reward = 0
    step_count = 0
    max_steps = 1000  # Adjust as needed
    
    while not done and not truncated and step_count < max_steps:
        # Predict the action using the trained model
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, done, truncated, info = env.step(action)
        total_reward += reward
        env.render()  # Render the environment
        step_count += 1
        pygame.time.delay(100)  # Adjust delay for observation speed
    
    print(f"Evaluation Episode Reward: {total_reward}")
    print("You can close the window manually.")
    
    # Keep the window open until manually closed
    env.render_environment()

if __name__ == "__main__":
    main()
