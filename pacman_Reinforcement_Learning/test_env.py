# test_env.py

from pacman_env import PacManEnv
import pygame
import time

def main():
    # Initialize the environment with rendering enabled
    env = PacManEnv(render_enabled=True)
    observation, info = env.reset()
    done = False
    truncated = False
    total_reward = 0
    step_count = 0
    max_steps = 1000  # Prevent infinite loops during testing

    # Run the episode
    while not done and not truncated and step_count < max_steps:
        action = env.action_space.sample()  # Random action
        observation, reward, done, truncated, info = env.step(action)
        total_reward += reward
        env.render()  # Update the rendering
        step_count += 1
        # Optional: Add a short delay to observe movements
        pygame.time.delay(100)  # Delay in milliseconds

    print(f"Test Episode Reward: {total_reward}")

    # Keep the window open until manually closed
    print("Episode ended. You can close the window manually.")
    env.render_environment()

if __name__ == "__main__":
    main()
