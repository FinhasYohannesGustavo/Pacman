# test_env_registration.py

import gymnasium as gym
import pacman_Reinforcement_Learning  # This will trigger the registration

def main():
    try:
        env = gym.make('PacMan-v0')
        print("Environment 'PacMan-v0' successfully created!")
        env.close()
    except gym.error.UnregisteredEnv:
        print("Environment 'PacMan-v0' not found. Please check the registration.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
