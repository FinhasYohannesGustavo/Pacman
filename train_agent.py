# train_agent.py

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os

# Import the package to trigger environment registration
import pacman_Reinforcement_Learning

def main():
    # Create the environment
    env = gym.make('PacMan-v0', render_enabled=False)  # Disable rendering for faster training
    
    # Optional: Check if the environment follows Gymnasium's API
    check_env(env, warn=True)
    
    # Define the model
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        tensorboard_log="./ppo_pacman_tensorboard/",
        learning_rate=0.001,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        clip_range_vf=0.2,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )
    
    # Define callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./checkpoints/',
        name_prefix='ppo_pacman_model'
    )
    
    eval_callback = EvalCallback(
        env,
        best_model_save_path='./best_model/',
        log_path='./logs/',
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    
    # Create directories if they don't exist
    os.makedirs('./checkpoints/', exist_ok=True)
    os.makedirs('./best_model/', exist_ok=True)
    os.makedirs('./logs/', exist_ok=True)
    
    # Train the agent
    model.learn(
        total_timesteps=1_000_000,
        callback=[checkpoint_callback, eval_callback]
    )
    
    # Save the final model
    model.save("ppo_pacman_final")
    print("Training completed and model saved.")

if __name__ == "__main__":
    main()
