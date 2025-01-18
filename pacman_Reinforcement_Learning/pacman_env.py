# pacman_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from collections import deque
from typing import Optional, Tuple
import os
import sys

# Initialize Pygame (required for rendering)
pygame.init()
pygame.mixer.init()

class Ghost:
    def __init__(self, id_, state, env, strategy='chase', scatter_target=(0,0)):
        """
        Initialize a Ghost.

        Parameters:
            id_ (int): Unique identifier for the ghost.
            state (Tuple[int, int]): Current position of the ghost on the grid.
            env (PacManEnv): Reference to the environment.
            strategy (str): Movement strategy ('chase', 'scatter', 'random').
            scatter_target (Tuple[int, int]): Target position for 'scatter' strategy.
        """
        self.id = id_
        self.state = state
        self.initial_state = state
        self.env = env  # Reference to the environment
        self.strategy = strategy  # e.g., 'chase', 'scatter', 'random'
        self.scatter_target = scatter_target
        self.invalid_move_attempts = {}
        self.move_frequency = 1.5  # <-- ghost moves only once every 3 times 'move()' is called
        self.step_counter = 0

    def move(self, pacman_pos):
        """
        Move the ghost based on its strategy.

        Parameters:
            pacman_pos (Tuple[int, int]): Current position of Pac-Man.
        """

         # Increase our counter each call
        self.step_counter += 1
        # If we haven't reached move_frequency yet, skip actual move
        if self.step_counter < self.move_frequency:
            return

        self.step_counter = 0
        if self.strategy == 'chase':
            self.chase_pacman(pacman_pos)
        elif self.strategy == 'scatter':
            self.scatter()
        elif self.strategy == 'random':
            self.random_move()


    def chase_pacman(self, pacman_pos):
        """
        Move the ghost towards Pac-Man using BFS to find the shortest path.

        Parameters:
            pacman_pos (Tuple[int, int]): Current position of Pac-Man.
        """
        path = self.bfs(self.state, pacman_pos)
        if path and len(path) > 1:
            new_pos = path[1]
            if self.env.is_valid_position(new_pos):
                self.env.grid[self.state] = 0  # Remove ghost from current position
                self.state = new_pos
                self.env.grid[self.state] = 4  # Place ghost in new position
            else:
                self.log_invalid_move(new_pos)
        else:
            self.random_move()

    def scatter(self):
        """
        Move the ghost towards its scatter target using BFS.
        """
        path = self.bfs(self.state, self.scatter_target)
        if path and len(path) > 1:
            new_pos = path[1]
            if self.env.is_valid_position(new_pos):
                self.env.grid[self.state] = 0
                self.state = new_pos
                self.env.grid[self.state] = 4
            else:
                self.log_invalid_move(new_pos)
        else:
            self.random_move()

    def bfs(self, start, goal):
        """
        Perform Breadth-First Search to find the shortest path from start to goal.

        Parameters:
            start (Tuple[int, int]): Starting position.
            goal (Tuple[int, int]): Goal position.

        Returns:
            List[Tuple[int, int]] or None: The path as a list of positions, or None if no path found.
        """
        queue = deque()
        queue.append((start, [start]))
        visited = set()
        visited.add(start)

        while queue:
            current_pos, path = queue.popleft()
            if current_pos == goal:
                return path

            for action in [0, 1, 2, 3]:  # Up, Down, Left, Right
                new_pos = self.get_new_position(current_pos, action)
                if self.env.is_valid_position(new_pos) and new_pos not in visited:
                    visited.add(new_pos)
                    queue.append((new_pos, path + [new_pos]))
        return None  # No path found

    def get_new_position(self, position, action):
        """
        Calculate the new position based on the action.

        Parameters:
            position (Tuple[int, int]): Current position.
            action (int): Action identifier (0=Up, 1=Down, 2=Left, 3=Right).

        Returns:
            Tuple[int, int]: New position after action.
        """
        row, col = position
        if action == 0:  # Up
            return (row - 1, col)
        elif action == 1:  # Down
            return (row + 1, col)
        elif action == 2:  # Left
            return (row, col - 1)
        elif action == 3:  # Right
            return (row, col + 1)
        else:
            return position  # No movement

    def random_move(self):
        """
        Move the ghost in a random valid direction.
        """
        actions = [0, 1, 2, 3]
        random.shuffle(actions)
        for action in actions:
            new_pos = self.get_new_position(self.state, action)
            if self.env.is_valid_position(new_pos):
                self.env.grid[self.state] = 0  # Remove ghost from current position
                self.state = new_pos
                self.env.grid[self.state] = 4  # Place ghost in new position
                # print(f"Ghost {self.id} moved randomly to {new_pos}")
                return
        # If no valid moves, do nothing
        print(f"Ghost {self.id} cannot move; no valid positions available.")

    def log_invalid_move(self, new_pos):
        """
        Log invalid move attempts, limiting the number of logs per position.

        Parameters:
            new_pos (Tuple[int, int]): The invalid position attempted.
        """
        if new_pos in self.invalid_move_attempts:
            self.invalid_move_attempts[new_pos] += 1
        else:
            self.invalid_move_attempts[new_pos] = 1

        if self.invalid_move_attempts[new_pos] <= 5:
            print(f"Ghost {self.id} cannot move to position: {new_pos}")

    def reset(self):
        """
        Reset the ghost to its initial position.
        """
        self.env.grid[self.state] = 0  # Remove ghost from current position
        self.state = self.initial_state
        self.env.grid[self.state] = 4  # Place ghost back at initial position
        self.invalid_move_attempts = {}


class PacManEnv(gym.Env):
    """
    Custom Environment for Pac-Man compatible with OpenAI Gymnasium.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, render_enabled=True):
        super(PacManEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # 0=Up, 1=Down, 2=Left, 3=Right

        # Define observation space (adjust as per your features)
        self.num_ghosts = 4
        self.state_size = 2 + 1 + 1 + (4 * self.num_ghosts) + 4 + 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_size,), dtype=np.float32)

        # Initialize game variables
        self.grid = self.initialize_grid()
        self.pacman_pos = self.get_pacman_start_pos()
        self.ghosts = self.initialize_ghosts()
        self.dots_left = self.count_pacdots()

        # Rendering variables
        self.screen = None
        self.clock = pygame.time.Clock()
        self.scale = 30  # Pixels per grid cell

        # Load Images
        self.load_images()

        # Reward structure
        self.PACDOT_REWARD = 20
        self.MOVE_PENALTY = -1
        self.WIN_REWARD = 100
        self.LOSE_PENALTY = -40

        # Additional attributes for animation
        self.pacman_mouth_open = True

        # Rendering control
        self.render_enabled = render_enabled

    
    def load_images(self):
        """
        Load and scale all necessary images for rendering.
        """
        # Define the path to the assets directory
        assets_path = os.path.join(os.path.dirname(__file__), 'assets')

        # Load Pac-Man images
        try:
            self.pac_man_image_open = pygame.image.load(os.path.join(assets_path, "pac_man_abierto.png"))
            self.pac_man_image_open = pygame.transform.scale(self.pac_man_image_open, (self.scale, self.scale))

            self.pac_man_image_closed = pygame.image.load(os.path.join(assets_path, "pac_man_cerrado.png"))
            self.pac_man_image_closed = pygame.transform.scale(self.pac_man_image_closed, (self.scale, self.scale))
        except pygame.error as e:
            print(f"Error loading Pac-Man images: {e}")
            raise

        # Load Ghost images
        self.ghost_images = []
        ghost_filenames = ["biene.png", "biene2.png", "biene3.png", "biene4.png"]
        for filename in ghost_filenames:
            try:
                ghost_image = pygame.image.load(os.path.join(assets_path, filename))
                ghost_image = pygame.transform.scale(ghost_image, (self.scale, self.scale))
                self.ghost_images.append(ghost_image)
            except pygame.error as e:
                print(f"Error loading Ghost image '{filename}': {e}")
                raise

    def initialize_grid(self):
        """
        Initialize the game grid.
        0: Empty, 1: Pac-dot, 2: Wall, 3: Pac-Man, 4: Ghost
        """
        grid = np.array([
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 2, 2, 0, 2, 2, 0, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1],
            [0, 2, 4, 0, 4, 2, 0, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1],
            [0, 2, 0, 0, 0, 2, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 1],
            [0, 2, 4, 0, 4, 2, 0, 2, 2, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1],
            [0, 2, 2, 2, 2, 2, 0, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1],
            [0, 0, 0, 2, 0, 0, 0, 2, 1, 0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
            [1, 2, 1, 1, 1, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1],
            [1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1]
        ])
        return grid

    def state_to_pos(self, state: Tuple[int, int]) -> Tuple[int, int]:
        """
        Convert grid-based state to pixel coordinates.

        Parameters:
            state (Tuple[int, int]): The (row, col) position on the grid.

        Returns:
            Tuple[int, int]: The (x, y) pixel coordinates.
        """
        row, col = state
        x = col * self.scale
        y = row * self.scale
        return x, y

    def get_pacman_start_pos(self):
        """
        Define Pac-Man's starting position.
        """
        positions = np.where(self.grid == 3)
        if len(positions[0]) == 0:
            raise ValueError("Pac-Man starting position (value '3') not found in the grid.")
        return (positions[0][0], positions[1][0])

    def initialize_ghosts(self):
        """
        Initialize ghosts' positions with assigned strategies and scatter targets.
        """
        positions = np.where(self.grid == 4)
        ghost_positions = list(zip(positions[0], positions[1]))
        if len(ghost_positions) != self.num_ghosts:
            raise ValueError(f"Expected {self.num_ghosts} ghosts, but found {len(ghost_positions)} in the grid.")
        
        # Define strategies and scatter targets for each ghost
        strategies = ['chase', 'scatter', 'random', 'chase']  # Example strategies
        scatter_targets = [(0,0), (0,24), (9,0), (9,24)]  # Example scatter targets
        
        ghosts = []
        for i, pos in enumerate(ghost_positions):
            strategy = strategies[i % len(strategies)]
            scatter_target = scatter_targets[i % len(scatter_targets)]
            ghosts.append(Ghost(i, pos, self, strategy, scatter_target))
        return ghosts

    def count_pacdots(self):
        """
        Count the number of pac-dots on the grid.
        """
        return np.sum(self.grid == 1)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the state of the environment to an initial state.
        
        Parameters:
            seed (int, optional): Seed for the environment's random number generator.
            options (dict, optional): Additional options for resetting the environment.
        
        Returns:
            observation (np.array): The initial observation of the environment.
            info (dict): Additional information about the reset.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            # If you use other random libraries, set their seeds here as well

        # Reset the game state
        self.grid = self.initialize_grid().copy()
        self.pacman_pos = self.get_pacman_start_pos()
        self.ghosts = self.initialize_ghosts()
        self.dots_left = self.count_pacdots()

        # Place Pac-Man and ghosts on the grid
        self.grid[self.pacman_pos] = 3
        for ghost in self.ghosts:
            self.grid[ghost.state] = 4

        # Reset additional attributes
        self.pacman_mouth_open = True

        # Return the initial observation and info
        observation = self.get_observation()
        info = {}
        return observation, info

    def step(self, action):
        """
        Execute one time step within the environment.
        
        Parameters:
            action (int): Action taken by the agent.
        
        Returns:
            observation (np.array): Next state.
            reward (float): Reward obtained.
            done (bool): Whether the episode has ended.
            truncated (bool): Whether the episode was truncated (e.g., due to step limit).
            info (dict): Additional information.
        """
        reward = self.MOVE_PENALTY
        done = False
        truncated = False
        info = {}

        try:
            # Move Pac-Man
            new_pacman_pos = self.get_new_position(self.pacman_pos, action)
            if self.is_valid_position(new_pacman_pos):
                self.grid[self.pacman_pos] = 0  # Remove Pac-Man from current position
                self.pacman_pos = new_pacman_pos

                # Check what's in the new position
                if self.grid[self.pacman_pos] == 1:
                    reward += self.PACDOT_REWARD
                    self.dots_left -= 1
                    self.grid[self.pacman_pos] = 3
                elif self.grid[self.pacman_pos] == 4:
                    reward += self.LOSE_PENALTY
                    done = True
                else:
                    self.grid[self.pacman_pos] = 3
            else:
                # Invalid move (wall or out of bounds)
                reward += self.MOVE_PENALTY  # Penalize invalid moves

            # Move Ghosts
            for ghost in self.ghosts:
                ghost.move(self.pacman_pos)
                if ghost.state == self.pacman_pos:
                    reward += self.LOSE_PENALTY
                    done = True

            # Check for win condition
            if self.dots_left == 0:
                reward += self.WIN_REWARD
                done = True

            # Distance-based reward adjustments
            pac_r, pac_c = self.pacman_pos
            ghost_distances = [abs(pac_r - ghost.state[0]) + abs(pac_c - ghost.state[1]) for ghost in self.ghosts]
            min_ghost_distance = min(ghost_distances) if ghost_distances else 10

            # Encourage moving towards pac-dots
            pac_dots_positions = list(zip(*np.where(self.grid == 1)))
            if pac_dots_positions:
                distances_to_dots = [abs(pac_r - dot_r) + abs(pac_c - dot_c) for dot_r, dot_c in pac_dots_positions]
                min_dot_distance = min(distances_to_dots)
                reward += (10 - min_dot_distance) * 0.1  # Smaller scaling factor

            # Discourage moving towards ghosts
            if min_ghost_distance < (self.grid.shape[0] + self.grid.shape[1]):
                reward -= (10 - min_ghost_distance) * 0.1  # Smaller scaling factor

            done = done or False  # Ensure it's a boolean

        except IndexError as e:
            print(f"IndexError encountered during step: {e}")
            done = True
            reward += self.LOSE_PENALTY

        # Toggle Pac-Man's mouth state for animation
        self.pacman_mouth_open = not self.pacman_mouth_open

        # Construct observation
        observation = self.get_observation()

        # Return the step information
        return observation, reward, done, truncated, info

    def render(self, mode='human'):
        """
        Render the environment to the screen.
        """
        if not self.render_enabled:
            return

        if self.screen is None:
            self.screen = pygame.display.set_mode((self.grid.shape[1] * self.scale, self.grid.shape[0] * self.scale))
            pygame.display.set_caption("Pac-Man AI")

        # Handle Pygame events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill((0, 0, 0))  # Black background

        # Draw walls and pac-dots
        for row in range(self.grid.shape[0]):
            for col in range(self.grid.shape[1]):
                cell = self.grid[row, col]
                x = col * self.scale
                y = row * self.scale
                if cell == 1:
                    # Pac-dot
                    pygame.draw.circle(self.screen, (255, 255, 0), (x + self.scale//2, y + self.scale//2), 3)
                elif cell == 2:
                    # Wall
                    pygame.draw.rect(self.screen, (0, 0, 255), (x, y, self.scale, self.scale))  # Blue walls
                elif cell == 3:
                    # Pac-Man
                    if self.pacman_mouth_open:
                        self.screen.blit(self.pac_man_image_open, (x, y))
                    else:
                        self.screen.blit(self.pac_man_image_closed, (x, y))
                # Ghosts are drawn separately below

        # Draw Ghosts
        for ghost in self.ghosts:
            ghost_x, ghost_y = self.state_to_pos(ghost.state)
            # Select the appropriate ghost image based on ghost ID
            if ghost.id < len(self.ghost_images):
                ghost_image = self.ghost_images[ghost.id]
            else:
                ghost_image = self.ghost_images[0]  # Fallback to first image if ID exceeds list
            self.screen.blit(ghost_image, (ghost_x, ghost_y))

        # Optionally, draw score or other info here

        pygame.display.flip()
        self.clock.tick(60)  # Control the rendering speed

    def close(self):
        """
        Clean up resources.
        """
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

    def get_new_position(self, position, action):
        """
        Calculate the new position based on the action.
        """
        row, col = position
        if action == 0:  # Up
            return (row - 1, col)
        elif action == 1:  # Down
            return (row + 1, col)
        elif action == 2:  # Left
            return (row, col - 1)
        elif action == 3:  # Right
            return (row, col + 1)
        else:
            return position  # No movement

    def is_valid_position(self, position):
        """
        Check if the new position is within bounds, not a wall, and not occupied by another ghost.

        Parameters:
            position (Tuple[int, int]): The (row, col) position to check.

        Returns:
            bool: True if valid, False otherwise.
        """
        row, col = position
        rows, cols = self.grid.shape
        if not (0 <= row < rows and 0 <= col < cols):
            return False  # Out of bounds
        if self.grid[position] == 2:
            return False  # Wall
        if self.grid[position] == 4:
            return False  # Another ghost
        return True

    def get_observation(self):
        # Normalize features
        pac_r, pac_c = self.pacman_pos
        pac_r_norm = pac_r / self.grid.shape[0]
        pac_c_norm = pac_c / self.grid.shape[1]

        # Nearest ghost distance (normalized)
        ghost_distances = [abs(pac_r - ghost.state[0]) + abs(pac_c - ghost.state[1]) for ghost in self.ghosts]
        min_ghost_distance = min(ghost_distances) if ghost_distances else 10
        ghost_dist_norm = min_ghost_distance / (self.grid.shape[0] + self.grid.shape[1])

        # Number of pac-dots left (normalized)
        initial_dots = np.sum(self.grid == 1) + (self.dots_left if hasattr(self, 'dots_left') else 0)
        dots_left_norm = self.dots_left / initial_dots if initial_dots > 0 else 0.0

        # One-hot encoding for ghost directions (Up, Down, Left, Right for each ghost)
        ghost_directions = []
        for ghost in self.ghosts:
            delta_row = self.pacman_pos[0] - ghost.state[0]
            delta_col = self.pacman_pos[1] - ghost.state[1]
            if abs(delta_row) > abs(delta_col):
                direction = 1 if delta_row > 0 else 0  # Down or Up
            elif abs(delta_col) > 0:
                direction = 3 if delta_col > 0 else 2  # Right or Left
            else:
                direction = -1  # Same position
            ghost_directions.append(direction)

        ghost_dir_one_hot = np.zeros(4 * self.num_ghosts, dtype=np.float32)
        for idx, dir_ in enumerate(ghost_directions):
            if dir_ != -1:
                ghost_dir_one_hot[idx * 4 + dir_] = 1.0  # [Up, Down, Left, Right]

        # Proximity to walls (Up, Down, Left, Right)
        walls = []
        for action in range(4):
            new_pos = self.get_new_position(self.pacman_pos, action)
            walls.append(0.0 if self.is_valid_position(new_pos) else 1.0)
        walls = np.array(walls, dtype=np.float32)

        # Local dot density (within a radius of 2)
        dot_density = self.get_local_dot_density(radius=2)
        dot_density = np.array(dot_density, dtype=np.float32)

        # Convert scalar features to float32
        pac_r_norm = np.float32(pac_r_norm)
        pac_c_norm = np.float32(pac_c_norm)
        ghost_dist_norm = np.float32(ghost_dist_norm)
        dots_left_norm = np.float32(dots_left_norm)

        # Concatenate all features as float32
        state_vector = np.concatenate((
            [pac_r_norm, pac_c_norm],
            [ghost_dist_norm],
            [dots_left_norm],
            ghost_dir_one_hot,
            walls,
            [dot_density]
        ), axis=0)

        return state_vector

    def get_local_dot_density(self, radius=2):
        """
        Calculate the normalized number of pac-dots within a given radius.

        Parameters:
            radius (int): The radius to consider around Pac-Man.

        Returns:
            float: Normalized dot density.
        """
        pac_r, pac_c = self.pacman_pos
        count = 0
        for r in range(max(0, pac_r - radius), min(self.grid.shape[0], pac_r + radius + 1)):
            for c in range(max(0, pac_c - radius), min(self.grid.shape[1], pac_c + radius + 1)):
                if self.grid[r, c] == 1:
                    count += 1
        max_possible = (2 * radius + 1) ** 2
        return count / max_possible if max_possible > 0 else 0.0

    def render_environment(self):
        """
        Render the environment and keep the window open until manually closed.
        """
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.render()
        self.close()