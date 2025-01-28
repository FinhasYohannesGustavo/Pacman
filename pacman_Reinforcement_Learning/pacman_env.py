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
    def __init__(self, id_, state, env, initial_strategy='chase', respawn_steps=10):
        """
        Initialize a Ghost.

        Parameters:
            id_ (int): Unique identifier for the ghost.
            state (Tuple[int, int]): Current position of the ghost on the grid.
            env (PacManEnv): Reference to the environment.
            initial_strategy (str): Initial movement strategy ('chase', 'random').
            respawn_steps (int): Number of steps before the ghost respawns after being eaten.
        """
        self.id = id_
        self.state = state
        self.initial_state = state
        self.env = env  # Reference to the environment
        self.current_strategy = initial_strategy  # Current movement strategy
        self.invalid_move_attempts = {}
        self.move_frequency = 1  # Ghost moves every step
        self.step_counter = 0
        self.alive = True  # Indicates if the ghost is active on the grid
        self.position_history = deque(maxlen=4)  # Track last 4 positions
        self.mode_timer = 100  # Steps before switching strategy
        self.respawn_steps = respawn_steps  # Steps before respawn
        self.respawn_timer = 0  # Counter for respawning

    def move(self, pacman_pos):
        if not self.alive:
            if self.respawn_timer > 0:
                self.respawn_timer -= 1
                # Ghost is in the process of respawning
                if self.respawn_timer == 0:
                    self.reset()
            return  # Ghost does not move if not alive

        self.step_counter += 1
        if self.step_counter < self.move_frequency:
            return
        self.step_counter = 0

        previous_position = self.state

        if self.current_strategy == 'chase':
            self.chase_pacman(pacman_pos)
        elif self.current_strategy == 'random':
            self.random_move()

        # Add position to history
        self.position_history.append(self.state)

        # Check for looping: if the last 4 positions have 2 or fewer unique positions
        if len(self.position_history) == self.position_history.maxlen:
            unique_positions = set(self.position_history)
            if len(unique_positions) <= 2:
                # Switch strategy to prevent looping
                self.switch_strategy()

        # Update mode timer
        self.mode_timer -= 1
        if self.mode_timer <= 0:
            self.switch_strategy()

    def switch_strategy(self):
        """
        Switch the ghost's strategy to prevent looping or to add dynamics.
        Only allow switching to 'random'. Prevent switching back to 'chase'.
        If the ghost is the current chasing ghost and switches to 'random',
        notify the environment to assign a new chasing ghost.
        """
        if self.current_strategy == 'chase':
            self.current_strategy = 'random'
            # Notify the environment to assign a new chasing ghost
            self.env.assign_new_chasing_ghost()
            # print(f"Ghost {self.id} switched strategy to 'random'.")
        elif self.current_strategy == 'random':
            # Prevent switching back to 'chase'
            self.current_strategy = 'random'
            # print(f"Ghost {self.id} remains on 'random' strategy.")
        self.mode_timer = 100  # Reset mode timer

    def chase_pacman(self, pacman_pos):
        """
        Move the ghost towards Pac-Man using BFS to find the shortest path.

        Parameters:
            pacman_pos (Tuple[int, int]): Current position of Pac-Man.
        """
        path = self.bfs(self.state, pacman_pos)
        if path and len(path) > 1:
            new_pos = path[1]
            if self.env.is_valid_position_for_ghost(new_pos):
                self.state = new_pos
            else:
                self.log_invalid_move(new_pos)
        else:
            # If no path found, perform a random move
            # print(f"Ghost {self.id} performing random move during chase.")
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
                if self.env.is_valid_position_for_ghost(new_pos) and new_pos not in visited:
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
        Move the ghost in a random valid direction, avoiding immediate backtracking.
        """
        actions = [0, 1, 2, 3]  # Up, Down, Left, Right
        opposite_action = {
            0: 1,  # Up -> Down
            1: 0,  # Down -> Up
            2: 3,  # Left -> Right
            3: 2   # Right -> Left
        }
        # Remove the opposite of the last move to prevent immediate backtracking
        if len(self.position_history) >= 2:
            last_move = self.get_action_from_positions(self.position_history[-2], self.position_history[-1])
            if last_move is not None and last_move in actions:
                opposite = opposite_action.get(last_move, None)
                if opposite is not None and opposite in actions:
                    actions.remove(opposite)

        random.shuffle(actions)
        for action in actions:
            new_pos = self.get_new_position(self.state, action)
            if self.env.is_valid_position_for_ghost(new_pos):
                self.state = new_pos
                return
        # If no valid moves, do nothing
        # print(f"Ghost {self.id} cannot move; no valid positions available.")

    def get_action_from_positions(self, from_pos, to_pos):
        """
        Determine the action taken based on movement from from_pos to to_pos.

        Parameters:
            from_pos (Tuple[int, int]): Previous position.
            to_pos (Tuple[int, int]): Current position.

        Returns:
            int or None: The action taken (0=Up, 1=Down, 2=Left, 3=Right) or None if no movement.
        """
        delta_row = to_pos[0] - from_pos[0]
        delta_col = to_pos[1] - from_pos[1]
        if delta_row == -1 and delta_col == 0:
            return 0  # Up
        elif delta_row == 1 and delta_col == 0:
            return 1  # Down
        elif delta_row == 0 and delta_col == -1:
            return 2  # Left
        elif delta_row == 0 and delta_col == 1:
            return 3  # Right
        else:
            return None  # No movement or invalid move

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
            pass  # Commented out to reduce verbosity during training

    def reset(self):
        """
        Reset the ghost to its initial position.
        """
        self.state = self.initial_state
        self.alive = True  # Ghost is active again
        self.respawn_timer = 0  # Reset respawn timer
        self.invalid_move_attempts = {}
        self.position_history.clear()
        # print(f"Ghost {self.id} has respawned at {self.state}")


class PacManEnv(gym.Env):
    """
    Custom Environment for Pac-Man compatible with OpenAI Gymnasium.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, render_enabled=True):
        super(PacManEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # 0=Up, 1=Down, 2=Left, 3=Right

        # Updated state_size and observation_space
        self.num_ghosts = 3  # Ensured to have three ghosts
        # state_size = 2 (Pac-Man position) 
        #            + 3 (ghost distances) 
        #            + 1 (dots left) 
        #            + 12 (ghost directions: 4 directions * 3 ghosts) 
        #            + 4 (walls proximity) 
        #            + 1 (dot density) 
        #            + 1 (power mode indicator) 
        #            = 24
        self.state_size = 2 + self.num_ghosts + 1 + (4 * self.num_ghosts) + 4 + 1 + 1  # 24
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_size,), dtype=np.float32)

        # Initialize game variables
        self.grid = self.initialize_grid()
        self.pacman_pos = self.get_pacman_start_pos()
        self.ghosts = self.initialize_ghosts()
        self.dots_left = self.count_pacdots()
        self.buff_dots = set()  # Set to track buff dots positions
        self.buff_duration = 40  # Timesteps for which buff is active (5 seconds at 60 steps/sec)
        self.buff_timer = 0
        self.power_mode = False  # Indicates if Pac-Man can eat ghosts

        # Rendering variables
        self.screen = None
        self.clock = pygame.time.Clock()
        self.scale = 30  # Pixels per grid cell

        # Load Images
        self.load_images()

        # Reward structure
        self.PACDOT_REWARD = 35  # Increased reward for eating a pac-dot
        self.BUFF_PACDOT_REWARD = 50
        self.MOVE_PENALTY = -5    # Increased penalty for each move
        self.WIN_REWARD = 3000
        self.LOSE_PENALTY = -350
        self.EAT_GHOST_REWARD = 10
        self.MILSTONE_REWARD = 1 #Small living reward

        # Additional attributes for animation
        self.pacman_mouth_open = True

        # Rendering control
        self.render_enabled = render_enabled

        # Step limit to prevent infinite episodes
        self.max_steps = 1000 
        self.current_step = 0

        # Current chasing ghost ID (added for optional rotation)
        self.current_chasing_ghost_id = 0  # Initially, the first ghost is chasing

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
        ghost_filenames = ["biene.png", "biene2.png", "biene3.png"]  # Exactly three ghosts
        for filename in ghost_filenames:
            try:
                ghost_image = pygame.image.load(os.path.join(assets_path, filename))
                ghost_image = pygame.transform.scale(ghost_image, (self.scale, self.scale))
                self.ghost_images.append(ghost_image)
            except pygame.error as e:
                print(f"Error loading Ghost image '{filename}': {e}")
                raise

        # Load Buff Pac-dot image (e.g., a different colored dot)
        try:
            self.buff_pac_dot_image = pygame.image.load(os.path.join(assets_path, "buff_pac_dot.png"))
            self.buff_pac_dot_image = pygame.transform.scale(self.buff_pac_dot_image, (self.scale // 2, self.scale // 2))
        except pygame.error as e:
            print(f"Error loading Buff Pac-dot image: {e}")
            # If buff pac-dot image not available, use a colored circle instead
            self.buff_pac_dot_image = None

    def initialize_grid(self):
        """
        Initialize the game grid.
        0: Empty, 1: Pac-dot, 2: Wall, 3: Pac-Man, 4: Ghost, 5: Buff Pac-dot
        """
        grid = np.array([
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 2, 2, 0, 2, 2, 0, 2, 1, 2, 2, 0, 2, 1, 2, 2, 0, 2, 2, 1, 2, 2, 2, 2, 1],
            [0, 2, 4, 0, 4, 2, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
            [0, 2, 0, 0, 0, 2, 0, 1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 1, 2, 1],
            [0, 2, 4, 0, 0, 2, 0, 2, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 1],
            [0, 2, 2, 2, 2, 2, 0, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1],
            [0, 0, 0, 2, 0, 0, 0, 2, 1, 0, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1],
            [0, 2, 0, 2, 0, 2, 1, 2, 1, 1, 1, 2, 1, 0, 0, 2, 1, 0, 0, 2, 1, 2, 1, 2, 1],
            [0, 2, 0, 0, 0, 2, 1, 2, 2, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1],
            [0, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1]
        ])
            
        return grid

    def get_pacman_start_pos(self):
        """
        Define Pac-Man's starting position.
        """
        # Search for Pac-Man's starting position in the current grid
        positions = np.where(self.grid == 3)
        if len(positions[0]) == 0:
            raise ValueError("Pac-Man starting position (value '3') not found in the grid.")
        return (positions[0][0], positions[1][0])

    def initialize_ghosts(self):
        """
        Initialize ghosts' positions with assigned strategies.
        Only 'chase' and 'random' strategies are used.
        """
        # Locate the positions where Ghosts were initially placed (value '4')
        positions = np.where(self.grid == 4)
        ghost_positions = list(zip(positions[0], positions[1]))
        if len(ghost_positions) != self.num_ghosts:
            raise ValueError(f"Expected {self.num_ghosts} ghosts, but found {len(ghost_positions)} in the grid.")

        # Define strategies: Only the first ghost is 'chase', others are 'random'
        strategies = ['chase'] + ['random'] * (self.num_ghosts - 1)

        ghosts = []
        for i, pos in enumerate(ghost_positions):
            strategy = strategies[i % len(strategies)]
            ghosts.append(Ghost(i, pos, self, strategy))
        
        # Set the current chasing ghost ID
        self.current_chasing_ghost_id = 0  # Assuming the first ghost is assigned 'chase'
        
        return ghosts

    def count_pacdots(self):
        """
        Count the number of pac-dots on the grid.
        """
        return np.sum(self.grid == 1)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the state of the environment to an initial state.
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
        self.buff_dots = set()  # Clear any existing buff dots
        self.buff_timer = 0
        self.power_mode = False

        # Reset additional attributes
        self.pacman_mouth_open = True
        self.current_step = 0  # Reset step counter

        # **Remove Pac-Man and Ghosts from the grid after extracting their positions**
        self.grid[self.pacman_pos] = 0
        for ghost in self.ghosts:
            self.grid[ghost.state] = 0

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

        # Possibly spawn a new Buff Pac-dot
        self.spawn_buff_dot()

        try:
            # Move Pac-Man
            new_pacman_pos = self.get_new_position(self.pacman_pos, action)
            if self.is_valid_position(new_pacman_pos):
                # Update Pac-Man's position
                self.pacman_pos = new_pacman_pos

                # Check what's in the new position
                cell = self.grid[self.pacman_pos]
                if cell == 1:
                    reward += self.PACDOT_REWARD
                    self.dots_left -= 1
                    self.grid[self.pacman_pos] = 0  # Remove pac-dot
                    # print(f"Pac-Man ate a pac-dot at {self.pacman_pos}.")
                elif cell == 5:
                    reward += self.BUFF_PACDOT_REWARD
                    self.power_mode = True
                    self.buff_timer = self.buff_duration
                    self.grid[self.pacman_pos] = 0  # Remove buff pac-dot
                    self.buff_dots.discard(self.pacman_pos)
                    # print(f"Pac-Man ate a buff pac-dot at {self.pacman_pos}. Power mode activated.")
            else:
                # Invalid move (wall or out of bounds)
                reward += self.MOVE_PENALTY  # Penalize invalid moves
                # print(f"Pac-Man attempted invalid move to {new_pacman_pos}. Penalized.")

            # Move Ghosts
            for ghost in self.ghosts:
                ghost.move(self.pacman_pos)
                if ghost.alive and ghost.state == self.pacman_pos:
                    if self.power_mode:
                        # Pac-Man eats the ghost
                        reward += self.EAT_GHOST_REWARD
                        ghost.alive = False
                        ghost.respawn_timer = ghost.respawn_steps
                        # print(f"Pac-Man ate Ghost {ghost.id} at {ghost.state}. Ghost will respawn shortly.")
                    else:
                        # Ghost eats Pac-Man
                        reward += self.LOSE_PENALTY
                        done = True
                        # print(f"Pac-Man was eaten by Ghost {ghost.id} at {ghost.state}. Game Over.")
                        break  # Exit loop since Pac-Man is dead

            # Check for win condition
            if self.dots_left == 0:
                reward += self.WIN_REWARD
                done = True
                # print("Pac-Man has collected all pac-dots. You Win!")

            # Handle Power Mode Timer
            if self.power_mode:
                self.buff_timer -= 1
                if self.buff_timer <= 0:
                    self.power_mode = False
                    # print("Power mode has worn off.")

            # Distance-based reward adjustments
            pac_r, pac_c = self.pacman_pos
            ghost_distances = [
                (abs(pac_r - ghost.state[0]) + abs(pac_c - ghost.state[1])) / (self.grid.shape[0] + self.grid.shape[1])
                for ghost in self.ghosts if ghost.alive
            ]
            min_ghost_distance = min(ghost_distances) if ghost_distances else 10

            # Encourage moving towards pac-dots
            pac_dots_positions = list(zip(*np.where(self.grid == 1)))
            if pac_dots_positions:
                distances_to_dots = [
                    (abs(pac_r - dot_r) + abs(pac_c - dot_c)) / (self.grid.shape[0] + self.grid.shape[1])
                    for dot_r, dot_c in pac_dots_positions
                ]
                min_dot_distance = min(distances_to_dots)
                reward += (1 - min_dot_distance) * 0.5  # Increased scaling factor

            # Discourage moving towards ghosts
            if min_ghost_distance < 1.0:
                reward -= (1.0 - min_ghost_distance) * 0.3  # Smaller scaling factor

            # If all ghosts are dead, prioritize eating pac-dots
            if all(not ghost.alive for ghost in self.ghosts):
                # Additional reward for collecting pac-dots when no ghosts are present
                reward += self.PACDOT_REWARD * 0.5  # Encourage collecting pac-dots

            total_dots = np.sum(self.grid == 1)
            milestone_threshold = total_dots // 10  # Reward every 10% of pac-dots collected
            if self.dots_left % milestone_threshold == 0 and self.dots_left != 0:
                reward += self.MILSTONE_REWARD

            # Increment step counter and check for step limit
            self.current_step += 1
            if self.current_step >= self.max_steps:
                done = True
                truncated = True
                reward += self.LOSE_PENALTY  # Optional: Penalize for truncation
                # print(f"Episode truncated after {self.max_steps} steps.")

        except IndexError as e:
            # print(f"IndexError encountered during step: {e}")
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
                    pygame.draw.circle(self.screen, (255, 255, 0), (x + self.scale // 2, y + self.scale // 2), 3)
                elif cell == 2:
                    # Wall
                    pygame.draw.rect(self.screen, (0, 0, 255), (x, y, self.scale, self.scale))  # Blue walls
                elif cell == 5:
                    # Buff Pac-dot (now replaces pac-dots and doesn't blink)
                    if self.buff_pac_dot_image:
                        # If image available, blit the image
                        self.screen.blit(self.buff_pac_dot_image, (x + self.scale // 4, y + self.scale // 4))
                    else:
                        # Otherwise, draw a different colored circle
                        pygame.draw.circle(self.screen, (255, 0, 255), (x + self.scale // 2, y + self.scale // 2), 5)

        # Draw Pac-Man
        pac_x, pac_y = self.state_to_pos(self.pacman_pos)
        if self.pacman_mouth_open:
            self.screen.blit(self.pac_man_image_open, (pac_x, pac_y))
        else:
            self.screen.blit(self.pac_man_image_closed, (pac_x, pac_y))

        # Draw Ghosts
        for ghost in self.ghosts:
            if ghost.alive:
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
        if any(ghost.state == position and ghost.alive for ghost in self.ghosts):
            return False  # Another ghost is occupying the position
        return True

    def is_valid_position_for_ghost(self, position):
        """
        Check if the ghost's new position is within bounds and not a wall.

        Parameters:
            position (Tuple[int, int]): The (row, col) position to check.

        Returns:
            bool: True if valid for ghost movement, False otherwise.
        """
        row, col = position
        rows, cols = self.grid.shape
        if not (0 <= row < rows and 0 <= col < cols):
            return False  # Out of bounds
        if self.grid[position] == 2:
            return False  # Wall
        return True

    def get_observation(self):
        """
        Construct the observation vector.

        Returns:
            np.array: Observation vector of shape (24,).
        """
        # Normalize Pac-Man's position
        pac_r, pac_c = self.pacman_pos
        pac_r_norm = pac_r / self.grid.shape[0]
        pac_c_norm = pac_c / self.grid.shape[1]

        # Distances to all ghosts (Manhattan distance, normalized)
        ghost_distances = [
            (abs(pac_r - ghost.state[0]) + abs(pac_c - ghost.state[1])) / (self.grid.shape[0] + self.grid.shape[1])
            for ghost in self.ghosts if ghost.alive
        ]
        # If no ghosts are alive, set distances to 1
        while len(ghost_distances) < self.num_ghosts:
            ghost_distances.append(1.0)

        # Number of pac-dots left (normalized)
        initial_dots = np.sum(self.grid == 1) + (self.dots_left if hasattr(self, 'dots_left') else 0)
        dots_left_norm = self.dots_left / initial_dots if initial_dots > 0 else 0.0

        # One-hot encoding for ghost directions (Up, Down, Left, Right for each ghost)
        ghost_directions = []
        for ghost in self.ghosts:
            if not ghost.alive:
                ghost_directions.extend([0, 0, 0, 0])  # No direction info for dead ghosts
                continue
            delta_row = self.pacman_pos[0] - ghost.state[0]
            delta_col = self.pacman_pos[1] - ghost.state[1]
            if abs(delta_row) > abs(delta_col):
                direction = 1 if delta_row > 0 else 0  # Down or Up
            elif abs(delta_col) > 0:
                direction = 3 if delta_col > 0 else 2  # Right or Left
            else:
                direction = -1  # Same position
            if direction != -1:
                one_hot = [0, 0, 0, 0]
                one_hot[direction] = 1.0
                ghost_directions.extend(one_hot)
            else:
                ghost_directions.extend([0, 0, 0, 0])  # No direction if overlapping

        # Convert ghost_directions to float32
        ghost_directions = np.array(ghost_directions, dtype=np.float32)

        # Proximity to walls (Up, Down, Left, Right)
        walls = []
        for action in range(4):
            new_pos = self.get_new_position(self.pacman_pos, action)
            walls.append(0.0 if self.is_valid_position(new_pos) else 1.0)
        walls = np.array(walls, dtype=np.float32)

        # Local dot density (within a radius of 2)
        dot_density = self.get_local_dot_density(radius=2)
        dot_density = np.array([dot_density], dtype=np.float32)

        # Power mode indicator
        power_mode_indicator = np.array([1.0 if self.power_mode else 0.0], dtype=np.float32)

        # Convert scalar features to float32
        pac_r_norm = np.float32(pac_r_norm)
        pac_c_norm = np.float32(pac_c_norm)
        dots_left_norm = np.float32(dots_left_norm)

        # Convert distances to float32
        ghost_dist_norm = np.array(ghost_distances, dtype=np.float32)

        # Concatenate all features into a single state vector
        state_vector = np.concatenate((
            [pac_r_norm, pac_c_norm],
            ghost_dist_norm,          # Distances to all ghosts
            [dots_left_norm],
            ghost_directions,         # Directions of all ghosts
            walls,
            dot_density,
            power_mode_indicator      # Power mode status
        ), axis=0)

        # Ensure the state_vector is float32
        state_vector = state_vector.astype(np.float32)

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

    def spawn_buff_dot(self):
        """
        Spawn buff pac-dots by replacing existing pac-dots.
        Ensure that there are at most 2 buff pac-dots at any time.
        Only spawn a new buff pac-dot when one is eaten.
        """
        max_buff_dots = 2
        current_buff_dots = len(self.buff_dots)
        spawn_chance = 0.05  # 5% chance to attempt spawning a buff dot each step

        # Only spawn if current buff dots are less than max
        if current_buff_dots < max_buff_dots and random.random() < spawn_chance:
            pacdot_positions = list(zip(*np.where(self.grid == 1)))
            if pacdot_positions:
                pos = random.choice(pacdot_positions)
                self.grid[pos] = 5  # Replace pac-dot with buff pac-dot
                self.buff_dots.add(pos)
                # print(f"Buff pac-dot spawned at {pos} by replacing a pac-dot.")

    def assign_new_chasing_ghost(self):
        """
        Assign the 'chase' strategy to a new ghost.
        This method ensures that only one ghost is chasing at any time.
        """
        # Find ghosts that are alive and not currently chasing
        available_ghosts = [ghost for ghost in self.ghosts if ghost.alive and ghost.id != self.current_chasing_ghost_id]
        
        if not available_ghosts:
            # No available ghosts to assign
            self.current_chasing_ghost_id = None
            return
        
        # Select a new ghost to chase (e.g., randomly)
        new_chasing_ghost = random.choice(available_ghosts)
        new_chasing_ghost.current_strategy = 'chase'
        self.current_chasing_ghost_id = new_chasing_ghost.id
        # print(f"Ghost {new_chasing_ghost.id} is now chasing Pac-Man.")

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
