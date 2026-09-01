import random
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image

class Maze(gym.Env):
    ''' Maze environment where the agent needs to find the exit without touching the walls.
        Internally, a simple grid is used but the observations are given as an image. '''
    def __init__(self, grid_size: int = 8, img_size: int = 64):
        self.grid_size = grid_size

        # 4 actions possible
        self.action_space = spaces.Discrete(4)
        self.actions_correspondences = {
            0: np.array([-1, 0]),   # Up
            1: np.array([0, -1]),   # Left
            2: np.array([1, 0]),    # Down
            3: np.array([0, 1])     # Right
        }

        # The observation is a single channel image with pixel values between 0-1
        self.img_size = img_size
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.img_size, self.img_size, 1), dtype=np.float32
        )
        
        # The values for the single channel image used by the models
        self.simplified_color_map = {
            -1: 85,         # Walls
            0: 0,           # Empty space
            1: 170,         # Exit
            42: 255,        # Agent
        }

        # The rendering will be shown as an RGB image
        self.color_map = {
            -1: (0, 0, 0),          # Walls: black
            0: (255, 255, 255),     # Empty space: white
            1: (0, 255, 0),         # Exit: red
            42: (255, 0, 0),        # Agent: green
        }

    def compute_distance_map(self):
        ''' Computes the distance of each grid cell to the exit at the beginnning of the episode '''
        path_map = np.full((self.grid_size, self.grid_size), np.inf)
        exit_position = tuple(np.argwhere(self.state == 1)[0])
        path_map[exit_position] = 0
        map_exploration_queue = deque([exit_position])
        while map_exploration_queue:
            row, column = map_exploration_queue.popleft()
            for vertical_move, horizontal_move in list(self.actions_correspondences.values()):
                new_row, new_column = row+vertical_move, column+horizontal_move
                if 0 <= new_row < self.grid_size and 0 <= new_column < self.grid_size and self.state[new_row, new_column] != -1 and path_map[new_row, new_column] == np.inf:
                    path_map[new_row, new_column] = path_map[row, column] + 1
                    map_exploration_queue.append((new_row, new_column))
        return path_map

    def reset(self, seed: int | None = None, curriculum_level = None):
        super().reset(seed=seed)
        self.steps = 0
        self.steps_beyond_terminated = None
        self.state = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

        # Outer walls
        self.state[0, :] = -1 
        self.state[-1, :] = -1
        self.state[:, 0] = -1
        self.state[:, -1] = -1

        # Exit position randomly chosen
        exit_position = np.array([random.randint(1, self.grid_size-2), random.randint(1, self.grid_size-2)])
        self.state[exit_position[0], exit_position[1]] = 1

        # Difficulty settings based on curriculum level
        if curriculum_level is None:
            min_inner_walls, max_inner_walls, max_wall_length, max_agent_exit_dist = 0, self.grid_size, self.grid_size, self.grid_size
        else:
            (min_inner_walls, max_inner_walls, max_wall_length, max_agent_exit_dist) = curriculum_level
        
        # Agent position randomly chosen within difficulty settings
        distance_matrix = np.maximum(
            np.abs(np.arange(self.grid_size)[:, None] - exit_position[0]),
            np.abs(np.arange(self.grid_size)[None, :] - exit_position[1])
        ).astype(np.float32)
        distance_matrix[np.where(self.state == -1)] = np.inf
        possible_agent_positions = np.where((distance_matrix > 0) & (distance_matrix <= max_agent_exit_dist))
        random_index = random.randint(0, len(possible_agent_positions[0])-1)
        self.agent_position = np.array([possible_agent_positions[0][random_index], possible_agent_positions[1][random_index]])
        self.state[self.agent_position[0], self.agent_position[1]] = 42

        # Inner walls randomly created based on difficulty settings
        n_inner_walls = random.randint(min_inner_walls, max_inner_walls)
        all_directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        diagonals = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        for _ in range(n_inner_walls):
            walls = np.where(self.state == -1)
            step = 1
            attemps = 0
            while step <= 1 and attemps<100:
                attemps += 1
                start_index = random.randint(0, len(walls[0])-1)
                root = np.array([walls[0][start_index], walls[1][start_index]])
                possible_directions = [d for d in all_directions if np.all(root+d >= 0) and np.all(root+d < self.grid_size) and np.all(self.state[(root+d)[0], (root+d)[1]] == 0)]
                if len(possible_directions) > 0:
                    direction = random.choice(possible_directions)
                    row_or_col = np.where(direction != 0)[0]
                    look_ahead_directions = [d for d in np.vstack((all_directions, diagonals)) if d[row_or_col] != -1*direction[row_or_col]]
                    length = random.randint(2, max_wall_length)
                    for step in range(1, length+1):
                        pos_delta = direction * step
                        if self.state[(root+pos_delta)[0], (root+pos_delta)[1]] != 0:
                            break
                        if np.any(np.array([self.state[(root+pos_delta+d)[0], (root+pos_delta+d)[1]] for d in look_ahead_directions]) == -1):
                            break
                        self.state[(root+pos_delta)[0], (root+pos_delta)[1]] = -1

        # Compute the distance map and the agent's max allowed steps (shortest path length)
        self.distance_map = self.compute_distance_map()
        self._max_episode_steps = int(self.distance_map[self.agent_position[0], self.agent_position[1]])
        return self.create_simple_image(), {}

    def step(self, action: int):
        self.steps += 1

        # Update the agent's position based on the action taken
        move = self.actions_correspondences[action]
        new_agent_position = self.agent_position + move

        # Set the reward and episode termination
        if self.steps > self._max_episode_steps:
            reward = -1
        else:
            reward = self.state[new_agent_position[0], new_agent_position[1]]
        terminated = reward == -1 or reward == 1

        # A distance_reward has been added to help with learning: 
        # It tells the agent at each step whether its move was good or bad, based on the ideal path computed at initialization
        # It will be stored in the Rollout Buffer
        if terminated:
            distance_reward = reward
        else:
            distance_reward = self.distance_map[self.agent_position[0], self.agent_position[1]] - self.distance_map[new_agent_position[0], new_agent_position[1]]

        # Update the map unless the episode is over
        if self.steps_beyond_terminated is not None:
            self.steps_beyond_terminated += 1
        elif terminated:
            self.steps_beyond_terminated = 0
        else:
            self.state[self.agent_position[0], self.agent_position[1]] = 0
            self.agent_position = new_agent_position
            self.state[self.agent_position[0], self.agent_position[1]] = 42

        return self.create_simple_image(), (reward, distance_reward), terminated, False, {}
    
    def expand_image(self, arr):
        img = Image.fromarray(arr)
        img_resized = img.resize((self.img_size, self.img_size), resample=Image.Resampling.NEAREST)
        return np.array(img_resized, dtype=np.uint8)

    def create_simple_image(self):
        color_grid = np.empty((self.grid_size, self.grid_size, 1), dtype=np.uint8)
        for value, color in self.simplified_color_map.items():
            color_grid[self.state == value] = color
        return np.expand_dims(self.expand_image(np.squeeze(color_grid[1:-1, 1:-1, :], axis=-1)), axis=-1) / 255.0
    
    def create_rgb_image(self):
        color_grid = np.empty((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        for value, color in self.color_map.items():
            color_grid[self.state == value] = color
        img = self.expand_image(color_grid)
        # Add cells outlines
        factor = self.img_size // self.grid_size
        outline_color = (70, 70, 70)
        img[factor::factor, :, :] = outline_color
        img[:, factor::factor, :] = outline_color
        return img
    
    def render(self):
        return self.create_rgb_image()

if __name__ == "__main__":
    # Test the environment
    from itertools import count
    import matplotlib.pyplot as plt
    plt.ion()

    curriculum_level = None

    seed = 42
    env = Maze()
    pause_time = 1

    for i in range(500):
        env.reset(seed=seed, curriculum_level=curriculum_level)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

        frame = env.render()
        img = plt.imshow(frame)
        plt.axis("off")
        plt.title("")
        plt.pause(pause_time)

        for _ in count():
            action = env.action_space.sample()
            _, (_, distance_reward), done, _, _ = env.step(action)

            if done:
                break

            frame = env.render()
            img.set_data(frame)
            
            plt.draw()
            # plt.title(f"Distance reward: {distance_reward}")
            plt.pause(pause_time)

    plt.ioff()
    plt.show()