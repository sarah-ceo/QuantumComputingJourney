import random
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import logger, spaces
from PIL import Image

class Maze(gym.Env):
    ''' Maze environment where the agent needs to find the exit without touching the walls. Observations are an image. '''
    def __init__(self, grid_size: int = 8, img_size: int = 64):
        self.grid_size = grid_size
        self._max_episode_steps = grid_size ** 2
        self.action_space = spaces.Discrete(4)
        self.actions_correspondences = {
            0: np.array([-1, 0]),
            1: np.array([0, -1]),
            2: np.array([1, 0]),
            3: np.array([0, 1])
        }
        self.img_size = img_size
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.img_size, self.img_size, 1), dtype=np.float32
        )

        self.color_map = {
            -1: (0, 0, 0),        # Walls: black
            0: (255, 255, 255),   # Empty space: white
            1: (0, 255, 0),       # Exit: green
            42: (0, 0, 255),      # Agent: blue
        }

        self.simplified_color_map = {
            -1: 85,      # Walls: red
            0: 0,   # Empty space: white
            1: 170,       # Exit: green
            42: 255,      # Agent: blue
        }

    def compute_distance_map(self):
        dist = np.full((self.grid_size, self.grid_size), np.inf)
        exit_pos = tuple(np.argwhere(self.state == 1)[0])
        dist[exit_pos] = 0
        q = deque([exit_pos])
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size \
                and self.state[nr, nc] != -1 and dist[nr, nc] == np.inf:
                    dist[nr, nc] = dist[r, c] + 1
                    q.append((nr, nc))
        return dist

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

        # Exit
        exit_position = np.array([random.randint(1, self.grid_size-2), random.randint(1, self.grid_size-2)])
        self.state[exit_position[0], exit_position[1]] = 1

        # Inner walls
        if curriculum_level is None:
            min_inner_walls, max_inner_walls, max_wall_length, max_agent_exit_dist = 0, self.grid_size, self.grid_size, self.grid_size
        else:
            (min_inner_walls, max_inner_walls, max_wall_length, max_agent_exit_dist) = curriculum_level
        
        # Agent
        distance_matrix = np.maximum(
            np.abs(np.arange(self.grid_size)[:, None] - exit_position[0]),
            np.abs(np.arange(self.grid_size)[None, :] - exit_position[1])
        ).astype(np.float32)
        distance_matrix[np.where(self.state == -1)] = np.inf
        possible_agent_positions = np.where((distance_matrix > 0) & (distance_matrix <= max_agent_exit_dist))
        random_index = random.randint(0, len(possible_agent_positions[0])-1)
        self.agent_position = np.array([possible_agent_positions[0][random_index], possible_agent_positions[1][random_index]])
        self.state[self.agent_position[0], self.agent_position[1]] = 42

        # Inner walls
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

        self.distance_map = self.compute_distance_map()
        self.episode_max_steps = self.distance_map[self.agent_position[0], self.agent_position[1]]
        return self.create_simple_image(), {}

    def step(self, action: int):
        assert self.action_space.contains(action), (
            f"{action!r} ({type(action)}) invalid"
        )
        assert self.state is not None, "Call reset before using step method."

        self.steps += 1

        move = self.actions_correspondences[action]
        new_agent_position = self.agent_position + move

        if self.steps > self.episode_max_steps:
            reward = -1
        else:
            reward = self.state[new_agent_position[0], new_agent_position[1]]
        truncated = self.steps >= self._max_episode_steps
        terminated = reward == -1 or reward == 1

        if self.steps_beyond_terminated is not None:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned terminated = True. "
                    "You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
        elif terminated:
            self.steps_beyond_terminated = 0
        else:
            self.state[self.agent_position[0], self.agent_position[1]] = 0
            self.agent_position = new_agent_position
            self.state[self.agent_position[0], self.agent_position[1]] = 42

        return self.create_simple_image(), reward, terminated, truncated, {}
    
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
        return self.expand_image(color_grid)
    
    def render(self):
        img = self.create_simple_image()
        # boundaries = np.round(np.arange(1, self.grid_size + 1) * self.img_size / self.grid_size).astype(int)
        # boundary_color = (90, 90, 90)
        # boundaries = boundaries[boundaries < self.img_size]
        # img[boundaries, :, :] = boundary_color
        # img[:, boundaries, :] = boundary_color
        return img
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.ion()

    curriculum_level = None

    seed = 42
    env = Maze()
    env.reset(seed=seed, curriculum_level=curriculum_level)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    frame = env.render()
    img = plt.imshow(frame)
    plt.axis("off")

    for i in range(500):
        action = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            env.reset(curriculum_level=curriculum_level)

        frame = env.render()
        img.set_data(frame)
        
        plt.draw()
        plt.pause(5)

    plt.ioff()
    plt.show()