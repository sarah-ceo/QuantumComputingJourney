import random
import numpy as np
import gymnasium as gym
from gymnasium import logger, spaces
import pygame
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
            low=0, high=255, shape=(self.img_size, self.img_size, 3), dtype=np.uint8
        )

        self.screen_width = 600
        self.screen_height = 600
        self.screen = None

        self.cell_w = self.screen_width // self.grid_size
        self.cell_h = self.screen_height // self.grid_size

    def reset(self, seed: int | None = None):
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
        exit_position = np.random.randint(1, self.grid_size-1, size=2)
        self.state[exit_position[0], exit_position[1]] = 1

        # Agent
        self.agent_position = np.random.randint(1, self.grid_size-1, size=2)
        while np.all(self.agent_position == exit_position):
            self.agent_position = np.random.randint(1, self.grid_size-1, size=2)
        self.state[self.agent_position[0], self.agent_position[1]] = 42

        # Inner walls
        n_inner_walls = random.randint(0, self.grid_size)
        all_directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
        diagonals = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        for _ in range(n_inner_walls):
            walls = np.where(self.state == -1)
            start_index = random.randint(0, len(walls[0])-1)
            root = np.array([walls[0][start_index], walls[1][start_index]])
            possible_directions = [d for d in all_directions if np.all(root+d >= 0) and np.all(root+d < self.grid_size) and np.all(self.state[(root+d)[0], (root+d)[1]] == 0)]
            if len(possible_directions) > 0:
                direction = random.choice(possible_directions)
                row_or_col = np.where(direction != 0)[0]
                look_ahead_directions = [d for d in np.vstack((all_directions, diagonals)) if d[row_or_col] != -1*direction[row_or_col]]
                length = random.randint(2, self.grid_size)
                for step in range(1, length+1):
                    pos_delta = direction * step
                    if self.state[(root+pos_delta)[0], (root+pos_delta)[1]] != 0:
                        break
                    if np.any(np.array([self.state[(root+pos_delta+d)[0], (root+pos_delta+d)[1]] for d in look_ahead_directions]) == -1):
                        break
                    self.state[(root+pos_delta)[0], (root+pos_delta)[1]] = -1

        state_img = self.render_image()
        return state_img, {}

    def step(self, action: int):
        assert self.action_space.contains(action), (
            f"{action!r} ({type(action)}) invalid"
        )
        assert self.state is not None, "Call reset before using step method."

        self.steps += 1

        move = self.actions_correspondences[action]
        new_agent_position = self.agent_position + move

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

        state_img = self.render_image()
        return state_img, reward, terminated, truncated, {}

    def render_image(self):
        frame = np.transpose(self.render(), (1, 0, 2))
        frame = frame[self.cell_h:-self.cell_h, self.cell_w:-self.cell_w, :]
        frame = np.array(Image.fromarray(frame).resize(size=(self.img_size, self.img_size)))
        return frame

    def render(self):
        if self.screen is None:
            pygame.display.init()
            self.screen = pygame.Surface((self.screen_width, self.screen_height))

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        colors = {
            -1: (0, 0, 0),         # Walls: black
            0: (255, 255, 255),   # Empty space: white
            1: (0, 255, 0),       # Exit: green
            42: (0, 0, 255),       # Agent: blue
        }

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                value = self.state[r, c]
                color = colors[value]

                rect = pygame.Rect(
                    c * self.cell_w,
                    r * self.cell_h,
                    self.cell_w,
                    self.cell_h,
                )

                pygame.draw.rect(self.surf, color, rect)
                pygame.draw.rect(self.surf, (150, 150, 150), rect, width=1)

        self.screen.blit(self.surf, (0, 0))

        frame = pygame.surfarray.array3d(self.screen)

        frame = np.transpose(frame, (1, 0, 2))

        return frame

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.ion()

    seed = 42
    env = Maze()
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    frame = env.render()
    img = plt.imshow(frame)
    plt.axis("off")
    for i in range(500):
        env.render()
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            env.reset()
        else:
            frame = env.render()
            img.set_data(frame)

            plt.draw()
            plt.pause(0.5)
        

    env.close()
    plt.ioff()
    plt.show()