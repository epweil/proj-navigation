import gymnasium as gym
import gymnasium
from matplotlib import patches
import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.registration import register
from setuptools import setup


class GPSD_ENV(gym.Env):
      
      def __init__(self, render=None, size  = 5):
            self.size = size
            self.window_size = 512
            
            self.observation_space= spaces.dict({
                  'agent': spaces.Box(0, size - 1, shape=(2,), dtype=int),
                  'target': spaces.Box(0, size - 1, shape=(2,), dtype=int),
            })
            
            self.action_space = spaces.Discrete(6)
            
            self._action_to_direction = {
                  'ne': np.array([1, 1]),
                  'se': np.array([-1, 1]),
                  'nw': np.array([1, -1]),
                  'sw': np.array([-1, -1]),
                  'e': np.array([1, 0]),
                  'w': np.array([-1, 0]),
            }
            
            
            
      def _get_obs(self):
            return {"agent": self._agent_location, "target": self._target_location}
      
      def _get_info(self):
            return({
                  'distance': np.lialg.norm(self._agent_location - self._target_location, ord=1)
            })
      def reset(self, starting_pos = None, seed = None, ):
            
            
            # super().reset(seed=seed, options={'starting_pos':starting_pos})
            super().reset(seed=seed)

            # Choose the agent's location uniformly at random
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

            # We will sample the target's location randomly until it does not coincide with the agent's location
            self._target_location = (5,10)
            self._agent_location = (1,1)
            
            observation = self._get_obs()
            info = self._get_info()

            if self.render_mode == "human":
                  self._render_frame()

            return observation, info
      
      def step(self,action):
            
            direction = self._action_to_direction[action]
            
            self._agent_location = self._agent_location + direction
            
            got_to_target = np.array_equal(self._agent_location, self._target_location)
            
            observation = self._get_obs()
            info = self._get_info()
            
            reward = -observation.distance 
            
            if(got_to_target):
                  reward = 1
                  terminated = True
            elif(self._agent_location.min() < 0 or self._agent_location.max() > self.size):
                  reward = -100
                  terminated = True
            else:
                  reward = observation.distance
                  terminated = False
                  
            if self.render_mode == "human":
                  self._render_frame()
            return observation, reward, terminated, info 

      def render(self):
            if self.render_mode == "rgb_array":
                  return self._render_frame()

      def hex_corner(self, center, size, i):
            """Helper function to calculate hexagon corners."""
            angle_deg = 60 * i + 30
            angle_rad = math.pi / 180 * angle_deg
            return (center[0] + size * math.cos(angle_rad), center[1] + size * math.sin(angle_rad))

      def draw_hexagon(self, surface, color, center, radius):
            """Draw a hexagon on the Pygame surface."""
            corners = [self.hex_corner(center, radius, i) for i in range(6)]
            pygame.draw.polygon(surface, color, corners, 1)
        
        
      def _render_frame(self):
            if self.window is None and self.render_mode == "human":
                  pygame.init()
                  pygame.display.init()
                  self.window = pygame.display.set_mode(
                        (self.window_size, self.window_size)
                  )
            if self.clock is None and self.render_mode == "human":
                  self.clock = pygame.time.Clock()

            canvas = pygame.Surface((self.window_size, self.window_size))
            canvas.fill((255, 255, 255))
            
            # Clear screen with background color
            self.window.fill('#FFFFFF')

            # Draw hexagonal grid
            for q in range(self.size):
                  for r in range(self.size):
                        # Calculate hexagon center position
                        x = (3/4 * q)
                        y = (r + 0.5 * (q % 2))

                        # Draw hexagon
                        self.draw_hexagon(self.window, '#000000', (x + 50, y + 50), self.hex_radius)

                        # If this hexagon is the agent's position, draw the agent
                        if np.array_equal(self._agent_location, np.array([q, r])):
                              pygame.draw.circle(self.window, '#0000FF', (int(x + 50), int(y + 50)), self.hex_radius // 3)
                              
                        if np.array_equal(self._target_location, np.array([q, r])):
                              pygame.draw.circle(self.window, '#FF0000', (int(x + 50), int(y + 50)), self.hex_radius // 3)




            

            # Finally, add some gridlines
            for x in range(self.size + 1):
                  pygame.draw.line(
                        canvas,
                        0,
                        (0, pix_square_size * x),
                        (self.window_size, pix_square_size * x),
                        width=3,
                  )
                  pygame.draw.line(
                        canvas,
                        0,
                        (pix_square_size * x, 0),
                        (pix_square_size * x, self.window_size),
                        width=3,
                  )

            if self.render_mode == "human":
                  # The following line copies our drawings from `canvas` to the visible window
                  self.window.blit(canvas, canvas.get_rect())
                  pygame.event.pump()
                  pygame.display.update()

                  # We need to ensure that human-rendering occurs at the predefined framerate.
                  # The following line will automatically add a delay to keep the framerate stable.
                  self.clock.tick(self.metadata["render_fps"])
            else:  # rgb_array
                  return np.transpose(
                        np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
                  )
      def close(self):
          if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            
      if __name__ == "__main__":
            register(
                  id="gym_examples/GridWorld-v0",
                  entry_point="gym_examples.envs:GridWorldEnv",
                  max_episode_steps=300,
            )
            
            setup(
            name="gym_examples",
            version="0.0.1",
            install_requires=["gymnasium==0.26.0", "pygame==2.1.0"],
            )
            import gym_examples
            env = gymnasium.make('gym_examples/GridWorld-v0')