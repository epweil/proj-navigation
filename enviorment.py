import gymnasium as gym
import gymnasium
from matplotlib import patches
import pygame
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gymnasium.envs.registration import register
import math
from PIL import Image
# from setuptools import setup


class GPSD_ENV(gym.Env):
      metadata = { "render_fps": 4}
      def __init__(self, image_path, render=None, size  = 10):
            self.render_mode = render
            super(GPSD_ENV, self).__init__()
            self.img = Image.open(image_path)
            self.img = self.img.resize((self.img.size[0]//2,self.img.size[1]//2))
            self.img_path = image_path
            self.window_size = self.img.size[0]
            
            self.observation_space= spaces.Dict({
                  'agent': spaces.Box(0, size - 1, shape=(2,), dtype=int),
                  'target': spaces.Box(0, size - 1, shape=(2,), dtype=int),
            })
            
            self.action_space = spaces.Discrete(6)
            

            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))  # Adjust size as needed
            pygame.display.set_caption('Hexagonal Grid Environment')
            
            self.background_color = (255, 255, 255)
            self.hex_color = (255, 0, 0)
            self.agent_color = (0, 0, 255)
            self.target_color = (0, 255, 0)

            # Hexagon size and geometry calculations
            self.hex_radius = 10
            self.hex_height = math.sqrt(3) * self.hex_radius
            self.hex_width = 2 * self.hex_radius
            self.size = self.window_size // self.hex_width
            
            
      def get_movement_from_action(self, action):
            if(self._agent_location[1] % 2 == 0):
                  _action_to_direction = {
                  0: np.array([-1, 1]),
                  1: np.array([0, 1]),
                  2: np.array([-1, -1]),
                  3: np.array([0, -1]),
                  4: np.array([1,0]),
                  5: np.array([-1, 0])
                  }
            else:
                  _action_to_direction = {
                  0: np.array([0, 1]),
                  1: np.array([1, 1]),
                  2: np.array([0, -1]),
                  3: np.array([1, -1]),
                  4: np.array([1,0]),
                  5: np.array([-1, 0])
                  }
            movement = _action_to_direction[action]
            

                        
                        
                  
            return movement
             
            
            
      def _get_obs(self):
            return {"agent": self._agent_location, "target": self._target_location}
      
      def _get_info(self):
            return({
                  'distance': np.linalg.norm(np.array(self._agent_location) - np.array(self._target_location), ord=1)
            })
      def reset(self, starting_pos = None, seed = None, ):
            
            
            # super().reset(seed=seed, options={'starting_pos':starting_pos})
            super().reset(seed=seed)

            # Choose the agent's location uniformly at random
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

            # We will sample the target's location randomly until it does not coincide with the agent's location
            # self._target_location = (5,10)
            # self._agent_location = (1,1)
            
            observation = self._get_obs()
            info = self._get_info()

            if self.render_mode == "human":
                  self.render()
            

            return observation, info
      
      def step(self,action):
            
            direction = self.get_movement_from_action(action)
            
            self._agent_location = self._agent_location + direction
            
            got_to_target = np.array_equal(self._agent_location, self._target_location)
            
            observation = self._get_obs()
            info = self._get_info()
            
            reward = -info['distance']
            
            if(got_to_target):
                  reward = 100000
                  terminated = True
                  print("Success")
            elif(self._agent_location.min() < 0 or self._agent_location.max() > self.size):
                  reward = -100000
                  terminated = True
                  print("Fail")
            else:
                  reward = -info['distance']**2
                  terminated = False
            if self.render_mode == "human":
                  self.render()
            return observation, reward, terminated, False, info 

      def hex_corner(self, center, size, i):
            """Helper function to calculate hexagon corners."""
            angle_deg = 60 * i
            angle_rad = math.pi / 180 * angle_deg
            return (center[0] + size * math.cos(angle_rad), center[1] + size * math.sin(angle_rad))

      def draw_hexagon(self, surface, color, center, radius):
            """Draw a hexagon on the Pygame surface."""
            corners = [self.hex_corner(center, radius, i) for i in range(6)]
            pygame.draw.polygon(surface, color, corners, 1)

      def render(self, mode='human'):
            # Clear screen with background color
            img = pygame.image.load(self.img_path)

            self.screen.blit(img, (0, 0))

            # Draw hexagonal grid
            for q in range(self.size):
                  for r in range(self.size):
                  # Calculate hexagon center position
                        x = r * (self.hex_width - (math.cos(1.0472) * self.hex_radius)) 

                        y = q * (self.hex_height) + ((r%2) * math.sin(1.0472) * self.hex_radius)

                        # Draw hexagon
                        self.draw_hexagon(self.screen, self.hex_color, (x + 50, y + 50), self.hex_radius)

                        # If this hexagon is the agent's position, draw the agent
                        if np.array_equal(self._agent_location, np.array([q, r])):
                              pygame.draw.circle(self.screen, self.agent_color, (int(x + 50), int(y + 50)), self.hex_radius // 2)
                         # If this hexagon is the target's position, draw the agent
                        if np.array_equal(self._target_location, np.array([q, r])):
                              pygame.draw.circle(self.screen, self.target_color, (int(x + 50), int(y + 50)), self.hex_radius // 2)

            # Update the display
            pygame.display.flip()
            pygame.display.flip()
           
                  
      def close(self):
          if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            
if __name__ == '__main__':
      env = GPSD_ENV('./images/test.png', render='human')
      env.reset()
      env.render()

      # Pygame loop to interact and render with manual control (arrow keys)
      running = True
      while running:
            for event in pygame.event.get():
                  terminated = False
                  if event.type == pygame.QUIT:
                        running = False
                  elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_w: # Move north
                              observation, reward, terminated, trun,  info  = env.step(5)  
                        elif event.key == pygame.K_s:  # Move south
                              observation, reward, terminated,trun, info  =env.step(4) 
                        elif event.key == pygame.K_e: # Move north-east
                              observation, reward, terminated, trun,info  =env.step(0)  
                        elif event.key == pygame.K_d: # Move south-east
                              observation, reward, terminated, trun,info  =env.step(1)  
                        elif event.key == pygame.K_q:
                             observation, reward, terminated, trun,info  =env.step(2)  # Move north-west
                        elif event.key == pygame.K_a:
                              observation, reward, terminated, trun,info  =env.step(3)  # Move south-west
                        env.render()
                  print(terminated)
                  if (terminated):
                        env.reset()


      # Render the environment
      

      

      env.close()
