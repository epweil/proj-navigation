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
import random

from Landscape import Region, Year_ENV
# from setuptools import setup

λ = 1

class GPSD_ENV(gym.Env):
      metadata = { "render_fps": 4}
      def __init__(self, Y_ENV:Year_ENV, render=None, size  = 10):
            self.render_mode = render
            super(GPSD_ENV, self).__init__()
            self.hex_radius = 50
            
            self.area = Y_ENV
            
            
            
            
            self.window_size = max(self.area.Input_Image.image.size[0], self.area.Input_Image.image.size[1])
            
            self.observation_space= spaces.Dict({
                  'agent': spaces.Box(0, size - 1, shape=(2,), dtype=int),
                  'target': spaces.Box(0, size - 1, shape=(2,), dtype=int),
            })
            
            self.action_space = spaces.Discrete(6)
            

            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))  
            pygame.display.set_caption('Hexagonal Grid Environment')
            
            self.background_color = (255, 255, 255)
            self.hex_color = (255, 255, 255)
            self.agent_color = (0, 0, 255)
            self.target_color = (0, 255, 0)
            self.predicted_locatation_color = (255, 255, 255)

            # Hexagon size and geometry calculations
            
            self.hex_height = self.area.Input_Image.hex_height
            self.hex_width = self.area.Input_Image.hex_width
            
            self.size_width = self.area.Input_Image.num_hexes_width
            self.size_height = self.area.Input_Image.num_hexes_height
            
            print(self.area.Input_Image.num_hexes_width, self.area.Input_Image.num_hexes_height)
            
            
            
            
      
            
      def get_movement_from_action(self, action):
            
            if(self._agent_location[1] % 2 == 0):
                  _action_to_direction = {
                  1: np.array([-1, 1]), #NE
                  2: np.array([0, 1]), #SE
                  5: np.array([-1, -1]),#NW
                  4: np.array([0, -1]), #SW
                  3: np.array([1,0]), #South 
                  0: np.array([-1, 0]) #North 
                  }
            else:
                  _action_to_direction = {
                  1: np.array([0, 1]), #NE
                  2: np.array([1, 1]),#SE
                  5: np.array([0, -1]),#NW
                  4: np.array([1, -1]), #SW
                  3: np.array([1,0]), #South 
                  0: np.array([-1, 0]) #North 
                  }
            movement = _action_to_direction[action]
            

                        
                        
                  
            return movement
             
            
            
      def _get_obs(self):
            return {"agent": self._agent_location, "target": self.area.target_locatation}
            
     
            
      def _get_info(self, action_in = None):
            if(action_in is None):
                  MLP_SCORE = 0
            
            else:
                  MLP_SCORE = 0
                  


                  
            
                  
            return({
                  'prediction_similiarity': MLP_SCORE
            })
            
            
      def reset(self, starting_pos = None, seed = None, ):
            
            
            super().reset(seed=seed)

            self._agent_location = self.area.starting_locatation
            observation = self._get_obs()
            info = self._get_info()

            if self.render_mode == "human":
                  self.render()
            

            return observation, info
      
      
      
      
      def step(self,action_in):
            
            action = self.area.move_drift(action_in)
            direction = self.get_movement_from_action(action)
            self._agent_location = self._agent_location + direction
            
            got_to_target = np.array_equal(self._agent_location,self.area.target_locatation)
            observation = self._get_obs()
            info = self._get_info(action_in)
            reward = -1 +  λ*info['prediction_similiarity']
            
            
            
            if(got_to_target):
                  reward = 100000
                  terminated = True
                  print("Success")
            elif(self._agent_location.min() < 0 or self._agent_location.max() > max(self.size_height, self.size_width)):
                  reward = -100000
                  terminated = True
                  print("Fail")
            else:
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
            img = pygame.image.load(self.area.Input_Image.image_path)

            self.screen.blit(img, (0, 0))

            # Draw hexagonal grid
            for q in range(self.size_height):
                  for r in range(self.size_width):
                  # Calculate hexagon center position
                        x = r * (self.hex_width - (math.cos(1.0472) * self.hex_radius)) 

                        y = q * (self.hex_height) + ((r%2) * math.sin(1.0472) * self.hex_radius)

                        # Draw hexagon
                        self.draw_hexagon(self.screen, self.hex_color, (x, y), self.hex_radius)

                        # If this hexagon is the agent's position, draw the agent
                        if np.array_equal(self._agent_location, np.array([q, r])):
                              pygame.draw.circle(self.screen, self.agent_color, (int(x), int(y)), self.hex_radius // 2)
                         # If this hexagon is the target's position, draw the agent
                        if np.array_equal(self.area.target_locatation, np.array([q, r])):
                              pygame.draw.circle(self.screen, self.target_color, (int(x), int(y)), self.hex_radius // 2)

            # Update the display
            pygame.display.flip()
            pygame.display.flip()
           
                  
      def close(self):
          if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            
if __name__ == '__main__':
      landscape = Region((11.22378,35.22622, 11.02178,35.02822)) 
      env = GPSD_ENV(landscape.images[2018], render='human')
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
                              observation, reward, terminated, trun,  info  = env.step(0)  
                        elif event.key == pygame.K_s:  # Move south
                              observation, reward, terminated,trun, info  =env.step(3) 
                        elif event.key == pygame.K_e: # Move north-east
                              observation, reward, terminated, trun,info  =env.step(1)  
                        elif event.key == pygame.K_d: # Move south-east
                              observation, reward, terminated, trun,info  =env.step(2)  
                        elif event.key == pygame.K_q:
                             observation, reward, terminated, trun,info  =env.step(5)  # Move north-west
                        elif event.key == pygame.K_a:
                              observation, reward, terminated, trun,info  =env.step(4)  # Move south-west
                        env.render()
                  if (terminated):
                        env.reset()


      # Render the environment
      

      

      env.close()
