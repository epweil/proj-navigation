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

from torch import uint8

from Landscape import Region, Year_ENV
# from setuptools import setup

λ = 1
class GPSD_ENV(gym.Env):
      metadata = { "render_fps": 4}
      def __init__(self, Y_ENV:Year_ENV, render=None, start_position = None , target_position= None, tracker = None, mode= 'RGB'):
            self.render_mode = render
            super(GPSD_ENV, self).__init__()
            
            
            self.tracker = tracker
            self.area = Y_ENV
            self.mode = mode
            self.action_space = spaces.Discrete(6)
            
            if(start_position):
                  try:
                        self.area.Input_Image.hexagon_images[start_position[0]][start_position[1]]
                  except:
                        raise Exception("Start Location Not IN Grid Shape")
            self.starting_locatation = start_position 
            if(target_position):
                  try:
                        self.area.Input_Image.hexagon_images[target_position[0]][target_position[1]]
                  except:
                        raise Exception("Start Location Not IN Grid Shape")
            self.target_locatation = target_position
                  
           
            
            
            
            
            self.window_size = max(self.area.Input_Image.image.size[0], self.area.Input_Image.image.size[1])
            
            self.hex_height = self.area.Input_Image.hex_height
            self.hex_width = self.area.Input_Image.hex_width
            self.size_width = self.area.Input_Image.num_hexes_width
            self.size_height = self.area.Input_Image.num_hexes_height
            
            ###MODE RGB VS L FOR PHOTOS 
            if(mode == 'RGB'):
                  self.observation_space= spaces.Box(low=0, high=255, shape=(self.area.Input_Image.hexagon_images[0][0].cropped_size[0],self.area.Input_Image.hexagon_images[0][0].cropped_size[1], 3),dtype=np.uint8)
            else:
                  self.observation_space= spaces.Box(low=0, high=255, shape=(1,self.area.Input_Image.hexagon_images[0][0].cropped_size[0],self.area.Input_Image.hexagon_images[0][0].cropped_size[1]),dtype=np.uint8)
            
            
            
            
            
            
            
            
            
            
            ###RENDER PARAMS 
            if(self.render_mode =='human'):
                  pygame.init()
                  self.screen = pygame.display.set_mode((self.window_size, self.window_size))  
                  pygame.display.set_caption('Hexagonal Grid Environment')
                  
            self.background_color = (255, 255, 255)
            self.hex_color = (255, 255, 255)
            self.agent_color = (0, 0, 255)
            self.target_color = (0, 255, 0)
            self.predicted_locatation_color = (255, 255, 255)


            
            
            
            
            
            
            
            
            
      
      """
      SQUARE GRID -> HEX MOVEMENTS 
      """
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
             
      
      ### GET HEX PICTURE BELOW 
      def _get_obs(self):
            arr = np.array(self.area.Input_Image.hexagon_images[self._agent_location[0]][self._agent_location[1]].cropped_image.convert(self.mode)).astype(np.uint8)
            if(self.mode == 'L'):
                  arr = arr.reshape((1,arr.shape[0], arr.shape[1]))
            return arr
      
            
     

      def _get_info(self, action_in = None):
            return {}
            
      
      """
      RESET FUNCTION
      """
      def reset(self, seed = None ):
            
            self.steps = 0
            super().reset(seed=seed)
            
            # START THE TARGET IF THE TARGET LOCATATION SHOULD BE RANDOM
            if(self.target_locatation is None):
                  self.target_locatation = [np.random.randint(0, self.area.Input_Image.num_hexes_height, size=1, dtype=int)[0], np.random.randint(0, self.area.Input_Image.num_hexes_width,size=1, dtype=int)[0]] 
            else:
                  self.target_locatation = self.target_locatation
                  
            # START THE AGENT AT A RANDOM LOCATION (NOT THE TARGET) IF THE STARTING LOCATATION IS RANDOM
            if(self.starting_locatation is None):
                  self._agent_location = [np.random.randint(0, self.area.Input_Image.num_hexes_height, size=1, dtype=int)[0], np.random.randint(0, self.area.Input_Image.num_hexes_width,size=1, dtype=int)[0]] 
                  
                  while (self._agent_location == self.target_locatation):
                        self._agent_location = [np.random.randint(0, self.area.Input_Image.num_hexes_height, size=1, dtype=int)[0], np.random.randint(0, self.area.Input_Image.num_hexes_width,size=1, dtype=int)[0]]    
            else:
                  self._agent_location = self.starting_locatation
            
            # RESET TRACKER
            if(self.tracker):
                  self.tracker.reset()
                  self.tracker.step(None, self._agent_location, 0)
                  
            self.running_reward = 0
            
            observation = self._get_obs()
            info = self._get_info()

            if self.render_mode == "human":
                  self.render()
            return observation, info
      
      
      
      """
      STEP FUNCTION FOR AGENT
      """
      def step(self,action_in):
            
            
            self.steps += 1
            #CURRENTLY NOT LOOKING AT DRIFT IN THE ENVIORMENT 
            action = self.area.move_drift(action_in)
            action = action_in 
            direction = self.get_movement_from_action(action)
            self._agent_location = self._agent_location + direction
            
            
            
            if(self.mode == 'RGB'):
                  observation = np.zeros((self.area.Input_Image.hexagon_images[0][0].cropped_size[0],self.area.Input_Image.hexagon_images[0][0].cropped_size[1], 3))
            else:
                  observation = np.zeros((1,self.area.Input_Image.hexagon_images[0][0].cropped_size[0],self.area.Input_Image.hexagon_images[0][0].cropped_size[1]))
            info = {}
            
            
            
            
            
            #SUCESS 
            got_to_target = np.array_equal(self._agent_location,self.target_locatation)
            if(got_to_target):
                  reward = 10
                  terminated = True
                  info['SUCESS'] = True
            
            #GOES OFF THE MAP
            elif(self._agent_location.min() < 0 or self._agent_location[0] >= self.size_height or  self._agent_location[1] >= self.size_width):
                  reward = -10
                  terminated = True
                  info['SUCESS'] = False
            #GOES OVER 25 MOVES 
            elif(self.steps > 25):
                  reward = -10
                  terminated = True
                  info['SUCESS'] = False
            #ELSE 
            else:
                  terminated = False
                  info = self._get_info(action_in)
                  observation = self._get_obs()
                  reward = -0.1 
            
            self.running_reward += reward
            
            #TRACKER FOR ANALYSIS AND PENELTY FOR RETUTNING TO THE SAME LOCATION
            if(self.tracker):
                  past_positions = np.asarray(self.tracker.positions[-1])
                  if(np.asarray(self._agent_location) in past_positions):
                        reward -= 0.5
                  self.tracker.step(action_in, self._agent_location, self.running_reward)
                  
                  
            
            if self.render_mode == "human":
                  self.render()
            return observation, reward, terminated, False, info 





      """
      FUNCTIONS TO HELP WITH RENDERING THE GAME IN HUMAN MODE 
      """
      def hex_corner(self, center, size, i):
            """Helper function to calculate hexagon corners."""
            angle_deg = 60 * i
            angle_rad = math.pi / 180 * angle_deg
            return (center[0] + size * math.cos(angle_rad), center[1] + size * math.sin(angle_rad))

      def draw_hexagon(self, surface, color, center, radius):
            """Draw a hexagon on the Pygame surface."""
            corners = [self.hex_corner(center, radius, i) for i in range(6)]
            pygame.draw.polygon(surface, color, corners, 1)


      """
      RENDER THE GAME FOR HUMAN MODE 
      """
      def render(self, mode='human'):
            # Clear screen with background color
            img = pygame.image.load(self.area.Input_Image.image_path)

            self.screen.blit(img, (0, 0))

            # Draw hexagonal grid
            for q in range(self.size_height):
                  for r in range(self.size_width):
                  # Calculate hexagon center position
                        x = r * (self.hex_width - (math.cos(1.0472) * self.area.Input_Image.hex_radius)) 

                        y = q * (self.hex_height) + ((r%2) * math.sin(1.0472) * self.area.Input_Image.hex_radius)

                        # Draw hexagon
                        self.draw_hexagon(self.screen, self.hex_color, (x, y), self.area.Input_Image.hex_radius)

                        # If this hexagon is the agent's position, draw the agent
                        if np.array_equal(self._agent_location, np.array([q, r])):
                              pygame.draw.circle(self.screen, self.agent_color, (int(x), int(y)), self.area.Input_Image.hex_radius // 2)
                         # If this hexagon is the target's position, draw the agent
                        if np.array_equal(self.target_locatation, np.array([q, r])):
                              pygame.draw.circle(self.screen, self.target_color, (int(x), int(y)), self.area.Input_Image.hex_radius // 2)


            pygame.display.flip()
           
                  
      def close(self):
          if self.screen is not None:
            pygame.display.quit()
            pygame.quit()