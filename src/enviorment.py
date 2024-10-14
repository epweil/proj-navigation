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

from feature_extraction import FIASS_Embedding, Query_Image, Refrence_Image
from helper import uncertinity_function
# from setuptools import setup

CORRECT_MOVE_PROBABILITY = 0.775
SAVED = True

class GPSD_ENV(gym.Env):
      metadata = { "render_fps": 4}
      def __init__(self, refrence_image_path, query_image_path, render=None, size  = 10):
            self.render_mode = render
            super(GPSD_ENV, self).__init__()
            self.hex_radius = 50
            
            
            self.FIASS = FIASS_Embedding()
            self.refrence_image = Refrence_Image(refrence_image_path, self.FIASS, self.hex_radius, SAVED)
            self.query_image = Query_Image(query_image_path, self.hex_radius, SAVED)
            
            self.window_size = max(self.query_image.image.size[0], self.query_image.image.size[1])
            
            self.observation_space= spaces.Dict({
                  'agent': spaces.Box(0, size - 1, shape=(2,), dtype=int),
                  'target': spaces.Box(0, size - 1, shape=(2,), dtype=int),
            })
            
            self.action_space = spaces.Discrete(6)
            

            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))  
            pygame.display.set_caption('Hexagonal Grid Environment')
            
            self.background_color = (255, 255, 255)
            self.hex_color = (255, 0, 0)
            self.agent_color = (0, 0, 255)
            self.target_color = (0, 255, 0)
            self.predicted_locatation_color = (255, 255, 255)

            # Hexagon size and geometry calculations
            
            self.hex_height = math.sqrt(3) * self.hex_radius
            self.hex_width = 2 * self.hex_radius
            self.size = self.window_size // self.hex_width
            
            self.size_width = math.ceil(self.window_size / self.hex_height ) 
            self.size_height = math.ceil(self.window_size / self.hex_width)
            
            
            
            
      
            
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
            
     
            
      def _get_info(self, action_in = None):
            if(action_in is None):
                  predicted_location_x = self._agent_predicted_location[0]
                  predicted_location_y = self._agent_predicted_location[1] 
            else:
                  number_to_check = 2
                  loc = []
                  probability_of_position_locations = []
                  best_feature_mapping = []
                  loc
                  for ind, y in enumerate(range(max(0,self._agent_predicted_location[0] - number_to_check), min(self._agent_predicted_location[0]+number_to_check, self.size_width))):
                        probability_of_position_locations.append([])
                        best_feature_mapping.append([])
                        loc.append([])
                        for x in range(max(0,self._agent_predicted_location[1] - number_to_check), min(self._agent_predicted_location[1]+number_to_check, self.size_height)):
                              probability_of_position_locations[ind].append(uncertinity_function(self._agent_predicted_location, (x,y)) )
                              best_feature_mapping[ind].append(self.query_image.get_best_guess_of_positon((x,y), self.refrence_image, self.FIASS, top_k=1))
                              loc.append([y,x])
                  best_feature_mapping = np.asarray(best_feature_mapping)
                  probability_of_position_locations = np.asarray(probability_of_position_locations)
                  
                  probability_of_position_locations = probability_of_position_locations / probability_of_position_locations.max()
                  probability_of_position_locations = 1-(10*probability_of_position_locations)
                  # probability_of_position_locations = 10*probability_of_position_locations / probability_of_position_locations.mean()
                  
                  best_feature_mapping = best_feature_mapping / best_feature_mapping.max()
                  best_feature_mapping = best_feature_mapping / best_feature_mapping.mean()
                  
                  maping = np.asarray(best_feature_mapping * probability_of_position_locations)
                  predicted_location_y,predicted_location_x = loc[maping.argmax()]
                  print(predicted_location_y,predicted_location_x)
                  
                  # location_from_features_x *= probability_of_position_locations
                  # location_from_features_y *= probability_of_position_locations
                  # predicted_location_x = round(location_from_features_x.mean())
                  # predicted_location_y = round(location_from_features_y.mean())
                  # location_from_features_y, location_from_features_x
                  
            return({
                  'distance': np.linalg.norm(np.array([predicted_location_x,predicted_location_y]) - np.array(self._target_location), ord=1),
                  'predicted_locatation': (predicted_location_x,predicted_location_y)
            })
      def reset(self, starting_pos = None, seed = None, ):
            
            
            # super().reset(seed=seed, options={'starting_pos':starting_pos})
            super().reset(seed=seed)

            # Choose the agent's location uniformly at random
            self._agent_location = [self.np_random.integers(0, self.size_height, size=1, dtype=int)[0], self.np_random.integers(0, self.size_width,size=1, dtype=int)[0]]
            self._agent_predicted_location = self._agent_location
            self._target_location = [self.np_random.integers(0, self.size_height, size=1, dtype=int)[0], self.np_random.integers(0, self.size_width, size=1, dtype=int)[0]]

            # We will sample the target's location randomly until it does not coincide with the agent's location
            # self._target_location = (5,10)
            # self._agent_location = (1,1)
            
            observation = self._get_obs()
            info = self._get_info()

            if self.render_mode == "human":
                  self.render()
            

            return observation, info
      
      def step(self,action_in):
            #ADD NOISE 
            action = action_in
            if(random.random() <= 1-CORRECT_MOVE_PROBABILITY):
                  action = random.randint(0,5)
                  
                  
            direction = self.get_movement_from_action(action)
            direction_predicted = self.get_movement_from_action(action_in)
            self._agent_predicted_location = self._agent_predicted_location + direction_predicted
            self._agent_location = self._agent_location + direction
            
            got_to_target = np.array_equal(self._agent_location, self._target_location)
            
            observation = self._get_obs()
            info = self._get_info(action_in)
            self._agent_predicted_location = info['predicted_locatation']
            reward = -info['distance']
            
            if(got_to_target):
                  reward = 100000
                  terminated = True
                  print("Success")
            elif(self._agent_location.min() < 0 or self._agent_location.max() > max(self.size_height, self.size_width)):
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
            img = pygame.image.load(self.query_image.image_path)

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
                        if np.array_equal(self._target_location, np.array([q, r])):
                              pygame.draw.circle(self.screen, self.target_color, (int(x), int(y)), self.hex_radius // 2)
                        # If this hexagon is the predicted position, draw the agent
                        if np.array_equal(self._agent_predicted_location, np.array([q, r])):
                              pygame.draw.circle(self.screen, self.predicted_locatation_color, (int(x), int(y)), self.hex_radius // 4)

            # Update the display
            pygame.display.flip()
            pygame.display.flip()
           
                  
      def close(self):
          if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            
if __name__ == '__main__':
      env = GPSD_ENV('./../images/2016.png', './../images/2018.png', render='human')
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
                  if (terminated):
                        env.reset()


      # Render the environment
      

      

      env.close()
