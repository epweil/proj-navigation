import numpy as np
class Tracker():
      def __init__(self,height, width):
            self.width = width
            self.height = height
            self.arr = np.zeros((width+2,height+2))
            self.action = np.zeros((width+2,height+2, 6))
            self.start = np.zeros((width+2,height+2))
            self.reward = np.zeros((width+2,height+2))
            self.reward_back = np.zeros((width+2,height+2))
            self.trials = []
            
      def step(self,action_taken, location, running_reward):
            if(action_taken is None):
                  self.start[location[0]+1, location[1]+1] += 1
            self.trials[-1].append((action_taken, location, running_reward))
            self.arr[location[0]+1, location[1]+1] += 1
            self.reward[location[0]+1, location[1]+1] += running_reward 
            self.action[location[0]+1, location[1]+1, action_taken] += 1 
            
            
      def reset(self):
            if(len(self.trials) > 0):
                  for i in self.trials[-1]:
                        self.reward_back[i[1][0]+1, i[1][1]+1] += self.trials[-1][-1][2]
            self.trials.append([])
            
      def clear(self):
            self.arr = np.zeros((self.width+2,self.height+2))
            self.action = np.zeros((self.width+2,self.height+2, 6))
            self.reward = np.zeros((self.width+2,self.height+2))
            self.reward_back = np.zeros((self.width+2,self.height+2))
            self.trials = []
            
            
            
            