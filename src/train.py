from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from Landscape import Region
from enviorment import GPSD_ENV
import torch 
from tracker import Tracker
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np



class CustomEvalCallback(EvalCallback):
      def __init__(self, eval_env, best_model_save_path, log_path, eval_freq):
            self.eval_env = eval_env
            self.best_model_save_path = best_model_save_path
            self.log_path = log_path
            self.eval_freq = eval_freq
            self.vals = []
            self.suc = []
            self.runs = []
            super().__init__(eval_env=eval_env, best_model_save_path=best_model_save_path, log_path=log_path, eval_freq=eval_freq)

      def custom_evaluation(self,eval_env,  model):
            suc = []
            vals = []
            env_eval = GPSD_ENV(landscape.images[2018], start_position=None, target_position=target_position, tracker = None, mode=mode)
            
            #GET RESULTS FROM EACH STARTING LOCATATION
            for y in range(env_eval.size_height):
                  for x in range(env_eval.size_width):
                        if([x,y] != target_position):
                              env_eval = GPSD_ENV(landscape.images[2018], start_position=[y,x], target_position=target_position, tracker = None, mode=mode)
                              track_eval.clear()
                              observation, info = env_eval.reset()
                              terminated = False
                              run_reward = 0
                              iterat = 0
                              while not terminated:
                                    action = model.predict(observation, deterministic  = True)
                                    observation, reward, terminated, k, info  = env_eval.step(int(action[0]))
                                    run_reward += reward
                                    iterat += 1
                                    
                              suc.append(info['SUCESS'])
                              vals.append(run_reward)

                        
            return np.asarray(suc).mean(), np.asarray(vals).mean()

      def _on_step(self) -> bool:
            if self.eval_freq > 0 and ((self.n_calls ) % self.eval_freq == 0 or self.n_calls ==1):
                  success_rate, mean_reward = self.custom_evaluation(self.eval_env, self.model)
                  
                  self.model.save(f'./../models/model_{self.n_calls}')
                  self.vals.append(mean_reward)
                  self.suc.append(success_rate)
                  self.runs.append(self.n_calls)
                  
                  # Log or print custom metrics
                  print(f"Step {self.n_calls}: Mean Reward = {mean_reward}, Success Rate = {success_rate}")

            return True
      def save(self):
            df = pd.DataFrame([self.runs, self.vals,self.suc])
            df.to_csv("./../metrics/RESULTS.csv")
            
            plt.figure()
            plt.plot(self.runs, self.vals)
            plt.savefig("./../metrics/Rewards.png")
            plt.figure()
            plt.plot(self.runs, self.suc)
            plt.savefig("./../metrics/Sucesses.png")
            
      



print("STARTING")
landscape = Region((103.809643,1.243246,103.904400,1.325623), hex_size=25) 

mode = 'RGB'
target_position = [3,3]

print("STARTING TRAINING")
track = Tracker(landscape.images[2018].Input_Image.num_hexes_width,landscape.images[2018].Input_Image.num_hexes_height)
track_eval = Tracker(landscape.images[2018].Input_Image.num_hexes_width,landscape.images[2018].Input_Image.num_hexes_height)
env = GPSD_ENV(landscape.images[2018], start_position=None, target_position=target_position, tracker = track, mode = mode)

env_eval = GPSD_ENV(landscape.images[2018], start_position=None, target_position=target_position, tracker = track_eval, mode = mode)

track.clear()
track_eval.clear()


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")



sucesses = []
rewards = []
runs = []
model = DQN(
      policy="CnnPolicy",
      env=env,
      learning_rate=0.001,              
      buffer_size=10000,               
      batch_size=32,
      gamma=0.99,
      exploration_initial_eps=0.1,     
      exploration_final_eps=0.001,
      target_update_interval=10000,     
      learning_starts=10000,            
      max_grad_norm=1,                
      train_freq=10,
      verbose=0,
      policy_kwargs=dict(normalize_images=False))

t_steps = 10000
max_train = 250000

custom_eval_callback = CustomEvalCallback(
      eval_env=env_eval,
      best_model_save_path='./../model_best/',
      log_path='./../models/',
      eval_freq=t_steps)


      
model.learn(total_timesteps=max_train, progress_bar = True, callback=custom_eval_callback)
model.save(f"./../models/model_done")
custom_eval_callback.save()




