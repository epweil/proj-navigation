

from stable_baselines3 import DDPG, DQN, A2C
from Landscape import Region
from enviorment import GPSD_ENV
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np 
import torch 
from stable_baselines3.dqn.policies import QNetwork
from stable_baselines3.common.env_checker import check_env
print("STARTING")
landscape = Region((103.809643,1.243246,103.904400,1.325623), hex_size=25) 
landscape.images[2018].Input_Image.image
import importlib
import Landscape
import enviorment 
import stable_baselines3
import tracker
import matplotlib.pyplot as plt
import pandas as pd 
from stable_baselines3.common.vec_env import VecFrameStack
Landscape = importlib.reload(Landscape)
enviorment = importlib.reload(enviorment)
tracker = importlib.reload(tracker)



model = DQN.load('./../models/model_done')
track = tracker.Tracker(landscape.images[2018].Input_Image.num_hexes_width,landscape.images[2018].Input_Image.num_hexes_height)
env_eval = enviorment.GPSD_ENV(landscape.images[2018], start_position=[10,10], target_position=[3,3], tracker = track, mode='RGB')
observation, info = env_eval.reset()
terminated = False
run_reward = 0
iterat = 0
start = [0,0]
while not terminated:
      action = model.predict(observation, deterministic  = True)
      observation, reward, terminated, k, info  = env_eval.step(int(action[0]))
      run_reward += reward
      iterat += 1
print(track.trials)