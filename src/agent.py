from stable_baselines3 import DDPG, DQN
from Landscape import Region
from enviorment import GPSD_ENV
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np 
import torch 
from stable_baselines3.dqn.policies import QNetwork
landscape = Region((11.22378,35.22622, 11.02178,35.02822), hex_size=25) 
env = GPSD_ENV(landscape.images[2016], render='human', start_position=[1,1], target_position=[10,10])





device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create the model
model = DQN("MultiInputPolicy", env, verbose=1, device=device, learning_rate = 0.1)

model.learn(total_timesteps=1000)
model.save("DQN_test")

