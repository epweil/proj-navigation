from stable_baselines3 import DDPG, DQN
from enviorment import GPSD_ENV
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np 
import torch 
env = GPSD_ENV('./../images/test.png')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create the model
model = DQN("MultiInputPolicy", env, verbose=1, device=device, learning_rate = 0.01)
model.learn(total_timesteps=10000)
model.save("DQN_test")