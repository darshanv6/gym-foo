import pandas as pd
import random
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

TOTAL_NUMBER_OF_CONTAINERS = 50
INITIAL_NUMBER_OF_CONTAINERS = 10
MAX_STEPS = TOTAL_NUMBER_OF_CONTAINERS
TOTAL_NUMBER_OF_CONTAINERS_LABEL = "Total number of containers"
CURRENT_NUMBER_OF_CONTAINERS_LABEL = "Current number of containers"

class FooEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, total_containers, num_containers, avg_mem_utilization):
    self.action_space = spaces.Discrete(2) #There are two actions: upscale and downscale
    self.observation_space = spaces.Tuple((
      spaces.Discrete(TOTAL_NUMBER_OF_CONTAINERS), #total number of containers
      spaces.Discrete(50), #number of containers being used
      spaces.Box(np.array([0]), np.array([100]),dtype=np.float16))) #average memory utilization 
    self.num_containers = num_containers
    self.total_containers = TOTAL_NUMBER_OF_CONTAINERS
    self.avg_mem_utilization = avg_mem_utilization
    
  def step(self, action):
    # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        if self.current_step > TOTAL_NUMBER_OF_CONTAINERS:
            self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = 3 * delay_modifier
        done = self.avg_mem_utilization >= 0.70

        obs = self._next_observation()

        return obs, reward, done, {}
  
  def reset(self):
    self.num_containers = INITIAL_NUMBER_OF_CONTAINERS
    # Set the current step to a random point within the range
    self.current_step = random.randint(0, TOTAL_NUMBER_OF_CONTAINERS)
    return self._next_observation()

  def _next_observation(self):
    frame = np.array([
          random.randint(0,TOTAL_NUMBER_OF_CONTAINERS),
          random.randint(0,TOTAL_NUMBER_OF_CONTAINERS),
        ])

    # Append additional data and scale each value to between 0-1
    obs = np.append(frame, [[
      self.num_containers,
      self.total_containers,
    ]], axis=0)
    return obs


  def _take_action(self, action):
    if action < 1: #upscaling action
      self.num_containers+=1;
    elif action < 2:
      self.num_containers-=1;
  
  def render(self, mode='human', close=False):
    # Render the environment to the screen
    print(f'Step: {self.current_step}')
    print(f'Average memory utilization: {self.avg_mem_utilization}')
    print(f'Number of containers: {self.num_containers}')