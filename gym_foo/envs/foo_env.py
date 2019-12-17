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

  def __init__(self):
    self.action_space = spaces.Discrete(2) #There are two actions: upscale and downscale
    self.observation_space = spaces.Tuple((
      spaces.Discrete(TOTAL_NUMBER_OF_CONTAINERS), #total number of containers
      spaces.Discrete(INITIAL_NUMBER_OF_CONTAINERS), #number of containers being used
      spaces.Box(low = -1, high = 1,shape=(3,2),dtype=np.int8))) #The range of actions possible (-1 for downscale and +1 for upscale)
    self.total_containers = TOTAL_NUMBER_OF_CONTAINERS
    self.avg_mem_utilization = 20.0
    self.next_state = None
    self.current_state = None
    self.current_action = None
    
  def step(self, action):
    # Execute one time step within the environment
        self.current_step += 1

        #if self.current_step > TOTAL_NUMBER_OF_CONTAINERS:
        #self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)

        #Creating the DataFrame 
        #self.df = pd.DataFrame([[0, 0, 3, 0, 1, 0], [0, -3, 3, 3, -1, 0], [0, 0, 0, +3, -1, -1]],
        #index=['S1', 'S2', 'S3'],
        #columns=['S1d', 'S1u', 'S2d', 'S2u', 'S3d', 'S3u'])

        if self.avg_mem_utilization < 30.0:
          self.next_state = 'S1'
        elif self.avg_mem_utilization >= 30.0 and self.avg_mem_utilization < 70.0:
          self.next_state = 'S2'
        elif self.avg_mem_utilization >= 70.0:
          self.next_state = 'S3'

        if self.next_state == 'S1':
         if self.current_state == 'S1':
            if self.current_action == 1:
              reward = 0
            elif self.current_action == -1:
              reward = 0
        
        if self.next_state == 'S1':
         if self.current_state == 'S2':
            if self.current_action == 1:
              reward = 0
            elif self.current_action == -1:
              reward = 3

        if self.next_state == 'S1':
         if self.current_state == 'S3':
            if self.current_action == 1:
              reward = 0
            elif self.current_action == -1:
              reward = 1

        if self.next_state == 'S2':
         if self.current_state == 'S1':
            if self.current_action == 1:
              reward = -3
            elif self.current_action == -1:
              reward = 0

        if self.next_state == 'S2':
         if self.current_state == 'S2':
            if self.current_action == 1:
              reward = 3
            elif self.current_action == -1:
              reward = 3

        if self.next_state == 'S2':
         if self.current_state == 'S3':
            if self.current_action == 1:
              reward = 0
            elif self.current_action == -1:
              reward = -1

        if self.next_state == 'S3':
         if self.current_state == 'S1':
            if self.current_action == 1:
              reward = 0
            elif self.current_action == -1:
              reward = 0

        if self.next_state == 'S3':
         if self.current_state == 'S2':
            if self.current_action == 1:
              reward = 3
            elif self.current_action == -1:
              reward = 0

        if self.next_state == 'S3':
         if self.current_state == 'S3':
            if self.current_action == 1:
              reward = -1
            elif self.current_action == -1:
              reward = -1

        reward = reward * delay_modifier

        self._take_action(action)

        done = self.current_step >= 40

        obs = self._next_observation()

        return obs, reward, done, {}
  
  def reset(self):
    self.num_containers = INITIAL_NUMBER_OF_CONTAINERS
    # Set the current step to a random point within the range
    self.current_step = random.randint(0, TOTAL_NUMBER_OF_CONTAINERS)
    return self._next_observation()

  def _next_observation(self):
    frame = np.array([
          self.num_containers,
          self.total_containers,
          self.current_state,
          self.next_state,
          self.current_action
        ])

    # Append additional data and scale each value to between 0-1
    obs = np.append(frame, axis=0)
    return obs

  def _take_action(self, action):
    if action < 1: #upscaling action
      self.num_containers+=1
      self.next_state = self.current_state
      self.current_action = 1
    elif action < 2: #downscaling action
      self.num_containers-=1
      self.next_state = self.current_state
      self.current_action = -1
  
  def render(self, mode='human', close=False):
    # Render the environment to the screen
    print(f'Step: {self.current_step}')
    print(f'Average memory utilization: {self.avg_mem_utilization}')
    print(f'Number of containers: {self.num_containers}')
