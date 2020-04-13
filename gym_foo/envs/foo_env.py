import pandas as pd
import random
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

TOTAL_NUMBER_OF_CONTAINERS = 10
INITIAL_NUMBER_OF_CONTAINERS = 5
MAX_STEPS = 1000

class FooEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.action_space = spaces.Discrete(2) #There are two actions: upscale and downscale
    # print(self.action_space)
    self.total_containers = TOTAL_NUMBER_OF_CONTAINERS
    self.num_containers = INITIAL_NUMBER_OF_CONTAINERS
    self.avg_mem_utilization = random.randint(20,80)
    self.next_state = random.randint(1,3)
    self.current_state = random.randint(1,3)
    self.current_action = 0
    self.min_no_containers = 2

    high = np.array([
          self.num_containers,
          self.total_containers,
          self.current_state,
          self.next_state,
          self.current_action])

    self.observation_space = spaces.Box(-high, high, dtype=np.int8)
    # self.observation_space = spaces.Tuple((
    #   spaces.Box(low = 0, high = TOTAL_NUMBER_OF_CONTAINERS, shape=(1,TOTAL_NUMBER_OF_CONTAINERS), dtype = np.int8), #total number of containers
    #   spaces.Box(low = 0, high = INITIAL_NUMBER_OF_CONTAINERS, shape=(1,INITIAL_NUMBER_OF_CONTAINERS), dtype = np.int8), #number of containers being used
    #   spaces.Box(low = -1, high = 1,shape=(5,),dtype=np.int8))) #The range of actions possible (-1 for downscale and +1 for upscale)
    
    
  def step(self, action):
    # Execute one time step within the environment
        self.current_step += 1

        #if self.current_step > TOTAL_NUMBER_OF_CONTAINERS:
        #self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS) #delay modifier is epsilon

        #Creating the DataFrame 
        #self.df = pd.DataFrame([[0, 0, 3, 0, 1, 0], [0, -3, 3, 3, -1, 0], [0, 0, 0, +3, -1, -1]],
        #index=['S1', 'S2', 'S3'],
        #columns=['S1d', 'S1u', 'S2d', 'S2u', 'S3d', 'S3u'])

        x = 0

        if self.avg_mem_utilization < 30.0:
          self.current_state = self.next_state
          self.next_state = 1 #S1
#          print('----------------------------------------------------------------------------- ')

        elif self.avg_mem_utilization >= 30.0 and self.avg_mem_utilization < 70.0:
          self.current_state = self.next_state
          self.next_state = 2 #'S2'
#          print('-----------------------------------------------------------------------------')
        elif self.avg_mem_utilization >= 70.0:
          self.current_state = self.next_state
          self.next_state = 3 #'S3'
#          print('-----------------------------------------------------------------------------')

          self.current_action = action


        if self.next_state == 1:
         if self.current_state == 1:
            if self.current_action == 0: # 0 means upscaling
              x = 0
            elif self.current_action == 1: # 1 means downscaling
              x = 0
        
        if self.next_state == 1:
         if self.current_state == 2:
            if self.current_action == 0:
              x = 0
            elif self.current_action == 1:
              x = 3

        if self.next_state == 1:
         if self.current_state == 3:
            if self.current_action == 0:
              x = 0
            elif self.current_action == 1:
              x = 1

        if self.next_state == 2:
         if self.current_state == 1:
            if self.current_action == 0:
              x = -3
            elif self.current_action == 1:
              x = 0

        if self.next_state == 2:
         if self.current_state == 2:
            if self.current_action == 0:
              x = 3
            elif self.current_action == 1:
              x = 3

        if self.next_state == 2:
         if self.current_state == 3:
            if self.current_action == 0:
              x = 0
            elif self.current_action == 1:
              x = -1

        if self.next_state == 3:
         if self.current_state == 1:
            if self.current_action == 0:
              x = 0
            elif self.current_action == 1:
              x = 0

        if self.next_state == 3:
         if self.current_state == 2:
            if self.current_action == 0:
              x = 3
            elif self.current_action == 1:
              x = 0

        if self.next_state == 3:
         if self.current_state == 3:
            if self.current_action == 0:
              x = -1
            elif self.current_action == 1:
              x = -1

        reward = x * delay_modifier

        # print(f'Reward: {reward}')
        # print(f'x: {x}')
        # print(f'delay_modifier: {delay_modifier}')
        # print(f'action: {action}')
        # print(f'Current state: {self.current_state}')
        # print(f'Next state: {self.next_state}')
        # print(f'-----------------------------------------------------------------------------')
        # print(f'-----------------------------------------------------------------------------s')


        self._take_action(action)

        done = self.current_step >= MAX_STEPS
        self.avg_mem_utilization = random.randint(20,80)

        obs = self._next_observation()

        return obs, reward, done, {}
  
  def reset(self):
    self.num_containers = INITIAL_NUMBER_OF_CONTAINERS
    # Set the current step to a random point within the range
    self.current_step = 1
    self.avg_mem_utilization = random.randint(20,80)

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
    
    # np.append(obs,frame, axis=0)
    return frame

  def _take_action(self, action):
    if action == 0: #upscaling action
      if self.num_containers == self.total_containers:
        return
      self.num_containers += 1
      # self.next_state = self.current_state
      self.current_action = 0
    elif action == 1:
      if self.num_containers == self.min_no_containers:
        return
         #downscaling action
      self.num_containers -= 1
      # self.next_state = self.current_state
      self.current_action = 1
  
  def render(self, mode='human', close=False):
    # Render the environment to the screen
    # print(f'Step: {self.current_step}')
    # print(f'Average memory utilization: {self.avg_mem_utilization}')
    # print(f'Total number of containers: {self.total_containers}')
    # print(f'Number of containers in use: {self.num_containers}')
    # print(f'Current state: {self.current_state}')
    # print(f'Next state: {self.next_state}')
    # print(f'Current action: {self.current_action}')
    return

