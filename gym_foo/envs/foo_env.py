import gym
from gym import error, spaces, utils
from gym.utils import seeding

class FooEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, num_containers, avg_mem_utilization):
    self.action_space = spaces.Discrete(2) #There are two actions: upscale and downscale
    self.observation_space = spaces.Tuple((
      spaces.Discrete(TOTAL_NUMBER_OF_CONTAINERS), #total number of containers
      spaces.Discrete(50), #number of containers being used
      spaces.Box(np.array([0]), np.array([100]),dtype=np.float16)) #average memory utilization 
    
  def step(self, action):
    ...
  def reset(self):
    ...
  def render(self, mode='human'):
    ...
  def close(self):
    ...
