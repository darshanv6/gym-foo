import gym
from gym import error, spaces, utils
from gym.utils import seeding

TOTAL_NUMBER_OF_CONTAINERS = 50
INITIAL_NUMBER_OF_CONTAINERS = 10

class FooEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, num_containers, avg_mem_utilization):
    self.action_space = spaces.Discrete(2) #There are two actions: upscale and downscale
    self.observation_space = spaces.Tuple((
      spaces.Discrete(TOTAL_NUMBER_OF_CONTAINERS), #total number of containers
      spaces.Discrete(50), #number of containers being used
      spaces.Box(np.array([0]), np.array([100]),dtype=np.float16))) #average memory utilization 
    
  def step(self, action):
    ...
  def reset(self):
    self.num_containers = INITIAL_NUMBER_OF_CONTAINERS
    # Set the current step to a random point within the data frame
    self.current_step = random.randint(0, len(self.num_containers.loc[:, 'Open'].values) - 6)
    return self._next_observation()

  def _next_observation(self):
    

  def render(self, mode='human'):
    ...
  def close(self):
    ...
