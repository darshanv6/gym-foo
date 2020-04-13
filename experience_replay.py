import logging
import random

from collections import deque
from rl_exceptions import FunctionNotImplemented

logging.basicConfig()
logger = logging.getLogger()

logger.setLevel(logging.DEBUG)

class ExperienceReplayMemory:
    """BAse class for all the extended versions for the 
    ExperienceReplayMemory Class Implementation
    """
    pass

class SequentialDequeMemory(ExperienceReplayMemory):
    """Extension of ExperienceReplayMemory class with
    deque based sequential memory
    """
    def __init__(self,queue_capacity =2000):
        self.queue_capacity = 2000
        self.memory = deque(maxlen=self.queue_capacity)

    def add_to_memory(self,experience_tuple):
        self.memory.append(experience_tuple)

    def get_random_batch_for_replay(self,batch_size=64):
        return random.sample(self.memory,batch_size)

    def get_memory_size(self):
        return len(self.memory)

    if __name__ == "__main__":
        raise FunctionNotImplemented("this class needs to be imported and instantiated from a reinforcement learning agent class and doesnot contain any invokable code in  the main function ")