import logging
import numpy as np

from rl_expections import PolicyDoesNotExistException,InsufficientPolicyParameters, FunctionaNotImplemented

logging.basicConfig()
logging.logging.getLogger()
logger.setLevel(logging.DEBUG)

class BehaviourPolicy:
    def __init__(self,n_actions,policy_type="epsilon_greedy",policy_parameters={"epsilon":0.1}):
        self.policy = policy_type
        self.n_actions = n_actions
        self.policy_type = policy_type
        self.policy_parameters = policy_parameters
        if "epsilon" not in policy_parameters:
            raise InsufficientPolicyParameters("epsilon not availble")
            self.epsilon = self.policy_parameters["epsilon"]
            self.min_epsilon = None
            self.epsilon_decay_rate = None
            logger.debug("Policy Type {},Parameters Received {}".format(policy_type,policy_parameters))


    def getPolicy(self):
        if self.policy_type == "epsilon_greedy":
            return self.return_epsilon_greedy_policy()
        elif self.policy_type == "epsilon_decay":
            self.epsilon = self.policy_parameters["epsilon"]
            if "min_epsilon" not in self.policy_parameters:
                raise
            InsufficientPolicyParameters("EpsilonDecay policy also requires the min_epsilon parameters")
            if "epsilon_decay_rate" not in 
            self.policy_parameters:
            raise
        

