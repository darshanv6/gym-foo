import logging
import numpy as np

from rl_exceptions import PolicyDoesNotExistException,InsufficientPolicyParameters,FunctionNotImplemented

logging.basicConfig()
logger = logging.getLogger()
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
                raise InsufficientPolicyParameters("EpsilonDecay policy also requires the min_epsilon parameters")
            if "epsilon_decay_rate" not in self.policy_parameters:
                raise InsufficientPolicyParameters("EpsilonDecay policy also requires the epsilon_decay_rate parameter")
            self.min_epsilon = self.policy_parameters["min_epsilon"]
            self.epsilon_decay_rate = self.policy_parameters["epsilon_decay_rate"]
            return self.return_epsilon_greedy_policy()
        else:
            raise PolicyDoesNotExistException("The selected policy does not exist! The implemented policies are epsilon-greedy and epsilon-decay. ")

    def return_epsilon_decay_policy(self):

	    def choose_action_by_epsilon_decay(values_of_all_possible_actions):
	        logger.debug("Taking e-decay action for action values"+str(values_of_all_possible_actions))
	        prob_taking_best_action_only = 1-self.epsilon 
	        prob_taking_any_random_action = self.epsilon/self.n_actions
	        action_probability_vector = [prob_taking_any_random_action]*self.n_actions
	        exploitation_action_index = np.argmax(values_of_all_possible_actions)
	        action_probability_vector[exploitation_action_index]+= prob_taking_best_action_only
	        chosen_action = np.random.choice(np.arange(self.n_actions),p=action_probability_vector)
	        if self.epsilon >self.min_epsilon:
	            self.epsilon *= self.epsilon_decay_rate
	        logger.debug("decayed epsilon value after the current iteration: {}".format(self.epsilon))
	        return chosen_action
	    return choose_action_by_epsilon_decay

    def return_epsilon_greedy_policy(self):
    
	    def choose_action_by_epsilon_greedy(values_of_all_possible_actions):
	        logger.debug("Taking e-greedy action for action values"+str(values_of_all_possible_actions))
	        prob_taking_best_action_only = 1-self.epsilon 
	        prob_taking_any_random_action = self.epsilon/self.n_actions
	        action_probability_vector = [prob_taking_any_random_action]*self.n_actions
	        exploitation_action_index = np.argmax(values_of_all_possible_actions)
	        action_probability_vector[exploitation_action_index]+= prob_taking_best_action_only
	        chosen_action = np.random.choice(np.arange(self.n_actions),p=action_probability_vector)
	        return chosen_action
	    return choose_action_by_epsilon_greedy

if __name__ == "__main__":
    raise FunctionNotImplemented("This class needs to be imported and instaniated from  a reinforcement learning agent class and does not contain any invokable code in the main function")
