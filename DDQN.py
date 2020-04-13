""" DQN in code - BehaviourPolicy """

import logging
import numpy as np
from itertools import count
import matplotlib.pyplot as plt
import time
import os
import gym
import gym_foo


from tensorflow import keras
from tensorflow.python.keras import backend 
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.losses import mean_squared_error

from experience_replay import SequentialDequeMemory
from behaviour_policy import BehaviourPolicy

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class DoubleDQN:
    def __init__(self,agent_name=None,env=gym.make('foo-v0'),number_episodes = 500,discounting_factor = 0.9,learning_rate=0.001,behaviour_policy = "epsilon_decay",policy_parameters={"epsilon":1.0,"min_epsilon":0.01,"epsilon_decay_rate":0.99}, deep_learning_model_hidden_layer_configuration = [32,16,8]):
        self.agent_name="ddqa_"+str(time.strftime("%Y%m%d-%H%M%S")) if agent_name is None else agent_name
        self.model_weights_dir = "model_weights"
        self.env = env
        # print(env.observation_space[2].shape[0])
        self.n_states = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.n_episodes = number_episodes
        self.episodes_completed = 0
        self.gamma = discounting_factor
        self.alpha = learning_rate
        self.policy = BehaviourPolicy(n_actions = self.n_actions, policy_type = behaviour_policy,policy_parameters = policy_parameters).getPolicy()
        self.PolicyParameter = policy_parameters
        self.model_hidden_layer_configuration = deep_learning_model_hidden_layer_configuration
        self.online_model = self._build_sequential_dnn_model()
        self.target_model = self._build_sequential_dnn_model()
        self.trainingStats_steps_in_each_episode = []
        self.trainingStats_rewards_in_each_episode = []
        self.trainingStats_discountedrewards_in_each_episode = []
        self.memory = SequentialDequeMemory(queue_capacity = 3000)
        self.experience_replay_batch_size = 32

    def _build_sequential_dnn_model(self):
        model = Sequential()
        hidden_layers = self.model_hidden_layer_configuration
        model.add(Dense(hidden_layers[0],input_dim = self.n_states,activation = 'relu'))
        for layer_size in hidden_layers[1:]:
            model.add(Dense(layer_size,activation='relu'))
        model.add(Dense(self.n_actions,activation='linear'))
        model.compile(loss=mean_squared_error,optimizer = Adam(lr = self.alpha))
        print(model.summary())
        print(model.get_weights())
        # exit(0)
        plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        return model

    def _sync_target_model_with_online_model(self):
        self.target_model.set_weights(self.online_model.get_weights())

    def _update_online_model(self,experience_tuple):
        current_state,action,instantaneos_reward,next_state, done_flag = experience_tuple
        action_target_values = self.online_model.predict(current_state)
        action_values_for_state = action_target_values[0]
        if done_flag:
            action_values_for_state[action] = instantaneos_reward
        else:
            action_values_for_next_state = self.target_model.predict(next_state)[0]
            max_next_state_value = np.max(action_values_for_next_state)
            target_action_value = instantaneos_reward+self.gamma*max_next_state_value
            action_values_for_state[action]=target_action_value
        action_target_values[0] = action_values_for_state
        logger.debug("fitting online model with Current_State:{},Action_Value:{}".format(current_state,action_values_for_state))
        self.online_model.fit(current_state,action_target_values,epochs=1)

    def _reshape_state_for_model(self,state):
        return np.reshape(state,[1,self.n_states])

    def train_agent(self):
        self.load_model_weights()
        for episode in range(self.n_episodes):
            logger.debug("-"*30)
            logger.debug("EPISODE {}/{}".format(episode,self.n_episodes))
            logger.debug("-"*30)
            current_state = self._reshape_state_for_model(self.env.reset())
            # print("STATE: ", current_state)
            cumulative_reward = 0
            discounted_cumulative_reward = 0
            for n_step in count():
                # print(self.online_model.predict(current_state))
                all_action_value_for_current_state = self.online_model.predict(current_state)[0]
                policy_defined_action = self.policy(all_action_value_for_current_state)
                next_state, instantaneos_reward,done, _ =self.env.step(policy_defined_action)
                next_state = self._reshape_state_for_model(next_state)
                experience_tuple = (current_state,policy_defined_action,instantaneos_reward,next_state,done)
                self.memory.add_to_memory(experience_tuple)
                cumulative_reward+= instantaneos_reward
                discounted_cumulative_reward=instantaneos_reward+self.gamma*discounted_cumulative_reward
                if done:
                    self.trainingStats_steps_in_each_episode.append(n_step)
                    self.trainingStats_rewards_in_each_episode.append(cumulative_reward)
                    self.trainingStats_discountedrewards_in_each_episode.append(discounted_cumulative_reward)
                    self._sync_target_model_with_online_model()
                    logger.debug("episode:{}/{},reward:{},discounted_reward:{}".format(n_step,self.n_episodes,cumulative_reward,discounted_cumulative_reward))
                    break
        self.replay_experience_from_memory()
        if episode % 2 == 0: self.plot_training_statistics()
        if episode % 5 == 0: self.save_model_weights()
        return self.trainingStats_steps_in_each_episode,self.trainingStats_rewards_in_each_episode,self.trainingStats_discountedrewards_in_each_episode
        
    def replay_experience_from_memory(self):
        if self.memory.get_memory_size()<self.experience_replay_batch_size:
            return False
        experience_mini_batch = self.memory.get_random_batch_for_replay(batch_size=self.experience_replay_batch_size)
        for experience_tuple in experience_mini_batch:
            self._update_online_model(experience_tuple)
        return True

    def save_model_weights(self,agent_name = None):
        if agent_name is None:
            agent_name = self.agent_name

        model_file = os.path.join(os.path.join(self.model_weights_dir,agent_name+".hs"))
        self.online_model.load_weights(model_file,overwrite=True)

    def load_model_weights(self,agent_name=None):
        if agent_name is None:
            agent_name = self.agent_name
    
        model_file = os.path.join(os.path.join(self.model_weights_dir,agent_name+".hs"))
        if os.path.exists(model_file):
            self.online_model.load_weights(model_file)
            self.online_model.load_weights(model_file)


    
    def plot_training_statistics(self,training_statistics = None):
        steps = self.trainingStats_steps_in_each_episode if training_statistics is None else training_statistics[0]
        rewards = self.trainingStats_rewards_in_each_episode if training_statistics is None else training_statistics[1]
        discounted_rewards = self.trainingStats_discountedrewards_in_each_episode if training_statistics is None else training_statistics[2]
        episodes = np.arange(len(self.trainingStats_steps_in_each_episode))
        fig,ax1=plt.subplots()
        ax1.set_xlabel('Episodes (e)')
        ax1.set_ylabel('Steps To EPisode Completion',color="red")
        ax1.plot(episodes,steps,color="red")
        ax2 = ax1.twinx()
        ax2.set_ylabel('Reward in each episode',color='blue')
        ax2.plot(episodes,rewards,color="blue")
        fig.tight_layout()
        plt.show()

        fig,ax1=plt.subplots()
        ax1.set_xlabel('Episodes (e)')
        ax1.set_ylabel('Steps To EPisode Completion',color="red")
        ax1.plot(episodes,steps,color="red")
        ax2 = ax1.twinx()
        ax2.set_ylabel('Discounted Reward in each episode',color='green')
        ax2.plot(episodes,discounted_rewards,color="green")
        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
    """Main fnction 
        
        A sample implementation of the above DDQN agent for testing purpose
        this function is executed when this file is run from the command promt directly or by selection
    """
    agent = DoubleDQN()
    training_statistics=agent.train_agent()
    agent.plot_training_statistics(training_statistics)
        




