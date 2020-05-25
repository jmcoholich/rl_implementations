# DQN for OpenAI Gym cartpole
# Jeremiah Coholich 5/10/2020

import gym
import random
import numpy as np
import tensorflow as tf
import sys

# Resources used: 
# https://towardsdatascience.com/dqn-part-1-vanilla-deep-q-networks-6eb4a00febfb (main one)
# https://towardsdatascience.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288

#TODO - increase discount factor (maybe to 1.0) to see if we can get it to hit 500 everytime. 0.99 has a half-life of only 68 timesteps.
class DQN_agent: 
    def __init__(self, existing_model = None):
        # Use MSE or Huber loss (supposedly the latter works better because it doesn't make super large changes)
        if existing_model: #load existing model
            self.model= tf.keras.models.load_model(existing_model)

        else:
            self.model = tf.keras.models.Sequential()
            self.model.add(tf.keras.layers.Dense(50, input_dim =4, activation = 'relu'))
            # self.model.add(tf.keras.layers.Dense(100,activation = 'relu'))
            self.model.add(tf.keras.layers.Dense(50,activation = 'relu'))
            self.model.add(tf.keras.layers.Dense(2,activation = 'linear'))


            # #what if I just try a linear model, so that maybe it will just learn to do pd control essentially
            # self.model = tf.keras.models.Sequential()
            # self.model.add(tf.keras.layers.Dense(2, input_dim =4, activation = 'linear'))
            self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), #so the faster my learning rate, the faster my error increases... But it seems like it increases then decreases 
                      loss='mean_squared_error',
                      metrics=['mean_squared_error']) 


        #initialize member vars
        self.replay_buffer = -np.ones((5000000,11)) #just preallocate a super large replay buffer so I can train for a long time. It should be obs, action, reward, done, next state
        self.idx = 0 #necessary for adding to replay_buffer, since it is preallocated 


        # Training params
        self.batch_size = 2048
        self.discount_factor = 1.0
        self.epsilon_lb = 0.0
        self.epsilon_ub = 0.3
        self.num_episodes = 100
        self.epochs = 1

    def add_to_buffer(self, observation, action, reward, done, next_observation):
        self.replay_buffer[self.idx] = np.concatenate((observation,action,reward, done, next_observation)) 
        self.idx += 1
        
    def sample_from_buffer(self):
        rand_idx = np.random.randint(self.idx,size=self.batch_size)
        batch_X = self.replay_buffer[rand_idx,:]

        batch_Y = self.model(batch_X[:,0:4]).numpy() #np.zeros((self.batch_size,)) #TODO figure out the loss function TODO make sure the buffer includes action -- so it needs to be 11 long

        assert self.batch_size == len(batch_X)
        assert self.batch_size == len(batch_Y)

        #calculate q values
        for i in range(self.batch_size):
            step = batch_X[i,:]
            assert (step != -np.ones(11)).all()
            action = int(step[4])

            assert (action == 1) or (action == 0)
            if step[6]: #if done is true, ie its a terminal step
                # batch_Y[i,action] = step[5] 
                batch_Y[i,action] = -1.0
            else:
                batch_Y[i,action] = step[5] + self.discount_factor * np.max(self.model(np.expand_dims(batch_X[i,7:11],0))) #do fwd pass of dqn to find the maxQval of next state

        return batch_X[:,0:4], batch_Y

    def train(self):
        if self.batch_size > self.idx:
            return

        batch_X, batch_Y = self.sample_from_buffer()
        asdf = self.model.fit(x=batch_X, y=batch_Y, batch_size=self.batch_size, epochs = self.epochs) 
        # print(asdf.history)

    def pick_action(self, observation, epsilon):
        if random.random() < epsilon: # Pick random action
            return random.getrandbits(1)
        else: #pick action based on argmax of fwd pass of DQN
            return int(np.argmax(self.model(np.expand_dims(observation,0))))


# Now do stuff

# Define stuff
def do_training(already_trained_model = None):
    env = gym.make('CartPole-v1')
    dqn = DQN_agent(already_trained_model)

    # Loop through episodes
    for i in range(dqn.num_episodes):
        observation = env.reset()
        step_counter = 0
        # Carry out the episode, storing everything in an experience replay buffer
        while True: 
            # env.render()
            epsilon = dqn.epsilon_ub - (i/dqn.num_episodes)*(dqn.epsilon_ub - dqn.epsilon_lb) 
            action = dqn.pick_action(observation, epsilon)
           
            next_observation, reward, done, _ = env.step(action)
            step_counter +=1
        
            dqn.add_to_buffer(observation, np.array([action]), np.array([reward]), np.array([done]), next_observation)

            observation = next_observation

            if done:
                break

        print("On episode:",i,"Lasted",step_counter,"Epsilon:",epsilon,"replay_buffer_size:",dqn.idx)


        dqn.train()
        print()

    dqn.model.save('test.h5')
    dqn.model.summary()
    test_load_model = tf.keras.models.load_model('test.h5')
    test_load_model.summary()

def do_testing(test_model):
    reward_history = np.zeros(100)
    model = tf.keras.models.load_model(test_model)
    env = gym.make('CartPole-v1')
    for i in range(100):
        step_counter = 0

        observation = env.reset()
        while True: 
            # env.render()
            # print(model(np.expand_dims(observation,0)))
            # print(np.argmax(model(np.expand_dims(observation,0))))
            action = np.argmax(model(np.expand_dims(observation,0)))
           
            observation, reward, done, _ = env.step(action)
            step_counter +=1


            if done:
                break

        reward_history[i] = step_counter
        print(step_counter,"steps")

    print('Final average reward is', np.mean(reward_history))


do_testing('test.h5')

# do_training('test.h5')


