# DQN for OpenAI Gym cartpole
# Jeremiah Coholich 5/10/2020

import gym
import random
import numpy as np
import sys
import matplotlib.pyplot as plt
from torch import nn, optim, from_numpy, manual_seed
import torch
from torch.utils.tensorboard import SummaryWriter
import time


global writer
global writer_idx
writer = SummaryWriter()
writer_idx = 0



# import wandb
manual_seed(2021)
np.random.seed(2020)
random.seed(2020)

# Resources used: 
# https://towardsdatascience.com/dqn-part-1-vanilla-deep-q-networks-6eb4a00febfb (main one)
# https://towardsdatascience.com/cartpole-introduction-to-reinforcement-learning-ed0eb5b58288

class DQN_agent: 
    def __init__(self, existing_model = None):
        # Use MSE or Huber loss (supposedly the latter works better because it doesn't make super large changes)
        if existing_model: #load existing model
            self.model= torch.load(existing_model)

        else:
            self.model = nn.Sequential(
                nn.Linear(4,32),
                nn.ReLU(),
                nn.Linear(32,2),
                nn.ReLU()
            )
            self.model = self.model.double()

        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()

        #initialize member vars
        self.replay_buffer = -np.ones((50000000,11)) #just preallocate a super large replay buffer so I can train for a long time. It should be obs, action, reward, done, next state
        self.idx = 0 #necessary for adding to replay_buffer, since it is preallocated 


        # Training params
        self.batch_size = 64
        self.discount_factor = 1.0
        self.epsilon_lb = 0.0
        self.epsilon_ub = 0.3
        self.num_episodes = 100000
        self.epochs = 1

    def add_to_buffer(self, observation, action, reward, done, next_observation):
        self.replay_buffer[self.idx] = np.concatenate((observation,action,reward, done, next_observation)) 
        self.idx += 1
        
    def sample_from_buffer(self):
        rand_idx = np.random.randint(self.idx,size=self.batch_size)
        batch_X = self.replay_buffer[rand_idx,:]

        batch_Y = self.model(from_numpy(batch_X[:,0:4])) #np.zeros((self.batch_size,)) #TODO figure out the loss function TODO make sure the buffer includes action -- so it needs to be 11 long

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
                batch_Y[i,action] = step[5] + self.discount_factor * np.max(self.model(from_numpy(batch_X[i,7:11])).detach().numpy()) #do fwd pass of dqn to find the maxQval of next state

        return batch_X[:,0:4], batch_Y

    def train(self):
        global writer_idx
        global writer
        if self.batch_size > self.idx:
            return

        batch_X, batch_Y = self.sample_from_buffer()

        for i in range(1):
            self.optimizer.zero_grad()
            # print('dtype',type(batch_X))
            y_pred = self.model(from_numpy(batch_X))
            loss = self.criterion(y_pred.squeeze(), batch_Y)
            # print('Loss: {}'.format(loss.item()))
            writer.add_scalar('DQN_Loss',loss,writer_idx)
            writer_idx +=1 
            loss.backward()
            self.optimizer.step()
            # asdf = self.model.fit(x=batch_X, y=batch_Y, batch_size=self.batch_size, epochs = self.epochs) 

    def pick_action(self, observation, epsilon):
        if random.random() < epsilon: # Pick random action
            return random.getrandbits(1)
        else: #pick action based on argmax of fwd pass of DQN
            return int(np.argmax(self.model(from_numpy(observation)).detach().numpy()))

    def plot_buffer_data(self):
        # print(self.replay_buffer[:self.idx,:])
        fig, axs = plt.subplots(4)
        names = ['Cart Position','Cart Velocity','Pole Angle','Pole Velocity']
        for i in range(4):
            axs[i].hist(self.replay_buffer[:self.idx,i])
            axs[i].set_title(names[i])
        plt.tight_layout()
        plt.grid()


# Now do stuff

# Define stuff
def do_training(already_trained_model = None):
    global writer_idx
    global writer
    env = gym.make('CartPole-v1')
    env.seed(2020)
    dqn = DQN_agent(already_trained_model)


    # print(dqn.model(from_numpy(np.array([.5,.5,.5,.5]))))
    # sys.exit()

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
        writer.add_scalar('Reward',step_counter,i)
        writer.add_scalar('Epsilon',epsilon,i)


        dqn.train()
        
        if (i%1000) == 0:
            torch.save(dqn.model,'torch_dqn')


def do_testing(test_model):
    reward_history = np.zeros(100)
    model = torch.load(test_model)
    env = gym.make('CartPole-v1')
    for i in range(100):
        step_counter = 0

        observation = env.reset()
        while True: 
            env.render()
            # print(model(np.expand_dims(observation,0)))
            # print(np.argmax(model(np.expand_dims(observation,0))))
            action = int(np.argmax(model(from_numpy(observation)).detach().numpy()))
           
            observation, reward, done, _ = env.step(action)
            step_counter +=1
            time.sleep(.02)


            if done:
                break

        reward_history[i] = step_counter
        print(step_counter,"steps")

    print('Final average reward is', np.mean(reward_history))

if __name__ == '__main__':
    # wandb.init(project = 'dqn_cartpole')
    do_testing('torch_dqn_perfect')


    # do_training('torch_dqn')
    plt.show()


