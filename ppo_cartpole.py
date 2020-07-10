# Created by Jeremiah Coholich 7/08/2020
# Cart-pole env source code: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
import gym
import sys
import numpy as np 
from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter
import random
import time
import datetime

seed = 2020
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
writer = SummaryWriter('delete/' + str(time.time()))



class PPO:
    
    def __init__(self):

        # some parameters
        self.n_samples  = 300 # number of trajectory samples to take at each step of the algorithm (Zsolt says 40 seems low TODO: Check examples, code seems slow per iteration)
        self.n_training_steps = 100 # number of parameter updates
        self.value_net_training_epochs = 1
        self.policy_update_epochs = 15
        self.epsilon = 0.2

        #create policy network: inputs are state and outputs are probabilities 
        self.policy_network = nn.Sequential(
            nn.Linear(4,32),
            nn.ReLU(),
            nn.Linear(32,2),
            nn.Softmax(),
        )
        self.policy_network = self.policy_network.double()
        self.policy_network_optimizer = torch.optim.Adam(self.policy_network.parameters())

        #create value network: inputs are state and outputs are scalar value
        self.value_network = nn.Sequential(
            nn.Linear(4,32),
            nn.ReLU(),
            nn.Linear(32,1),
            nn.ReLU()
        )
        self.value_network = self.value_network.double()
        self.value_network_optimizer = torch.optim.Adam(self.value_network.parameters())
        self.value_network_criterion = torch.nn.MSELoss()


        #define the buffers
        self.state_buffer  = -np.ones((self.n_samples,500,4))
        self.action_buffer = -np.ones((self.n_samples,500), dtype = 'int8')
        self.reward_buffer = -np.ones((self.n_samples,501), dtype = 'int8')
        self.rewards_to_go = -np.ones((self.n_samples,500))
        self.advantages    = -np.ones((self.n_samples,500))
        self.values        = -np.ones((self.n_samples,500))
        self.pi_th_old     = -np.ones((self.n_samples,500)) # used in maximizing the surrogate objective function
   
        self.indicies = -np.ones(self.n_samples) #this will end up being a list of 5 indicies that correspond to the first non-trajectory element

    def get_action(self,state): #I'm positive there is a way to do this is one line with a built in function lol
            dist = self.policy_network(torch.from_numpy(state)).detach().numpy()
            action = list(np.random.multinomial(1,dist))
            return action.index(1)


    def add_to_buffers(self,state,action,reward,j,k): #j is the index of the trajectory, k is the index of the timestep
        self.state_buffer[j,k]  = state
        self.action_buffer[j,k] = action
        self.reward_buffer[j,k] = reward


    def compute_rewards_to_go(self):
        self.indicies = np.argmax(self.reward_buffer == -1, axis = 1)
        for i in range(self.n_samples):
            for j in range(self.indicies[i]):
                self.rewards_to_go[i,j] = sum(self.reward_buffer[i,j:self.indicies[i]])


    def update_policy(self):
        self.calculate_advantages()
        self.calculate_pi_th_old()
        for i in range(self.policy_update_epochs):
            self.policy_network_optimizer.zero_grad()
            objective = -self.ppo_clip_objective()
            objective.backward()
            self.policy_network_optimizer.step()


    def ppo_clip_objective(self):
        objective = 0
        for i in range(self.n_samples):
            for j in range(self.indicies[i]):
                pi_th = self.policy_network(torch.from_numpy(self.state_buffer[i,j]))[self.action_buffer[i,j]]
                advantage = self.advantages[i,j]
                g = self.g(self.epsilon, advantage)
                objective += min(pi_th*advantage/self.pi_th_old[i,j], g)

        return objective/float((np.sum(self.indicies)))


    def g(self, epsilon, advantage):
        if advantage >= 0: 
            return (1+epsilon)*advantage
        return (1-epsilon)*advantage


    def calculate_pi_th_old(self):
        for i in range(self.n_samples):
            for j in range(self.indicies[i]):
                self.pi_th_old[i,j] = self.policy_network(torch.from_numpy(self.state_buffer[i,j]))[self.action_buffer[i,j]]


    def calculate_advantages(self):
        for i in range(self.n_samples):
            for j in range(self.indicies[i]):
                self.advantages[i,j] = self.rewards_to_go[i,j] - self.value_network(torch.from_numpy(self.state_buffer[i,j]))


    def update_value_network(self):

        # concatenate all action value pairs from different episodes
        Xdata = self.state_buffer[0,:self.indicies[0]]
        ydata = self.rewards_to_go[0,:self.indicies[0]]
        for i in range(1,self.n_samples):
            np.append(Xdata,self.state_buffer[i,:self.indicies[i]])
            np.append(ydata,self.rewards_to_go[i,:self.indicies[i]])

        # train the model
        for i in range(self.value_net_training_epochs):
            self.value_network_optimizer.zero_grad()
            y_pred = self.value_network(torch.from_numpy(Xdata))     
            loss = self.value_network_criterion(y_pred, torch.from_numpy(np.expand_dims(ydata,1)))
            loss.backward()
            self.value_network_optimizer.step()

        return loss


    def reset_reward_buffer(self): # this is done so that self.indicies can be properly recalculated
        self.reward_buffer = -np.ones((self.n_samples,501), dtype = 'int8') 


    def get_average_actions(self):
        actions = self.action_buffer[0,:self.indicies[0]]
        for i in range(1,self.n_samples):
            np.append(actions,self.action_buffer[i,:self.indicies[i]])

        assert (actions != -1).all()

        return np.mean(actions)




if __name__ == '__main__':
    start_time = time.clock()
    writer_idx = 0
    ppo = PPO()
    env = gym.make('CartPole-v1')
    env.seed(69420)

    for i in range(ppo.n_training_steps):
        for j in range(ppo.n_samples): # sample trajectories from the environment given the current policy

            state = env.reset()
            for k in range(500): # carry out a single trajectory

                action = ppo.get_action(state)

                next_state, reward, done, _ = env.step(action)
                ppo.add_to_buffers(state,action,reward, j, k) #jth trajectory, kth timestep
                state = next_state
                if done:
                    break

        ppo.compute_rewards_to_go()
        ppo.update_policy()
        loss = ppo.update_value_network()
        avg_reward = np.mean(ppo.indicies)
        print('Finished training iteration', writer_idx,'**********************************************************************************************')

        writer.add_scalar('Average Reward with current policy', avg_reward, writer_idx)
        writer.add_scalar('Value function loss',loss, writer_idx)
        writer.add_scalar('Average Action from current policy', ppo.get_average_actions(), writer_idx)
        print('Training time so far: ', str(datetime.timedelta(seconds=(time.clock() - start_time))))

        writer_idx +=1 

        ppo.reset_reward_buffer()

