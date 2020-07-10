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



class PPO:
    
    def __init__(self):

        # some parameters
        self.n_samples  = 100 # number of trajectory samples to take at each step of the algorithm (Zsolt says 40 seems low TODO: Check examples, code seems slow per iteration)
        self.traj_length = 200
        self.n_training_steps = 100 # number of parameter updates
        self.value_net_training_epochs = 10
        self.policy_update_epochs = 10
        self.epsilon = 0.2
        self.discount_factor = 0.99

        #create policy network: inputs are state and outputs are probabilities 
        self.policy_network = PolicyNetContinuous()
        self.policy_network = self.policy_network.double()
        self.policy_network_optimizer = torch.optim.Adam(self.policy_network.parameters())

        #create value network: inputs are state and outputs are scalar value
        self.value_network = nn.Sequential(
            nn.Linear(3,64),
            nn.ReLU(),
            nn.Linear(64,256),
            nn.ReLU(),
            nn.Linear(256,1)
        )
        self.value_network = self.value_network.double()
        self.value_network_optimizer = torch.optim.Adam(self.value_network.parameters())
        self.value_network_criterion = torch.nn.MSELoss()


        #define the buffers
        self.state_buffer  = -np.ones((self.n_samples, self.traj_length,3))
        self.action_buffer = -np.ones((self.n_samples, self.traj_length))
        self.reward_buffer = -np.ones((self.n_samples, self.traj_length + 1))
        self.rewards_to_go = -np.ones((self.n_samples, self.traj_length))
        self.advantages    = -np.ones((self.n_samples, self.traj_length))
        self.values        = -np.ones((self.n_samples, self.traj_length))
        self.log_pi_th_old     = -np.ones((self.n_samples, self.traj_length)) # used in maximizing the surrogate objective function
   
        # self.indicies = -np.ones(self.n_samples) #this will end up being a list of 5 indicies that correspond to the first non-trajectory element

    def get_action(self,state): #I'm positive there is a way to do this is one line with a built in function lol
        return self.policy_network.get_action(state)


    def add_to_buffers(self,state,action,reward,j,k): #j is the index of the trajectory, k is the index of the timestep
        self.state_buffer[j,k]  = state
        self.action_buffer[j,k] = action
        self.reward_buffer[j,k] = reward


    def compute_rewards_to_go(self):
        # self.indicies = np.argmax(self.reward_buffer == -1, axis = 1)
        for i in range(self.n_samples):
            for j in range(self.traj_length):
                self.rewards_to_go[i,j] = self.reward_buffer[i,-1]
                for k in range(self.traj_length-1, j-1, -1):
                    self.rewards_to_go[i,j] = self.rewards_to_go[i,j]*self.discount_factor + self.reward_buffer[i,k]


    def update_policy(self):
        for i in range(self.policy_update_epochs):
            ts1 = time.time()
            self.policy_network_optimizer.zero_grad()
            ts2 = time.time()
            objective = -self.ppo_clip_objective()
            ts3 = time.time()
            objective.backward()
            ts4 = time.time()
            self.policy_network_optimizer.step()
            ts5 = time.time()

            print('\t\t\tIter %d total time:' %i, str(datetime.timedelta(seconds=(ts5 - ts1))))
            print('\t\t\t\tzero grad time:', str(datetime.timedelta(seconds=(ts2 - ts1))))
            print('\t\t\t\tcalc obj time:', str(datetime.timedelta(seconds=(ts3 - ts2))))
            print('\t\t\t\tbackwards pass time:', str(datetime.timedelta(seconds=(ts4 - ts3))))
            print('\t\t\t\tAdam step time:', str(datetime.timedelta(seconds=(ts5 - ts4))))


    def ppo_clip_objective(self):
        objective = 0
        for i in range(self.n_samples):
            for j in range(self.traj_length):
                mu, stdev = self.policy_network.forward(torch.from_numpy(self.state_buffer[i,j]))
                n = torch.distributions.Normal(mu, stdev)
                log_pi_th = n.log_prob(self.action_buffer[i,j])
                advantage = self.advantages[i,j]
                g = self.g(self.epsilon, advantage)
                objective += min(torch.exp(log_pi_th - self.log_pi_th_old[i,j]) * advantage, g)

        return objective/float(self.n_samples * self.traj_length)


    def g(self, epsilon, advantage):
        if advantage >= 0: 
            return (1+epsilon)*advantage
        return (1-epsilon)*advantage


    def calculate_log_pi_th_old(self):
        with torch.no_grad():
            for i in range(self.n_samples):
                for j in range(self.traj_length):
                    mu, stdev = self.policy_network.forward(torch.from_numpy(self.state_buffer[i,j]))
                    n = torch.distributions.Normal(mu, stdev)
                    self.log_pi_th_old[i,j] = n.log_prob(self.action_buffer[i,j])

    def calculate_advantages(self):
        with torch.no_grad():
            for i in range(self.n_samples):
                for j in range(self.traj_length):
                    self.advantages[i,j] = self.rewards_to_go[i,j] - self.value_network(torch.from_numpy(self.state_buffer[i,j]))


    def update_value_network(self):

        # concatenate all action value pairs from different episodes
        Xdata = self.state_buffer[0]
        ydata = self.rewards_to_go[0]
        for i in range(1,self.n_samples):
            np.append(Xdata,self.state_buffer[i])
            np.append(ydata,self.rewards_to_go[i])

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
        actions = self.action_buffer[0]
        for i in range(1,self.n_samples):
            np.append(actions,self.action_buffer[i])
        return np.mean(actions)

    def complete_buffer(self, next_state, i):
        with torch.no_grad():
            next_state_value = self.value_network(torch.from_numpy(next_state)) #this will be reward number 201
            self.reward_buffer[i,-1] = next_state_value



class PolicyNetContinuous(nn.Module):

    def __init__(self):
        super(PolicyNetContinuous, self).__init__()
        self.linear1 = nn.Linear(3,64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64,256)
        self.linear_mu = nn.Linear(256,1)
        self.linear_stdev = nn.Linear(256,1)
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        mu = 2 * self.tanh(self.linear_mu(x))
        stdev = self.softplus(self.linear_stdev(x)) + 1e-3
        return mu, stdev

    def get_action(self, state):
        with torch.no_grad():
            mu, stdev = self.forward(state)
            n = torch.distributions.Normal(mu, stdev)
            action = n.sample()
        return np.clip(action.item(), -2, 2)


def do_testing(saved_net_path):
    env = gym.make('Pendulum-v0')
    model = torch.load(saved_net_path)

    n_traj = 10
    traj_len = 200
    rewards = np.empty((n_traj,traj_len))

    for i in range(n_traj):
        state = env.reset()
        for j in range(traj_len):
            env.render()

            with torch.no_grad():
                mu, stdev = model(torch.from_numpy(state))
                n = torch.distributions.Normal(mu, stdev)
                action = n.sample()
                action = np.clip(action.item(), -2, 2)

            next_state, reward, done, _ = env.step([action])
            rewards[i,j] = reward
            state = next_state
            if done:
                break

        print('Average reward per timestep for episode: ', np.mean(rewards[i]))
    print('Total average reward for policy over all episodes: ',np.mean(rewards))

def do_training():
    writer = SummaryWriter('ppo_cont/' + str(time.time()))
    start_time = time.time()
    ppo = PPO()
    env = gym.make('Pendulum-v0')
    env.seed(69420)

    for i in range(ppo.n_training_steps):
        start_iter_time = time.time()
        for j in range(ppo.n_samples): # sample trajectories from the environment given the current policy

            state = env.reset()
            for k in range(ppo.traj_length): # carry out a single trajectory

                action = ppo.get_action(torch.from_numpy(state))

                next_state, reward, done, _ = env.step([action])
                ppo.add_to_buffers(state,action,reward, j, k) #jth trajectory, kth timestep
                state = next_state
                if done:
                    ppo.complete_buffer(next_state,j) 
                    break

        end_sampling_time = time.time()
        ppo.compute_rewards_to_go()
        r2g_time = time.time()
        ppo.calculate_advantages()
        calc_adv_time = time.time()
        ppo.calculate_log_pi_th_old()
        calc_log_pi_th_old = time.time()
        ppo.update_policy()
        update_pol_time = time.time()
        loss = ppo.update_value_network()
        update_value_net_time = time.time()
        avg_reward = np.mean(ppo.reward_buffer) # max reward is 
        torch.save(ppo.policy_network, 'ppo_continuous_policy_net')

        end_iter_time = time.time()

        #logging
        print('Finished training iteration', i,'**********************************************************************************************')
        writer.add_scalar('Average reward at every timestep', avg_reward, i)
        writer.add_scalar('Value function loss',loss, i)
        writer.add_scalar('Iteration_time (s)', end_iter_time - start_iter_time, i)
        writer.add_scalar('Total time(s)',end_iter_time - start_time,i)
        writer.add_scalar('Average Action from current policy', ppo.get_average_actions(), i)

        print('Training time so far: ', str(datetime.timedelta(seconds=(end_iter_time - start_time))))
        print('\tIter time: ',str(datetime.timedelta(seconds=(end_iter_time - start_iter_time))))
        print('\t\tTrajectory sampling time: ', str(datetime.timedelta(seconds=(end_sampling_time - start_iter_time))))
        print('\t\tCompute rewards-to-go time: ', str(datetime.timedelta(seconds=(r2g_time - end_sampling_time))))
        print('\t\tCalc Advantages time: ', str(datetime.timedelta(seconds=(calc_adv_time - r2g_time))))
        print('\t\tcalculate log_pi_th_old time: ', str(datetime.timedelta(seconds=(calc_log_pi_th_old - calc_adv_time))))
        print('\t\tupdate policy time: ', str(datetime.timedelta(seconds=(update_pol_time - calc_log_pi_th_old))))
        print('\t\tupdate value net time: ', str(datetime.timedelta(seconds=(update_value_net_time - update_pol_time))))
        print('\t\tcalc avg rewards time ', str(datetime.timedelta(seconds=(end_iter_time - update_value_net_time))))
        print()



if __name__ == '__main__':
    # do_testing('ppo_continuous_policy_net')
    do_training()

