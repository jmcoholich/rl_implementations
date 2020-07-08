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

seed = 69420 # seed 111 is randomly successful
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
writer = SummaryWriter('vpg_final_with_lr_decay/' + str(time.time()))



class VPG:
    
    def __init__(self):

        # some parameters
        self.n_samples  = 300 # number of trajectory samples to take at each step of the algorithm (Zsolt says 40 seems low TODO: Check examples, code seems slow per iteration)
        self.n_training_steps = 100 # number of parameter updates
        # self.gamma = 0.99 # from spinningup vpg example 
        # self.lam = 0.95 # from spinningup vpg example
        self.sga_learning_rate = .001 #1E-5 pretty low (usually .01 .001) coarse grain search first  
        self.grad_clip = 100
        self.value_net_training_epochs = 1

        #create policy network: inputs are state and outputs are probabilities 
        self.policy_network = nn.Sequential(
            nn.Linear(4,32),
            nn.ReLU(),
            nn.Linear(32,2),
            nn.Softmax(),
        )
        self.policy_network = self.policy_network.double()

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
        # self.advantages    = -np.ones((self.n_samples,500))
        self.values        = -np.ones((self.n_samples,500))
        # self.deltas        = -np.ones((self.n_samples,499))
        # self.grad_log_pi   = [[-1 for _ in range(500)] for _ in range(self.n_samples)]
   
        #to be used later...
        self.indicies = -np.ones(self.n_samples) #this will end up being a list of 5 indicies that correspond to the first non-trajectory element

    def get_action(self,state): #I'm positive there is a way to do this is one line with a built in function lol
            dist = self.policy_network(torch.from_numpy(state)).detach().numpy()

            # dist = [-1,-1]
            # dist[0] = (1 + np.exp(logits[1] - logits[0]))**-1
            # dist[1] = 1 - dist[0]
            # if np.isnan(dist).any(): 
            #     print('encountered nan')
            #     sys.exit()
            #     # idx = np.argmax(dist)
            #     # dist = np.zeros(dist.shape)
            #     # dist[idx] = 1

            action = list(np.random.multinomial(1,dist))
            # print('action',action)
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

    def estimate_advantage(self): # The advantage needs to be estimated for every state-action pair from every trajectory
        pass
        # First find the value of every state by doing a fwd pass through the value network
        for i in range(self.n_samples):
            for j in range(self.indicies[i]):
                self.values[i,j] = self.value_network(torch.from_numpy(self.state_buffer[i,j]))

        # Then calculate delta^{V} for every timestep (except for the last one)
        for i in range(self.n_samples):
            for j in range(self.indicies[i]-1):
                self.deltas[i,j] = self.reward_buffer[i,j] + self.gamma*self.values[i,j+1] - self.values[i,j]

        # Now loop through every state of each trajectory and calculate the advantage as a sum 
        for i in range(self.n_samples):
            for j in range(self.indicies[i]): 
                self.advantages[i,j] = 0
                # T = self.indicies[i] -1
                # t = j
                for k in range(self.indicies[i]-j):
                    self.advantages[i,j] += self.deltas[i,j+k]*(1-self.lam**(self.indicies[i]-k-j-1))*(self.gamma*self.lam)**k

        # print('advantages',self.advantages[0,:self.indicies[0]])
        # print('actions')

    def estimate_policy_grad(self): #need to find gradient of policy network.
        #loop through all  trajectories collected

        total_gradient = [0,0,0,0] #TODO make this general to any architechture 
        # average_value = 0
        # average_rewards_to_go = 0
        # total_log_prob_gradient = [0,0,0,0]

        for i in range(self.n_samples):
            # average_rewards_to_go += np.mean(self.rewards_to_go[i,:self.indicies[i]]) 
            for j in range(self.indicies[i]): 

                # self.policy_network_optimizer.zero_grad()
                self.policy_network.zero_grad()
                input_state = self.state_buffer[i,j]
                dist = self.policy_network(torch.from_numpy(input_state))
                action_taken_log_probability = torch.log(dist[self.action_buffer[i,j]])
                action_taken_log_probability.backward()
                params = list(self.policy_network.parameters())


                value = self.value_network(torch.from_numpy(input_state)).detach().numpy()
                # average_value += value
                for k in range(len(total_gradient)):
                    # *** Changed from self.advantages to self.rewards_to_go - value****** 
                    total_gradient[k] += params[k].grad * (self.rewards_to_go[i,j] - value) 
                    # total_log_prob_gradient[k] = params[k].grad
                    # if i == self.n_samples-1: 
                        # print('SINGLE STEP LOG_PROB_GRAD',total_log_prob_gradient[k])


        for k in range(len(total_gradient)):
            total_gradient[k] = np.clip(total_gradient[k]/float(self.n_samples), -self.grad_clip, self.grad_clip)
            # total_log_prob_gradient[k] = total_log_prob_gradient[k]/float(self.n_samples)

        print(self.indicies)
        # average_value = average_value/float(np.sum(self.indicies))
        # average_rewards_to_go = average_rewards_to_go/float(self.n_samples)
        return total_gradient # , average_rewards_to_go, total_log_prob_gradient




        # total_weight_gradient = [0,0,0,0] #TODO make this general to any architecture
        # for i in range(self.n_samples):#loop through all trajectories 
        #     for j in range(self.indicies[i]): #loop through all timesteps #TODO make sure I'm not including an extra step at the end of this

        #         theta = self.policy_network.trainable_weights
        #         with GradientTape() as tape: 
        #             action_logits = self.policy_network(np.expand_dims(self.state_buffer[i,j],0))
        #             pi_of_at = action_logits[0,int(self.action_buffer[i,j])]

        #         temp = tape.gradient(pi_of_at, theta)
        #         self.grad_log_pi[i][j] = temp #confirmed correct
        #         for k in range(len(temp)):
        #             total_weight_gradient[k] += temp[k] * self.advantages[i,j]


        # for k in range(len(temp)):
        #     total_weight_gradient[k] = total_weight_gradient[k]/float(self.n_samples)
        # return total_weight_gradient


    def stochastic_grad_ascent_step(self, policy_grad):
        # current_weights = self.policy_network.get_weights()
        # new_weights = [0,0,0,0]
        # for i in range(len(policy_grad)):   
        #     new_weights[i] = current_weights[i] + self.sga_learning_rate*policy_grad[i]

        # self.policy_network.set_weights(new_weights) #TODO understand get_weights and set_weights


        weights_dict = self.policy_network.state_dict()
        keys = ['0.weight','0.bias','2.weight','2.bias']

        if np.mean(self.indicies) > 350: #introduce very rough lr decay
            lr = self.sga_learning_rate/5.
        else:
            lr = self.sga_learning_rate

        for i in range(len(keys)):
            weights_dict[keys[i]] += lr * policy_grad[i]

        self.policy_network.load_state_dict(weights_dict)
        return



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


    def visualize_prediction_landscape(self): #do a grid search over the 4-d state space. 
        pass
    #     resolution = [4,4,10,10]#cart position/velocity, pole angle/tip velocity
    #     limits = [0.6,1.0,0.3,1.0]
    #     states = 
    #     action_predictions = np.empty(tuple(resolution))
    #     for i in range(resolution(0)):
    #         for j in range(resolution(1)):
    #             for k in range(resolution(2)):
    #                 for l in range(resolution(3)):
    #                     indicies = [i,j,k,l]
    #                     state = [-(resolution[m] - indicies[m])*resolution[m] + (1-resolution[m] + indicies[m])*resolution[m]  for m in range(4)]
    #                     action_predictions[i,j,k,l] = self.policy_network(state)


if __name__ == '__main__':
    start_time = time.clock()
    writer_idx = 0
    vpg = VPG()
    env = gym.make('CartPole-v1')
    env.seed(69420)

    for i in range(vpg.n_training_steps):
        logits_differences_sum = 0

        for j in range(vpg.n_samples): # sample trajectories from the environment given the current policy

            state = env.reset()
            for k in range(500): # carry out a single trajectory
                if (i%10 == 0) and (j%30 == 0): 
                    env.render()

                action = vpg.get_action(state)
                # logits_differences_sum += np.absolute(logits[0] - logits[1])
                # print('Action',action)
                next_state, reward, done, _ = env.step(action)
                if (i%10 == 0) and (j%30 == 0):
                    print('\nstate',state,'\naction',action,'\nreward',reward,'\ndone',done,'\nnext_state',next_state)
                vpg.add_to_buffers(state,action,reward, j, k) #jth trajectory, kth timestep
                state = next_state
                if done:
                    break
            # print('Lasted %d steps' %k)

        vpg.compute_rewards_to_go()
        # vpg.estimate_advantage()
        policy_grad = vpg.estimate_policy_grad()
        # print('\naverage log prob grad',total_log_prob_gradient)
        print('\nfinal policy grad',policy_grad)
        vpg.stochastic_grad_ascent_step(policy_grad)
        loss = vpg.update_value_network()

        avg_reward = np.mean(vpg.indicies)
        print('Finished training iteration', writer_idx,'**********************************************************************************************')
        # print('Average Reward', avg_reward)
        # print()
        # writer.add_scalar('Average difference in logits (policy net output)', logits_differences_sum/float(vpg.n_samples + np.sum(vpg.indicies)), writer_idx)
        writer.add_scalar('Average Reward with current policy', avg_reward, writer_idx)
        # writer.add_scalar('Average value function output', average_value, writer_idx)
        writer.add_scalar('Value function loss',loss, writer_idx)
        writer.add_scalar('Average Action from current policy', vpg.get_average_actions(), writer_idx)
        print('Training time so far: ', str(datetime.timedelta(seconds=(time.clock() - start_time))))
        # writer.add_scalar('Average rewards-to-go', average_rewards_to_go, writer_idx)

        writer_idx +=1 

        vpg.reset_reward_buffer()





