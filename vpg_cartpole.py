# Created by Jeremiah Coholich 5/26/2020
# Cart-pole env source code: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
import gym
import sys
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import GradientTape
import tensorflow as tf

tf.random.set_seed(2020)
np.random.seed(2022)



class VPG:

    def __init__(self):

        # some parameters
        self.n_samples = 5  # number of trajectory samples to take at each step of the algorithm
        self.n_training_steps = 100  # number of episodes to train for
        self.gamma = 0.99  # from spinningup vpg example
        self.lam = 0.95  # from spinningup vpg example
        self.sga_learning_rate = 0.001

        # create policy network: inputs are state and outputs are logits (log probabilities of actions)
        self.policy_network = Sequential()
        self.policy_network.add(Dense(32, input_shape=(4,), activation='relu'))
        self.policy_network.add(Dense(2, activation='relu'))

        # create value network: inputs are state and outputs are scalar value
        self.value_network = Sequential()
        self.value_network.add(Dense(32, input_dim=4, activation='relu'))
        self.value_network.add(Dense(1, activation='relu'))
        self.value_network.compile(optimizer=tf.keras.optimizers.Adam(),  # so the faster my learning rate, the faster my error increases... But it seems like it increases then decreases
                                   loss='mean_squared_error',
                                   metrics=['mean_squared_error'])

        # define the buffers
        self.state_buffer = -np.ones((self.n_samples, 500, 4))
        self.action_buffer = -np.ones((self.n_samples, 500), dtype='int8')
        self.reward_buffer = -np.ones((self.n_samples, 500), dtype='int8')
        self.rewards_to_go = -np.ones((self.n_samples, 500))
        self.advantages = -np.ones((self.n_samples, 500))
        self.values = -np.ones((self.n_samples, 500))
        self.deltas = -np.ones((self.n_samples, 499))
        self.grad_log_pi = [[-1 for _ in range(500)] for _ in range(self.n_samples)]

        # to be used later...
        self.indicies = -np.ones(self.n_samples)  # this will end up being a list of 5 indicies that correspond to the first non-trajectory element

    def get_action(self, state):  # I'm positive there is a way to do this is one line with a built in function lol
        logits = self.policy_network(np.expand_dims(state, 0))
        dist = np.exp(logits[0])/np.sum(np.exp(logits[0]))
        dist = dist if (dist != [1, 1]).any() else [0.5, 0.5]
        # print('dist',dist)
        action = list(np.random.multinomial(1, dist))
        return action.index(1)

    def add_to_buffers(self, state, action, reward, j, k):  # j is the index of the trajectory, k is the index of the timestep
        self.state_buffer[j, k] = state
        self.action_buffer[j, k] = action
        self.reward_buffer[j, k] = reward

    def compute_rewards_to_go(self):
        self.indicies = np.argmax(self.reward_buffer == -1, axis=1)
        for i in range(self.n_samples):
            for j in range(self.indicies[i]):
                self.rewards_to_go[i, j] = sum(self.reward_buffer[i, j:self.indicies[i]])

    def estimate_advantage(self):  # The advantage needs to be estimated for every state-action pair from every trajectory
        # First find the value of every state by doing a fwd pass through the value network
        for i in range(self.n_samples):
            for j in range(self.indicies[i]):
                self.values[i, j] = self.value_network(np.expand_dims(self.state_buffer[i, j], 0))

        # Then calculate delta^{V} for every timestep (except for the last one)
        for i in range(self.n_samples):
            for j in range(self.indicies[i]-1):
                self.deltas[i, j] = self.reward_buffer[i, j] + self.gamma*self.values[i, j+1] - self.values[i, j]

        # Now loop through every state of each trajectory and calculate the advantage as a sum
        for i in range(self.n_samples):
            for j in range(self.indicies[i]):
                self.advantages[i, j] = 0
                # T = self.indicies[i] -1
                # t = j
                for k in range(self.indicies[i]-j):
                    self.advantages[i, j] += self.deltas[i, j+k]*(1-self.lam**(self.indicies[i]-k-j-1))*(self.gamma*self.lam)**k

        print('advantages', self.advantages[0, :self.indicies[0]])
        print('actions')

    def estimate_policy_grad(self):  # need to find gradient of policy network.
        total_weight_gradient = [0, 0, 0, 0]  # TODO make this general to any architecture
        for i in range(self.n_samples):  # loop through all trajectories
            for j in range(self.indicies[i]):  # loop through all timesteps #TODO make sure I'm not including an extra step at the end of this

                theta = self.policy_network.trainable_weights
                with GradientTape() as tape:
                    action_logits = self.policy_network(np.expand_dims(self.state_buffer[i, j], 0))
                    pi_of_at = action_logits[0, int(self.action_buffer[i, j])]

                temp = tape.gradient(pi_of_at, theta)
                self.grad_log_pi[i][j] = temp  # confirmed correct
                for k in range(len(temp)):
                    total_weight_gradient[k] += temp[k] * self.advantages[i, j]


        for k in range(len(temp)):
            total_weight_gradient[k] = total_weight_gradient[k]/float(self.n_samples)
        return total_weight_gradient


    def stochastic_grad_ascent_step(self, policy_grad):
        current_weights = self.policy_network.get_weights()
        new_weights = [0, 0, 0, 0]
        for i in range(len(policy_grad)):
            new_weights[i] = current_weights[i] + self.sga_learning_rate*policy_grad[i]

        self.policy_network.set_weights(new_weights)  # TODO understand get_weights and set_weights

    def update_value_network(self):
        # concatenate all action value pairs from different episodes
        Xdata = self.state_buffer[0, :self.indicies[0]]
        ydata = self.rewards_to_go[0, :self.indicies[0]]
        for i in range(1, self.n_samples):
            np.append(Xdata, self.state_buffer[i, :self.indicies[i]])
            np.append(ydata, self.rewards_to_go[i, :self.indicies[i]])

        self.value_network.fit(x=Xdata, y=ydata)  # just using rewards-to-go for value assumes no discounting I think

    def visualize_prediction_landscape(self):  # do a grid search over the 4-d state space.
        resolution = [4, 4, 10, 10]  # cart position/velocity, pole angle/tip velocity
        limits = [0.6, 1.0, 0.3, 1.0]
        # states =
        action_predictions = np.empty(tuple(resolution))
        for i in range(resolution(0)):
            for j in range(resolution(1)):
                for k in range(resolution(2)):
                    for l in range(resolution(3)):
                        indicies = [i, j, k, l]
                        state = [-(resolution[m] - indicies[m])*resolution[m] + (1-resolution[m] + indicies[m])*resolution[m] for m in range(4)]
                        action_predictions[i, j, k, l] = self.policy_network(state)


if __name__ == '__main__':

    vpg = VPG()
    env = gym.make('CartPole-v1')
    env.seed(2020)
    for i in range(vpg.n_training_steps):

        for j in range(vpg.n_samples):  # sample trajectories from the environment given the current policy

            state = env.reset()
            for k in range(500):  # carry out a single trajectory
                env.render()

                action = vpg.get_action(state)
                next_state, reward, done, _ = env.step(action)
                vpg.add_to_buffers(state, action, reward, j, k)  # jth trajectory, kth timestep
                state = next_state
                if done:
                    break
            print('Lasted %d steps' % k)

        vpg.compute_rewards_to_go()
        vpg.estimate_advantage()
        policy_grad = vpg.estimate_policy_grad()
        vpg.stochastic_grad_ascent_step(policy_grad)
        vpg.update_value_network()



