# Jeremiah Coholich 4/19/2020 
# MPC for OpenAI Gym cartpole-v1

import gym
import time
import numpy as np
import math
import itertools
from operator import itemgetter
import time
import sys
import matplotlib.pyplot as plt
from matplotlib import rc
import os

render_with_latex = False

if render_with_latex:
    PATHFLAG = 1
    # Add Latex to path (Otherwise, render using python)
    if os.path.isdir("/usr/local/texlive/2019/bin/x86_64-darwin") and PATHFLAG==1:
        os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2019/bin/x86_64-darwin'
        # Set Render &  Compile Settings
        rc('text', usetex=True)
        rc('font', family='serif')
        PATHFLAG = 0
    elif os.path.isdir("/usr/share/texlive/texmf-dist") and PATHFLAG==1:
        os.environ["PATH"] += os.pathsep + '/usr/share/texlive/texmf-dist'
        # Set Render &  Compile Settings
        rc('text', usetex=True)
        rc('font', family='serif')
        PATHFLAG = 0
    else:
        print('This code uses latex to render plots. Please add your latex installation path to python.')


class MPC:
    def __init__(self,env):
        self.horizon = 10 #number of timesteps
        self.replanning_period = 10
        self.current_trajectory         = np.ones(self.horizon) * -1.
        self.steps_since_last_replanning = self.replanning_period

        self.gravity =      env.gravity
        self.masscart =     env.masscart 
        self.masspole =     env.masspole
        self.total_mass = (env.masspole + env.masscart)
        self.length =       env.length # actually half the pole's length
        self.polemass_length = (env.masspole * env.length)
        self.force_mag =    env.force_mag
        self.tau =          env.tau # seconds between state updates

        self.max_values = [2.4, 2.4, 0.20944, 0.20944]#the max values of each state before the simulation terminates -- used to normalize state vectors before evaluation
        self.max_values = [5, 10, 1, 1000]#the max values of each state before the simulation terminates -- used to normalize state vectors before evaluation


    def get_action(self, current_state):
        if self.steps_since_last_replanning == self.replanning_period:
            self.replan(current_state)
        elif self.steps_since_last_replanning > self.replanning_period:
            print("ERROR: Steps since last replanning is greater than replanning period, which should never happen!")
            return

        action = self.current_trajectory[self.steps_since_last_replanning]

        self.steps_since_last_replanning +=1
        return action


    def replan(self,current_state):

        candidate_trajectories = list(map(list,itertools.product([0,1],repeat = self.horizon)))
        final_state_norms = [i for i in range(2**self.horizon)]

        #generate the result of following every trajectory
        i = 0
        while i != len(candidate_trajectories):
            trajectory = candidate_trajectories[i]
            state = current_state

            for j in range(len(trajectory)):
                state, failed = self.cp_dynamics(state, trajectory[j])
                # This stuff was added to remove infeasible trajectories, but searching the lists actually took more time than just evaluating all trajectories
                # if failed:
                #     candidate_trajectories.pop(i)#remove the failed trajectory from list of candidates
                #     final_state_norms.pop()
                #     if j != (len(trajectory) -1): #if this is NOT the last step, remove all other trajectories with the same sequence up to failure
                #         failure_sequence = trajectory[0:j+1] 
                #         k = i #not i+1, because we already removed the current trajectory that failed, so the next one is in its place
                #         while k != len(candidate_trajectories): #loop through remaining trajectories 
                #             if candidate_trajectories[k][0:j+1] == failure_sequence:
                #                 candidate_trajectories.pop(k)
                #                 final_state_norms.pop()
                #             else:
                #                 k +=1
                #     break #stop evaluating this trajectory
            # if not failed:
            #     final_state_norms[i] = np.linalg.norm(np.divide(state,self.max_values)) #don't update this if we have just failed and removed the trajectory
            #     i += 1
            final_state_norms[i] = np.linalg.norm(np.divide(state,self.max_values)) #don't update this if we have just failed and removed the trajectory
            i += 1

        self.current_trajectory = candidate_trajectories[min(enumerate(final_state_norms), key=itemgetter(1))[0]] 
        self.steps_since_last_replanning = 0



    def cp_dynamics(self,state, action):
        # copied from cart-pole source code


        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action==1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta* temp) / (self.length * (4.0/3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x  = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        new_state = np.array([x, x_dot, theta, theta_dot])

        if (abs(x) > 2.4) or (abs(theta) > 0.20944):
            failed = True
        else:
            failed = False

        return new_state, failed


def feedback_controller(observation):
    # Position and velocity feedback from the pendulum position with experimental 1.4 factor
    if observation[2] + 1.4*observation[3]<= 0:
        action = 0
    else:
        action = 1 
    return action


def main():

    env = gym.make('CartPole-v1')
    mpc = MPC(env)
    
    total_episodes = 100;
    rew = np.zeros((total_episodes,))
    done_mem = np.zeros((total_episodes,)) 
    for i_episode in range(total_episodes):
        observation = env.reset()

        if i_episode == (total_episodes -1): #if this is the last episode, record state history
            xpos = np.ones(500) 
            xvel = np.ones(500) 
            angle = np.ones(500) 
            anglevel = np.ones(500) 

            xpos[0] = observation[0]
            xvel[0] = observation[1]
            angle[0] = observation[2]
            anglevel[0] = observation[3]

        for t in range(600):

            # env.render()

            # action = feedback_controller(observation)
            action = mpc.get_action(observation)

            observation, reward, done, info = env.step(action) # info is empty for the cartpole

            if i_episode == (total_episodes -1): #if this is the last episode, record state history
                xpos[t] = observation[0] 
                xvel[t] = observation[1] 
                angle[t] = observation[2] 
                anglevel[t] = observation[3] 

            if done:
                print('Episode finished after {} timesteps'.format(t+1)) #In  this case the total reward is the same as the total timesteps survived.

                rew[i_episode] = t+1

                if i_episode == (total_episodes-1) and t != 499: #truncate the state arrays
                    xpos =  xpos[0:t+1] 
                    xvel =  xvel[0:t+1] 
                    angle = angle[0:t+1] 
                    anglevel = anglevel[0:t+1] 

                #find the reason that it terminated
                if t > 195:
                    done_mem[i_episode] = 4
                elif observation[0] < -env.x_threshold:
                    done_mem[i_episode] = 0
                elif observation[0] > env.x_threshold:
                    done_mem[i_episode] = 1 
                elif observation[2] < -env.theta_threshold_radians:
                    done_mem[i_episode] = 2
                elif observation[2] > env.theta_threshold_radians:
                    done_mem[i_episode] = 3

                break

    env.close()



    # Count number of Reasons for "Done"
    bins = np.zeros((5,))
    bins[0] = np.count_nonzero(done_mem == 0)
    bins[1] = np.count_nonzero(done_mem == 1)
    bins[2] = np.count_nonzero(done_mem == 2)
    bins[3] = np.count_nonzero(done_mem == 3)
    bins[4] = np.count_nonzero(done_mem == 4)

    #Plotting (copied from Ceasar's notebook)
    # Plot & Save State
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax1.plot(xpos)
    ax1.set_xlabel(r'time (s)')
    ax1.set_ylabel(r'x-position (m)')
    ax1.grid(True)

    ax2 = fig.add_subplot(2,2,2)
    ax2.plot(xvel)
    ax2.set_xlabel(r'time (s)')
    ax2.set_ylabel(r'velocity (m/s)')
    ax2.grid(True)

    # angle = (angle/(2*np.pi) - np.floor(angle/(2*np.pi)))*2*np.pi
    ax3 = fig.add_subplot(2,2,3)
    ax3.plot(angle)
    ax3.set_xlabel(r'time (s)')
    ax3.set_ylabel(r'angle (rad)')
    ax3.grid(True)

    ax4 = fig.add_subplot(2,2,4)
    ax4.plot(anglevel)
    ax4.set_xlabel(r'time (s)')
    ax4.set_ylabel(r'angle (rad/s)')
    ax4.grid(True)

    fig.tight_layout(pad=3.0)
    fig.show()

    if render_with_latex:
        fig.savefig('./plots/basic_anglexpos.pdf')

    # Plot & Save State Histogram
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1 ,1)
    ax1.hist(rew, color=(12/256, 123/256, 220/256), edgecolor='black', alpha=.5)
    ax1.axvline(rew.mean(), color='r', linestyle='dashed', linewidth=1, alpha=.75)

    min_ylim, max_ylim = ax1.get_ylim()
    ax1.text(rew.mean()*1.1, max_ylim*0.9, 'Mean Reward: {:.2f}'.format(rew.mean()))

    ax1.set_xlabel(r'Reward')
    ax1.set_ylabel(r'Frequency')
    ax1.set_title(r'Histogram of Rewards over ' + str(total_episodes) + ' episodes')

    if render_with_latex:
        fig.savefig('./plots/basic_histogram.pdf')
    # Plot & Save Done Histogram
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1 ,1)
    ax1.bar(np.arange(5), bins, color=(12/256, 123/256, 220/256), edgecolor='black', alpha=.5)
    ax1.set_xlabel(r'Exit Reason')
    ax1.set_ylabel(r'Frequency')
    ax1.set_xticks(np.arange(5))
    ax1.set_xticklabels(['x-pos. neg.','x-pos. pos.', 'angle. neg.', 'angle pos.', 'reward exceed'])
    if render_with_latex:
        fig.savefig('./plots/basic_histogram_done_memory.pdf')


    plt.show()

main()





