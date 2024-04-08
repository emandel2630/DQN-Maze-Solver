import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import collections
from matplotlib.colors import ListedColormap


Transition = collections.namedtuple('Experience',
                                    field_names=['state', 'action',
                                                 'next_state', 'reward',
                                                 'is_game_on'])


class Agent:
    def __init__(self, maze, memory_buffer, use_softmax = True):
        self.env = maze
        self.buffer = memory_buffer # this is actually a reference
        self.num_act = 4
        self.use_softmax = use_softmax
        self.total_reward = 0
        self.final_total_reward = 0
        self.min_reward = -self.env.maze.size
        self.isgameon = True

        
    def make_a_move(self, net, epsilon, stepNum, device = 'cuda:0', testing=False):
        action = self.select_action(net, epsilon, device)
        current_state = self.env.state()
        next_state, reward, self.isgameon = self.env.state_update(action)
        self.total_reward += reward
        
        if stepNum > 1000:
            self.isgameon = False
        if not self.isgameon:
            self.final_total_reward = self.total_reward
            self.total_reward = 0

        
        transition = Transition(current_state, action,
                                next_state, reward,
                                self.isgameon)
        if testing == False:
            self.buffer.push(transition)

    def make_a_move_doubleQ(self, Q1_net, Q2_net, epsilon, stepNum, device = 'cuda:0', testing=False):
        action = self.select_action_DoubleQ(Q1_net, Q2_net, epsilon, device)
        current_state = self.env.state()
        next_state, reward, self.isgameon = self.env.state_update(action)
        self.total_reward += reward
        
        if stepNum > 1000:
            self.isgameon = False
        if not self.isgameon:
            self.final_total_reward = self.total_reward
            self.total_reward = 0

        
        transition = Transition(current_state, action,
                                next_state, reward,
                                self.isgameon)
        if testing == False:
            self.buffer.push(transition)
            
    def select_action(self, net, epsilon, device = 'cuda:0'):
        pDesiredAction = 0.98
        state = torch.Tensor(self.env.state()).to(device).reshape(1, -1)
        qvalues = net(state).cpu().detach().numpy().squeeze()

        # softmax sampling of the qvalues NOT USED
        if self.use_softmax:
            p = sp.softmax(qvalues/epsilon).squeeze()
            p /= np.sum(p)
            action = np.random.choice(self.num_act, p = p)
            
        # else choose the best action with probability 1-epsilon
        # and with probability epsilon choose at random
        else:
            if np.random.random() < epsilon:
                action = np.random.randint(self.num_act, size=1)[0]
            else:                
                action = np.argmax(qvalues, axis=0)
                action = int(action)
        
        #Slightly stochastic environment
        if np.random.rand() > pDesiredAction:
            undesiredActions = [finalact for finalact in [0,1,2,3] if finalact != action]
            action = np.random.choice(undesiredActions, 1)[0]
        
        return action

    def select_action_DoubleQ(self, Q1_net, Q2_net, epsilon, device='cuda'):
        pDesiredAction = 0.98

        state = torch.Tensor(self.env.state()).to(device).reshape(1, -1)

        # Compute the average Q-values from both networks
        qvalues1 = Q1_net(state).cpu().detach().numpy().squeeze()
        qvalues2 = Q2_net(state).cpu().detach().numpy().squeeze()
        average_qvalues = (qvalues1 + qvalues2) / 2

        # softmax sampling of the qvalues NOT USED
        if self.use_softmax:
            p = sp.softmax(average_qvalues / epsilon).squeeze()
            p /= np.sum(p)
            action = np.random.choice(self.num_act, p=p)

        # else choose the best action with probability 1-epsilon
        # and with probability epsilon choose at random
        else:
            if np.random.random() < epsilon:
                action = np.random.randint(self.num_act, size=1)[0]
            else:
                action = np.argmax(average_qvalues, axis=0)
                action = int(action)

        # Slightly stochastic environment
        if np.random.rand() > pDesiredAction:
            undesiredActions = [finalact for finalact in [0, 1, 2, 3] if finalact != action]
            action = np.random.choice(undesiredActions, 1)[0]

        return action

    
    
    def plot_policy_map(self, net, filename, offset, hyperparams,epochnum,title="DQN"):
        net.eval()  # Set the network to evaluation mode
        with torch.no_grad():
            # Define the new colormap
            cmap = ListedColormap([
                [1, 1, 1],  # white for empty spaces
                [0, 0, 0],  # black for walls
                [0.9290, 0.6940, 0.1250],  # tan for special state 2
                [1, 0, 0],  # red for special state 3
                [0, 0, 1],  # blue for special state 4
                [0, 1, 0]   # green for special state 5
            ])
            
            # Create a figure to display the maze
            plt.figure()
            plt.imshow(self.env.maze, cmap=cmap)
            
            # Loop through the allowed states to display policies
            for free_cell in self.env.allowed_states:
                self.env.current_position = np.asarray(free_cell)
                qvalues = net(torch.Tensor(self.env.state()).reshape(1, -1).to('cuda'))
                action = int(torch.argmax(qvalues).detach().cpu().numpy())
                policy = self.env.directions[action]
                
                plt.text(free_cell[1] - offset[0], free_cell[0] - offset[1], policy, ha='center', va='center')
            
            # Plot the goal state
            plt.plot(self.env.goal[1], self.env.goal[0], 'bs', markersize=4)  # Mark the goal with a blue square
            
            plt.xticks([], [])  # Remove x-ticks
            plt.yticks([], [])  # Remove y-ticks
            plt.suptitle(f"{title} Policy (epoch: {epochnum})")
            plt.title(f"{hyperparams}")
            plt.axis('equal')
            plt.axis('off')  # Hide axes
            plt.savefig(filename, dpi=300, bbox_inches='tight')  # Save the figure
            plt.show()  # Display the figure

    def plot_Q_values(self, net, filename, offset, hyperparams,epochnum,title="DQN"):
        net.eval()  # Set the network to evaluation mode
        with torch.no_grad():
            # Define the new colormap
            cmap = ListedColormap([
                [1, 1, 1],  # white for empty spaces
                [0, 0, 0],  # black for walls
                [0.9290, 0.6940, 0.1250],  # tan for special state 2
                [1, 0, 0],  # red for special state 3
                [0, 0, 1],  # blue for special state 4
                [0, 1, 0]   # green for special state 5
            ])
            
            # Create a figure to display the maze
            plt.figure()
            plt.imshow(self.env.maze, cmap=cmap)
            
            # Loop through the allowed states to display policies
            for free_cell in self.env.allowed_states:
                self.env.current_position = np.asarray(free_cell)
                qvalues = net(torch.Tensor(self.env.state()).reshape(1, -1).to('cuda'))
                maxQ = torch.max(qvalues).detach().cpu().numpy()
                plt.text(free_cell[1] - offset[0], free_cell[0] - offset[1], "{:.1e}".format(maxQ.item()), ha='center', va='center', fontsize=3)
            
            # Plot the goal state
            plt.plot(self.env.goal[1], self.env.goal[0], 'bs', markersize=4)  # Mark the goal with a blue square
            
            plt.xticks([], [])  # Remove x-ticks
            plt.yticks([], [])  # Remove y-ticks
            plt.suptitle(f"{title} Q-values (epoch: {epochnum})")
            plt.title(f"{hyperparams}")
            plt.axis('equal')
            plt.axis('off')  # Hide axes
            plt.savefig(filename, dpi=300, bbox_inches='tight')  # Save the figure
            plt.show()  # Display the figure
