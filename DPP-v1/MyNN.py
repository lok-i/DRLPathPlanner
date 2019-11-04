import gym
import gym_minigrid
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


env = gym.make('MiniGrid-Dynamic-Obstacles-8x8-v0')#'MiniGrid-Empty-5x5-v0')

print(env.observation_space)
print(env.action_space)



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):




    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=2, stride=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 64
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

BATCH_SIZE = 256 #128
GAMMA = 0.999
EPS_START = 1
EPS_END = 0.05

EPS_DECAY = 4000 #200 for 50 epi
num_episodes = 500
TARGET_UPDATE = 30
#PATH ='./logs/'

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()


init_screen = env.reset()
screen_height =7 
screen_width = 7 
device = 'cpu'
render_status = False
save_model = True
PATH = './logs/'

# Get number of actions from gym action space
n_actions = env.action_space.n

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(10000)


episode_durations = []
eps_of_episode = []
reward_hist=[]
steps_done = 0


def save_logs(Network , grap_plot ,Test_name , ith_sample):
    torch.save(Network,PATH+Test_name+ith_sample)
    plt.savefig(PATH+ith_sample+'.png')
    

def get_state(x):
	#print(x)
	x.unsqueeze_(0)
	
	x.transpose_(1,3)
	#print(x)
	return x

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
   	

    steps_done += 1
    eps_of_episode.append(eps_threshold)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)




def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.subplot(221)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration_alive')
    plt.plot(durations_t.numpy())
    plt.subplot(222)
    #plt.title('Training...')
    plt.xlabel('Time_Steps')
    plt.ylabel('Epsilon')
    plt.plot(torch.tensor(eps_of_episode,dtype=torch.float).numpy())
    plt.subplot(223)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Total_reward')
    rew_hist_tensor = torch.tensor(reward_hist,dtype=torch.float)
    plt.plot(rew_hist_tensor.numpy())
    #plt.show()

    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:

    	plt.subplot(221)
    	means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    	means = torch.cat((torch.zeros(99), means))
    	plt.plot(means.numpy())
    	plt.subplot(223)
    	means = rew_hist_tensor.unfold(0, 100, 1).mean(1).view(-1)
    	means = torch.cat((torch.zeros(99), means))
    	plt.plot(means.numpy())


    plt.pause(0.001)  # pause a bit so that plots are updated
    # if is_ipython:
    #     display.clear_output(wait=True)
    #     display.display(plt.gcf())      

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def train():
	
	for i_episode in range(num_episodes):
	    # Initialize the environment and state
	    env.reset()
	    #last_screen = env.render()

	    #current_screen = env.render()
	    
	    state_dic,_,_,_=     env.step(env.action_space.sample())#env.render()
	    state = get_state(torch.tensor(state_dic['image'],dtype =torch.float))
	    #print(state.size())
	    #print(state)
	    reward_epi = 0 
	    for t in count():
	        # Select and perform an action
	        if render_status:
	        	env.render()
	        action = select_action(state)
	        observation, reward, done, _ = env.step(action.item())
	        
	        reward = torch.tensor([reward],dtype=torch.long, device=device)
	        

	        # Observe new state
	        
	        
	        if not done:
	            next_state =get_state(torch.tensor(observation['image'],dtype =torch.float)) #env.render()# current_screen - last_screen
	        else:
	            next_state = None

	        # Store the transition in memory
	        memory.push(state, action, next_state, reward)

	        # Move to the next state
	        state = next_state

	        # Perform one step of the optimization (on the target network)
	        reward_epi = reward_epi+reward.item()
	        optimize_model()

	        if done:
	        	if t != 0:
	        		reward_epi = reward_epi/t
	        	reward_hist.append(reward_epi)
	        	episode_durations.append(t + 1)
	        	plot_durations()
	        	break
	        	
	    # Update the target network, copying all weights and biases in DQN
	    if i_episode % TARGET_UPDATE == 0:
	        target_net.load_state_dict(policy_net.state_dict())

	print('Complete')
	#env.render()
	env.close()
	#plt.ioff()
	#plt.show()


if __name__ == '__main__':
	print("Training :")
	train()
	Folder_name = 'test_(8x8)'
	if save_model == True :
		save_logs(policy_net,plt,Folder_name,'2')

	
	








# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()