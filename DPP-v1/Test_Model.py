import gym
import gym_minigrid
from itertools import count
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

env = gym.make('MiniGrid-Dynamic-Obstacles-8x8-v0')#'MiniGrid-Empty-5x5-v0')
Fil_name='test_1'
PATH = './logs/'+Fil_name
print(env.observation_space)
print(env.action_space)


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
def get_state(x):
	#print(x)
	x.unsqueeze_(0)
	
	x.transpose_(1,3)
	#print(x)
	return x


model = torch.load(PATH)
model.eval()
device = 'cpu'
def test_model(num_of_episodes,):
	
	
	for i in range(num_of_episodes):
		env.reset()

		state_dic,_,_,_=     env.step(env.action_space.sample())#env.render()
		state = get_state(torch.tensor(state_dic['image'],dtype =torch.float))
		reward_this_epi = 0
		for t in count():
			env.render()
			time.sleep(0.1)
			action = model(state).max(1)[1].view(1, 1)
			observation,reward ,done,_ =env.step(action.item())
			reward = torch.tensor([reward],dtype=torch.long, device=device)
			reward_this_epi = reward_this_epi+ reward.item()
			state =get_state(torch.tensor(observation['image'],dtype =torch.float))
			if done:
				reward_this_epi =reward_this_epi/(t+1)
				print('Episode:',i,'Episode_Score:',reward_this_epi,'Steps_alive',t+1)
				break

		
if __name__ == '__main__':
	print('Testing..')
	test_model(10)
