import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import cv2

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

  
class QNetwork_img(nn.Module):
    def __init__(self, hidden_dim):
        super(QNetwork_img, self).__init__()

        # Q1 architecture
        self.conv1 = nn.Conv2d(2, 2, kernel_size=3)
        self.conv2 = nn.Conv2d(2, 2, kernel_size=3)
        self.conv3 = nn.Conv2d(2, 2, kernel_size=3)
        self.conv4 = nn.Conv2d(2, 1, kernel_size=3) 
        self.mp = nn.MaxPool2d(2)       
        self.linear1 = nn.Linear(26, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.conv5 = nn.Conv2d(2, 2, kernel_size=3)
        self.conv6 = nn.Conv2d(2, 2, kernel_size=3)
        self.conv7 = nn.Conv2d(2, 2, kernel_size=3)
        self.conv8 = nn.Conv2d(2, 1, kernel_size=3) 
        self.mp = nn.MaxPool2d(2)             
        self.linear4 = nn.Linear(26, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):


        xu = state

        x1 = self.mp(F.relu(self.conv1(xu)))
        x1 = self.mp(F.relu(self.conv2(x1)))
        x1 = self.mp(F.relu(self.conv3(x1)))
        x1 = self.mp(F.relu(self.conv4(x1)))
        x1 = x1.view(x1.size(0), -1)             
        x1 = torch.cat([x1, action], 1)       
        x1 = F.relu(self.linear1(x1))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = self.mp(F.relu(self.conv5(xu)))
        x2 = self.mp(F.relu(self.conv6(x2)))
        x2 = self.mp(F.relu(self.conv7(x2)))
        x2 = self.mp(F.relu(self.conv8(x2)))          
        x2 = x2.view(x2.size(0), -1)
        x2 = torch.cat([x2, action], 1)
        x2 = F.relu(self.linear1(x2))
        x2 = F.relu(self.linear2(x2))
        x2 = self.linear3(x2)

        return x1, x2

class GaussianPolicy_img(nn.Module):
    def __init__(self, num_actions, hidden_dim, action_space):
        super(GaussianPolicy_img, self).__init__()

        self.conv1 = nn.Conv2d(2, 2, kernel_size=3)
        self.conv2 = nn.Conv2d(2, 2, kernel_size=3)
        self.conv3 = nn.Conv2d(2, 2, kernel_size=3)
        self.conv4 = nn.Conv2d(2, 1, kernel_size=3)
        self.mp = nn.MaxPool2d(2)               

        self.linear1 = nn.Linear(25, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = self.mp(F.relu(self.conv1(state)))
        x = self.mp(F.relu(self.conv2(x)))
        x = self.mp(F.relu(self.conv3(x)))
        x = self.mp(F.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy_img, self).to(device)

class StateEstimate(nn.Module):
    def __init__(self):
        super(StateEstimate, self).__init__()
        
        self.conv1 = nn.Conv2d(2, 2, kernel_size=5)
        self.conv2 = nn.Conv2d(2, 2, kernel_size=5)
        self.conv3 = nn.Conv2d(2, 2, kernel_size=5)
        self.conv4 = nn.Conv2d(2, 2, kernel_size=3)
        self.mp = nn.MaxPool2d(2)   

        self.linear1 = nn.Linear(32, 12)
        self.linear2 = nn.Linear(12, 4)   


    def forward(self, x):
        x = self.mp(F.relu(self.conv1(x)))
        x = self.mp(F.relu(self.conv2(x)))
        x = self.mp(F.relu(self.conv3(x)))
        x = self.mp(F.relu(self.conv4(x)))
        x = x.view(-1)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return x
