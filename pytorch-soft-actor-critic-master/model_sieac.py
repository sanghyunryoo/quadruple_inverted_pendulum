import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)



 
class QNetwork_sieac(nn.Module):
    def __init__(self, o_dim, a_dim, eqi_idx, reg_idx, h1_dim, h2_dim): # num_inputs == o_dim, num_actions == a_dim
        super(QNetwork_sieac, self).__init__()

        self.inv_o_idx = eqi_idx
        self.reg_o_idx = reg_idx
        self.inv_o_dim = len(self.inv_o_idx)
        self.reg_o_dim = len(self.reg_o_idx)
        self.o_dim, self.a_dim = o_dim, a_dim
        self.l1_dim, self.l2_dim = h1_dim, h2_dim

        # Q1 architecture
        self.linear1 = nn.Linear(self.inv_o_dim + self.a_dim, self.l1_dim)
        self.linear2 = nn.Linear(self.reg_o_dim, self.l1_dim)

        self.linear3 = nn.Linear(self.l1_dim, self.l2_dim)      
        self.linear4 = nn.Linear(self.l2_dim, 1)


        self.linear5 = nn.Linear(self.inv_o_dim + self.a_dim, self.l1_dim)
        self.linear6 = nn.Linear(self.reg_o_dim, self.l1_dim)

        self.linear7 = nn.Linear(self.l1_dim, self.l2_dim)        
        self.linear8 = nn.Linear(self.l2_dim, 1)

        self.apply(weights_init_)

    def forward(self, o, a): # state == o
        inv_o = torch.index_select(o, 1, self.inv_o_idx)
        reg_o = torch.index_select(o, 1, self.reg_o_idx)

        inv_inputs = torch.cat([inv_o, a], 1)

        inv_feature1 = self.linear1(inv_inputs)
        reg_feature1 = self.linear2(reg_o)
        feature1 = torch.abs(inv_feature1) + reg_feature1
        layer1 = feature1
        layer1 = F.gelu(layer1)
        layer1 = self.linear3(layer1)
        layer1 = F.gelu(layer1)
        qval1 = self.linear4(layer1)


        inv_feature2 = self.linear5(inv_inputs)
        reg_feature2 = self.linear6(reg_o)
        feature2 = torch.abs(inv_feature2) + reg_feature2
        layer2 = feature2
        layer2 = F.gelu(layer2)
        layer2 = self.linear7(layer2)
        layer2 = F.gelu(layer2)
        qval2 = self.linear8(layer2)

        return qval1, qval2

class GaussianPolicy_sieac(nn.Module):
    def __init__(self, o_dim, a_dim, eqi_idx, reg_idx, h1_dim, h2_dim, action_space=None):
        super(GaussianPolicy_sieac, self).__init__()
        
        self.eqi_o_idx = eqi_idx
        self.reg_o_idx = reg_idx
        self.eqi_o_dim = len(self.eqi_o_idx)
        self.reg_o_dim = len(self.reg_o_idx)
        self.o_dim, self.a_dim = o_dim, a_dim
        self.l1_dim, self.l2_dim = h1_dim, h2_dim

        self.apply(weights_init_)

        self.linear1 = nn.Linear(self.eqi_o_dim, self.l1_dim, bias = False)
        self.linear2 = nn.Linear(self.reg_o_dim, self.l1_dim)
        self.linear3 = nn.Linear(self.l1_dim, self.l2_dim)
        self.linear4 = nn.Linear(self.l2_dim, self.a_dim)
        self.linear5 = nn.Linear(self.l2_dim, self.a_dim)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)
            
    def forward(self, o):

        
        eqi_obs_input = torch.index_select(o, 1, self.eqi_o_idx)
        reg_obs_input = torch.index_select(o, 1, self.reg_o_idx)
        eqi_feature = self.linear1(eqi_obs_input)
        source = torch.sum(eqi_feature,axis=1,keepdims=True)
        multiplier = torch.where(source < 0.0, -1.0*torch.ones_like(source), torch.ones_like(source))
        reg_feature = self.linear2(reg_obs_input)

        feature = torch.abs(eqi_feature) + reg_feature
        layer = F.gelu(feature)
        layer = self.linear3(layer)
        layer = F.gelu(layer)
        mu = self.linear4(layer)
        mu *= multiplier
        log_sigma = self.linear5(layer)
        log_sigma = torch.clamp(log_sigma, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        # sigma = torch.exp(torch.clamp(log_sigma, self.LOG_SIG_MIN, self.LOG_SIG_MAX))
        # dist = Independent(Normal(mu, sigma), 1)

        # samples = dist.rsample()

        # actions = torch.tanh(samples)
        # log_probs = torch.reshape(dist.log_prob(samples), [-1, 1])
        # log_probs -= torch.sum(torch.log(1 - actions ** 2 + 1e-10), axis=1, keepdims=True)
        return mu, log_sigma


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
        return super(GaussianPolicy_sieac, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
