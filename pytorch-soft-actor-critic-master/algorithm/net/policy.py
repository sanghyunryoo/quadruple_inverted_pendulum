from utility.utility import torch, nn, F, Normal, weights_init_

class GaussianPolicy(nn.Module):
    def __init__(self, o_dim, a_dim, h_dim):
        super(GaussianPolicy, self).__init__()
        self.o_dim, self.a_dim = o_dim, a_dim
        self.l1_dim, self.l2_dim, self.l3_dim = h_dim, h_dim, h_dim

        self.linear1 = nn.Linear(self.o_dim, self.l1_dim)
        self.linear2 = nn.Linear(self.l1_dim, self.l2_dim)
        self.linear3 = nn.Linear(self.l2_dim, self.l3_dim)
        self.linear4 = nn.Linear(self.l3_dim, self.a_dim)
        self.linear5 = nn.Linear(self.l3_dim, self.a_dim)

        self.apply(weights_init_)

    def forward(self, o):
        layer = self.linear1(o)
        layer = F.gelu(layer)
        layer = self.linear2(layer)
        layer = F.gelu(layer)
        layer = self.linear3(layer)
        layer = F.gelu(layer)
        mu = self.linear4(layer)
        log_sigma = self.linear5(layer)
        log_sigma = torch.clamp(log_sigma, min=-20., max=2.)
        return mu, log_sigma

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        sample = normal.rsample()
        action_mean = torch.tanh(mean)
        action_sample = torch.tanh(sample)
        log_prob = normal.log_prob(sample).sum(1, keepdim=True)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - action_sample.pow(2) + 1e-6).sum(1, keepdim=True)
        return action_sample, log_prob, action_mean

    def to(self, device):
        return super(GaussianPolicy, self).to(device)


class SIEGaussianPolicy(nn.Module):
    def __init__(self, o_dim, a_dim, eqi_idx, reg_idx, h_dim):
        super(SIEGaussianPolicy, self).__init__()
        self.eqi_o_idx = eqi_idx
        self.reg_o_idx = reg_idx
        self.eqi_o_dim = len(self.eqi_o_idx)
        self.reg_o_dim = len(self.reg_o_idx)
        self.o_dim, self.a_dim = o_dim, a_dim
        self.l1_dim, self.l2_dim, self.l3_dim = h_dim, h_dim, h_dim

        self.linear1 = nn.Linear(self.eqi_o_dim, self.l1_dim, bias=False)
        self.linear2 = nn.Linear(self.reg_o_dim, self.l1_dim)
        self.linear3 = nn.Linear(self.l1_dim, self.l2_dim)
        self.linear4 = nn.Linear(self.l2_dim, self.l3_dim)
        self.linear5 = nn.Linear(self.l3_dim, self.a_dim)
        self.linear6 = nn.Linear(self.l3_dim, self.a_dim)

        self.apply(weights_init_)

    def forward(self, o):

        eqi_obs_input = torch.index_select(o, 1, self.eqi_o_idx)
        reg_obs_input = torch.index_select(o, 1, self.reg_o_idx)
        eqi_feature = self.linear1(eqi_obs_input)
        source = torch.sum(eqi_feature, dim=1, keepdim=True)
        multiplier = torch.where(source < 0.0, -1.0 * torch.ones_like(source), torch.ones_like(source))
        reg_feature = self.linear2(reg_obs_input)
        feature = torch.abs(eqi_feature) + reg_feature
        layer = F.gelu(feature)
        layer = self.linear3(layer)
        layer = F.gelu(layer)
        layer = self.linear4(layer)
        layer = F.gelu(layer)
        mu = self.linear5(layer)
        mu *= multiplier
        log_sigma = self.linear6(layer)
        log_sigma = torch.clamp(log_sigma, min=-20., max=2.)
        return mu, log_sigma

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        sample = normal.rsample()
        action_mean = torch.tanh(mean)
        action_sample = torch.tanh(sample)
        log_prob = normal.log_prob(sample).sum(1, keepdim=True)
        # Enforcing Action Bound
        log_prob -= torch.log(1 - action_sample.pow(2) + 1e-6).sum(1, keepdim=True)
        return action_sample, log_prob, action_mean

    def to(self, device):
        return super(SIEGaussianPolicy, self).to(device)