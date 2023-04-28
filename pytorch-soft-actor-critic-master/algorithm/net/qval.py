from utility.utility import torch, nn, F, weights_init_

class Qnet(nn.Module):
    def __init__(self, o_dim, a_dim, h_dim):
        super(Qnet, self).__init__()
        self.o_dim, self.a_dim = o_dim, a_dim
        self.l1_dim, self.l2_dim, self.l3_dim = h_dim, h_dim, h_dim

        # Q1 architecture
        self.linear11 = nn.Linear(self.o_dim + self.a_dim, self.l1_dim)
        self.linear12 = nn.Linear(self.l1_dim, self.l2_dim)
        self.linear13 = nn.Linear(self.l2_dim, self.l3_dim)
        self.linear14 = nn.Linear(self.l3_dim, 1)

        # Q2 architecture
        self.linear21 = nn.Linear(self.o_dim + self.a_dim, self.l1_dim)
        self.linear22 = nn.Linear(self.l1_dim, self.l2_dim)
        self.linear23 = nn.Linear(self.l2_dim, self.l3_dim)
        self.linear24 = nn.Linear(self.l3_dim, 1)

        self.apply(weights_init_)

    def forward(self, o, a):
        inputs = torch.cat([o, a], 1)

        layer1 = self.linear11(inputs)
        layer1 = F.gelu(layer1)
        layer1 = self.linear12(layer1)
        layer1 = F.gelu(layer1)
        layer1 = self.linear13(layer1)
        layer1 = F.gelu(layer1)
        qval1 = self.linear14(layer1)

        layer2 = self.linear21(inputs)
        layer2 = F.gelu(layer2)
        layer2 = self.linear22(layer2)
        layer2 = F.gelu(layer2)
        layer2 = self.linear23(layer2)
        layer2 = F.gelu(layer2)
        qval2 = self.linear24(layer2)

        return qval1, qval2

    def to(self, device):
        return super(Qnet, self).to(device)


class SIEQnet(nn.Module):
    def __init__(self, o_dim, a_dim, eqi_idx, reg_idx, h_dim):
        super(SIEQnet, self).__init__()

        self.inv_o_idx = eqi_idx
        self.reg_o_idx = reg_idx
        self.inv_o_dim = len(self.inv_o_idx)
        self.reg_o_dim = len(self.reg_o_idx)
        self.o_dim, self.a_dim = o_dim, a_dim
        self.l1_dim, self.l2_dim, self.l3_dim = h_dim, h_dim, h_dim

        # Q1 architecture
        self.linear11 = nn.Linear(self.inv_o_dim + self.a_dim, self.l1_dim, bias=False)
        self.linear12 = nn.Linear(self.reg_o_dim, self.l1_dim)
        self.linear13 = nn.Linear(self.l1_dim, self.l2_dim)
        self.linear14 = nn.Linear(self.l2_dim, self.l3_dim)
        self.linear15 = nn.Linear(self.l3_dim, 1)

        # Q2 architecture
        self.linear21 = nn.Linear(self.inv_o_dim + self.a_dim, self.l1_dim, bias=False)
        self.linear22 = nn.Linear(self.reg_o_dim, self.l1_dim)
        self.linear23 = nn.Linear(self.l1_dim, self.l2_dim)
        self.linear24 = nn.Linear(self.l2_dim, self.l3_dim)
        self.linear25 = nn.Linear(self.l3_dim, 1)

        self.apply(weights_init_)

    def forward(self, o, a):  # state == o
        inv_o = torch.index_select(o, 1, self.inv_o_idx)
        reg_o = torch.index_select(o, 1, self.reg_o_idx)

        inv_inputs = torch.cat([inv_o, a], 1)

        inv_feature1 = self.linear11(inv_inputs)
        reg_feature1 = self.linear12(reg_o)
        feature1 = torch.abs(inv_feature1) + reg_feature1
        layer1 = feature1
        layer1 = F.gelu(layer1)
        layer1 = self.linear13(layer1)
        layer1 = F.gelu(layer1)
        layer1 = self.linear14(layer1)
        layer1 = F.gelu(layer1)
        qval1 = self.linear15(layer1)

        inv_feature2 = self.linear21(inv_inputs)
        reg_feature2 = self.linear22(reg_o)
        feature2 = torch.abs(inv_feature2) + reg_feature2
        layer2 = feature2
        layer2 = F.gelu(layer2)
        layer2 = self.linear23(layer2)
        layer2 = F.gelu(layer2)
        layer2 = self.linear24(layer2)
        layer2 = F.gelu(layer2)
        qval2 = self.linear25(layer2)

        return qval1, qval2