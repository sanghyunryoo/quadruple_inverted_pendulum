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

        self.apply(weights_init_)

    def forward(self, o, a): # state == o
        inv_o = torch.index_select(o, 1, self.inv_o_idx)
        reg_o = torch.index_select(o, 1, self.reg_o_idx)

        inv_inputs = torch.cat([inv_o, a], 1)
        inv_feature = self.linear1(inv_inputs)
        reg_feature = self.linear2(reg_o)
        feature = torch.abs(inv_feature) + reg_feature
        layer = feature
        layer = F.gelu(layer)
        layer = self.linear3(layer)
        layer = F.gelu(layer)
        qval = self.linear4(layer)

        return qval
    


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

        self.linear3 = nn.Linear(self.l1_dim, self.l2_dim*3) 
        self.linear4 = nn.Linear(self.l1_dim*3, self.l2_dim*2) 
        self.linear5 = nn.Linear(self.l1_dim*2, self.l2_dim)        
        self.linear6 = nn.Linear(self.l2_dim, 1)


        self.linear7 = nn.Linear(self.inv_o_dim + self.a_dim, self.l1_dim)
        self.linear8 = nn.Linear(self.reg_o_dim, self.l1_dim)

        self.linear9 = nn.Linear(self.l1_dim, self.l2_dim*3) 
        self.linear10 = nn.Linear(self.l1_dim*3, self.l2_dim*2) 
        self.linear11 = nn.Linear(self.l1_dim*2, self.l2_dim)        
        self.linear12 = nn.Linear(self.l2_dim, 1)

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
        layer1 = self.linear4(layer1)
        layer1 = F.gelu(layer1)
        layer1 = self.linear5(layer1)
        layer1 = F.gelu(layer1)
        qval1 = self.linear6(layer1)


        inv_feature2 = self.linear7(inv_inputs)
        reg_feature2 = self.linear8(reg_o)
        feature2 = torch.abs(inv_feature2) + reg_feature2
        layer2 = feature2
        layer2 = F.gelu(layer2)
        layer2 = self.linear9(layer2)
        layer2 = F.gelu(layer2)
        layer2 = self.linear10(layer2)
        layer2 = F.gelu(layer2)
        layer2 = self.linear11(layer2)
        layer2 = F.gelu(layer2)
        qval2 = self.linear12(layer2)

        return qval1, qval2


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
    



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   