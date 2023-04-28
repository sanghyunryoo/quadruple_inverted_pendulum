import os, yaml, math, torch, random
import pandas as pd
import numpy as np
import importlib.util
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Logger:

    def __init__(self, config):
        self.root_model = "./checkpoints/%s/%s_%s/" % (config.task_name, config.algorithm, config.seed)
        self.root_param = self.root_model + "parameter/"
        self.root_buffer = self.root_model + "buffer/"
        self.config = config
        self.create_dir()

    def create_dir(self):
        if not os.path.isdir("./checkpoints/%s" % self.config.task_name): os.mkdir("./checkpoints/%s" % self.config.task_name)
        if not os.path.isdir(self.root_model[:-1]): os.mkdir(self.root_model[:-1])
        if not os.path.isdir(self.root_param[:-1]): os.mkdir(self.root_param[:-1])
        if not os.path.isdir(self.root_buffer[:-1]): os.mkdir(self.root_buffer[:-1])
        self.monitoring_file = self.root_model + "learning_curve.csv"
        file = pd.DataFrame(columns=["Episode", "Step", "Max", "Min", "Average", "Avg.Rwd"])
        file.to_csv(self.monitoring_file, index=False)

    def log_eval(self, epi, step, returns, avg_reward):
        max_eval = np.max(returns)
        min_eval = np.min(returns)
        avg_eval = np.mean(returns)
        file = pd.read_csv(self.monitoring_file)
        file.loc[len(file)] = [epi, step, max_eval, min_eval, avg_eval, avg_reward]
        file.to_csv(self.monitoring_file, index=False)

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def arr2str(array):
    return np.array2string(array,precision=2,separator=',')

def read_config(path, struct=True):
    with open(path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, yaml.Loader)
    if struct:
        cfg = Struct(**cfg)
    return cfg

def load_class(path):
    spec = importlib.util.spec_from_file_location("ReadClass", path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

def set_lib_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)