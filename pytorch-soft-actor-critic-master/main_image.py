import argparse
import datetime
import gymnasium as gym
import numpy as np
import itertools
import torch
from sac import SAC, Image_SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import os
from tqdm import tqdm
import cv2
import pickle 
import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="inverted_pendulum_series",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=10001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=1, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()



class CustomDataset(Dataset): 
  def __init__(self, train_data, train_label):
    self.x_data = train_data
    self.y_data = train_label

  # 총 데이터의 개수를 리턴
  def __len__(self): 
    return len(self.x_data)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx): 
    x = torch.FloatTensor(self.x_data[idx])
    y = torch.FloatTensor(self.y_data[idx])
    return x, y


width = 64
height = 64
batch = 512
label_num = 4
action_len = 1




device = torch.device("cuda" if args.cuda else "cpu")


print('===================================Data Load Start===================================\n')
k_fold = 1

with open("/home/cocel/samsung/samsung_dataset/kfold_" + str(k_fold), "rb") as data_file:
    data = pickle.load(data_file)

train_data = data["X_train"]
test_data = data["X_test"]
train_label = data["y_train"]
test_label = data["y_test"]


test_data = np.expand_dims(test_data, axis=1)  # make a channel
test_label = np.eye(12)[test_label]  # to one-hot encoding



# Train / Test Data Checking
train_count = np.zeros((12,), dtype=int)
for index in range(len(train_label)):
    train_count[np.argmax(train_label[index])] += 1
test_count = np.zeros((12,), dtype=int)
for index in range(len(test_label)):
    test_count[np.argmax(test_label[index])] += 1

train_data = train_data[train_count[0]:np.sum(train_count[0:label_num+1])]
train_label = train_label[train_count[0]:np.sum(train_count[0:label_num+1])][:,1:label_num+1]
train_label = np.argmax(train_label, -1)
test_data = test_data[test_count[0]:np.sum(test_count[0:label_num+1])]


test_label = test_label[test_count[0]:np.sum(test_count[0:label_num+1]),1:label_num+1]
test_label = np.argmax(test_label, axis = 1)
# test_label = np.eye(label_num)[test_label]  # to one-hot encoding


train_label = train_label.reshape(-1, 1)
test_label = test_label.reshape(-1, 1)



train_dataset = CustomDataset(train_data, train_label)
test_dataset = CustomDataset(test_data, test_label)

train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=True)
print('===================================Data Load Done====================================\n')


torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = Image_SAC(width, height, args, action_len = action_len)


# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0


for i_episode in itertools.count(1):
    episode_reward = 0
    episode_rand_reward = 0
    episode_steps = 0
    done = False
    for state, label in tqdm(train_dataloader):
        state = state.to(device)
        for count in range(len(state)):

            action = agent.select_action(state[count])  # Sample action from policy
                
            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)
                
                    updates += 1


            next_state = state[count]
            # print(np.round(action)[0])
            # print(label[count].item())

            if np.round(action)[0] == label[count].item():
                reward = 1
            else:
                reward = 0

            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1
            memory.push(state[count].to('cpu'), torch.FloatTensor(action).to('cpu'), reward, next_state.to('cpu'), mask) # Append transition to memory



    print("Episode: {}, total numsteps: {}, episode steps: {}, average reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward/episode_steps*100, 2)))

    print('=====================================Test Begin========================================\n')

    test_reward = 0
    attempt = 0
    # try:


    for state, label in tqdm(test_dataloader):
        state = state.to(device)
        for count in range(len(state)):
        
            action = agent.select_action(state[count])  # Sample action from policy    
            if np.round(action)[0] == label[count].item():
                reward = 1  
            else:
                reward = 0

            test_reward += reward
            attempt += 1
    # except:
    print("Test Result: {}".format(round(test_reward/attempt*100, 2)))

    print('=====================================Test Done========================================\n') 
