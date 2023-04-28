import argparse
import datetime
import gymnasium as gym
import numpy as np
import itertools
import torch
from sac import SAC, SAC_complicated, RGB_SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
import os
from tqdm import tqdm
from PIL import Image
import cv2
from model_image import StateEstimate
import torch.nn as nn

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
parser.add_argument('--num_steps', type=int, default=1001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

env = gym.make("InvertedPendulum-v4", render_mode = "rgb_array")

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = RGB_SAC(env.observation_space.shape[0], env.action_space, args)

#Tesnorboard
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0
device = torch.device("cuda" if args.cuda else "cpu")
state_estimate = StateEstimate().to(device)


for i_episode in itertools.count(1):
    episode_reward = 0
    episode_rand_reward = 0
    episode_steps = 0
    done = False
    env.reset()
    state_array = env.render() # 480 * 480 * 3
    state_array = cv2.cvtColor(state_array,cv2.COLOR_BGR2GRAY)
    w, h = state_array.shape
    state_array = cv2.resize(state_array, (int(w/4), int(h/4)))
    _, state_array = cv2.threshold(state_array, 50, 255, cv2.THRESH_BINARY)
    state = state_array.reshape(-1, int(w/4), int(h/4))
    state = torch.FloatTensor(state.copy()).to(device)
    state = torch.cat([state, state], 0)

    with torch.no_grad():
        prev_latent_state = state_estimate(state)



    # while not done:
    for count in tqdm(range(args.num_steps)):
        if args.start_steps > total_numsteps: 
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(prev_latent_state)  # Sample action from policy
        if len(memory) > args.batch_size:

            for i in range(args.updates_per_step):

                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters_gpu(memory, args.batch_size, updates)
                
                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        ob, reward, done, _, _ = env.step(action)

        next_state_array = env.render() # 480 * 480 * 3
        next_state_array = cv2.cvtColor(next_state_array,cv2.COLOR_BGR2GRAY)
        next_state_array = cv2.resize(next_state_array, (int(w/4), int(h/4)))
        _, next_state_array = cv2.threshold(next_state_array, 50, 255, cv2.THRESH_BINARY)
        next_state = next_state_array.reshape(-1, int(w/4), int(h/4))        
    
        next_state = torch.FloatTensor(next_state.copy()).to(device)

        next_state = torch.cat([state[1].reshape(-1, int(w/4), int(h/4)), next_state], 0)
        with torch.no_grad():
            latent_state = state_estimate(next_state)

        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)


        optimizer = torch.optim.Adam(state_estimate.parameters(), lr=0.1) 
        criterion = nn.MSELoss() 
    
        loss = criterion(torch.FloatTensor(ob).to('cuda'), latent_state)
        # print(latent_state)
        # print(torch.FloatTensor(ob).to('cuda'))
        loss.requires_grad_(True)
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step()


        memory.push(prev_latent_state, torch.FloatTensor(action).to('cuda'), torch.FloatTensor(np.array([reward])).to('cuda'), latent_state, torch.FloatTensor(np.array([mask])).to('cuda'))
        cv2.imshow('Inverted Pendulum', state_array)
        cv2.waitKey(20)
        prev_latent_state = latent_state
        state_array = next_state_array


    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, average reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward/args.num_steps, 2)))
    agent.save_checkpoint(args.env_name)


