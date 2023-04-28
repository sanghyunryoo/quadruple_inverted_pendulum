import argparse, datetime
from utility.utility import torch, np, Logger, set_lib_seed
from algorithm.algorithms import SAC, SIESAC
from utility.replay_memory import ReplayMemory
from environment.envs import ENV
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--task_name', default="CoCELQIP",
                    help='CoCEL pendulum environment (default: CoCELQIP)')
parser.add_argument('--algorithm', default="siesac",
                    help='Algorithm name')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--target_entropy', type=float, default=-1.5, metavar='G',
                    help='target entropy (e.g., -dim(A) (default: -1.5)')
parser.add_argument('--reward_scale', type=float, default=1, metavar='G',
                    help='reward scale (default: 5)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--maximum_global_steps', type=int, default=4000000, metavar='N',
                    help='maximum number of global steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=512, metavar='N',
                    help='hidden size (default: 512)')
parser.add_argument('--n_history', type=int, default=3, metavar='N',
                    help='the number of history composition for constructing an observation (default: 3)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=1000, metavar='N',
                    help='Steps sampling random actions (default: 1000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--n_eval', type=int, default=10, metavar='N',
                    help='the number of episodes for each evaluation (default: 10)')
parser.add_argument('--cuda', type=bool, default=True,
                    help='run on CUDA (default: False)')
args = parser.parse_args()

# Define environment
env = ENV(args.task_name, n_history=args.n_history)

# Set logger
logger = Logger(args)
writer = SummaryWriter(
    'runs/{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                 args.algorithm, args.task_name, args.seed))

# Set random_seed
env.set_seed(args.seed)
set_lib_seed(args.seed)

# Define agent
device = torch.device("cuda" if args.cuda else "cpu")
if args.algorithm == "sac":
    agent = SAC(env.state_dim, env.action_dim, args)
elif args.algorithm == "siesac":
    eqi_idx = torch.tensor(env.eqi_idx).to(device)
    reg_idx = torch.tensor(env.reg_idx).to(device)
    agent = SIESAC(env.state_dim, env.action_dim, eqi_idx, reg_idx, args)
else:
    raise ValueError("[Error] Choose a valid algorithm.")

# Replay buffer
memory = ReplayMemory(args.replay_size)

# Training Loop
total_numsteps = 0
episode_steps = 0
updates = 0
step_since_eval = 0

while total_numsteps <= args.maximum_global_steps:
    episode_reward = 0
    local_step = 0
    obs, _ = env.reset()
    obs = torch.FloatTensor(obs).to(device)
    for step in range(env.max_step()):
        if args.start_steps > total_numsteps:
            action = np.random.uniform(-1.,1.,env.action_dim)
        else:
            action = agent.select_action(obs, evaluate=False)
        next_obs, reward, done, _, _ = env.step(action * env.action_max)
        local_step += 1
        step_since_eval += 1
        total_numsteps += 1
        episode_reward += reward
        mask = not done
        memory.push(obs.to('cuda'), torch.FloatTensor(action).to('cuda'),
                    torch.FloatTensor(np.array([reward * args.reward_scale])).to('cuda'), torch.FloatTensor(next_obs).to('cuda'),
                    torch.FloatTensor(np.array([mask])).to('cuda'))

        if len(memory) > args.batch_size:
            for i in range(args.updates_per_step):
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                     args.batch_size,
                                                                                                     updates)
                updates += 1
                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)

        obs = torch.FloatTensor(next_obs).to(device)
        if done: break

    episode_steps += 1
    writer.add_scalar('avg_reward/train', episode_reward/local_step, total_numsteps)
    print("[INTERACTION] Episode: %d, GlobalStep: %d, Total.Rwd: %.3f, Avg.Rwd: %.3f"%
          (episode_steps, total_numsteps, episode_reward, episode_reward/local_step))

    if step_since_eval > 10000 and args.eval:
        step_since_eval %= 10000
        eval_global_step = 0
        eval_cum_reward = 0
        returns = []
        for eval in range(args.n_eval):
            episode_reward = 0
            obs, _ = env.reset()
            obs = torch.FloatTensor(obs).to(device)
            for local_step in range(env.max_step()):
                env.render()
                action = agent.select_action(obs, evaluate=True)
                next_obs, reward, done, _, _ = env.step(action * env.action_max)
                obs = torch.FloatTensor(next_obs).to(device)
                episode_reward += reward
                eval_global_step += 1
                if done: break
            eval_cum_reward += episode_reward
            returns.append(episode_reward)

        eval_avg_reward = eval_cum_reward / eval_global_step
        avg_total_reward = float(np.mean(returns))
        writer.add_scalar('avg_reward/test', eval_avg_reward, total_numsteps)

        print("----------------------------------------")
        print("[EVALUATION] Episode: %d, GlobalStep: %d, Avg.Total.Rwd: %.3f, Avg.Rwd: %.3f"%
              (episode_steps, total_numsteps, avg_total_reward, eval_avg_reward))
        print("----------------------------------------")

        logger.log_eval(episode_steps, total_numsteps, returns, eval_avg_reward)
        agent.save_checkpoint(args.task_name, ckpt_path=logger.root_model+"checkpoint")