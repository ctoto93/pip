import numpy as np
import pandas as pd
import torch
import gym
import argparse
import os

import utils
import CACLA
import PIP
import PSOPlanner
import envs
import matplotlib.pyplot as plt
from tqdm import tqdm

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, n_eval=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(n_eval):
		state, done = eval_env.reset(), False
        while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes
	return avg_reward

def init_arguments():
    parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")                  # Policy name (CACLA, TD3, PIP)
	parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, TF and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=1000, type=int)      # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=50000, type=int) # Max time steps to run environment
	parser.add_argument("--n_steps", default=50, type=int)   		# 1 episode how many steps in the environment
	parser.add_argument("--expl_noise", default=0.5, type=float)    # CACLA and TD3 std of Gaussian exploration noise
	parser.add_argument("--buffer_size", default=1e6, type=int)     # replay buffer size
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.95)                 # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)         # TD3 Target network update rate
	parser.add_argument("--policy_noise", default=0.2, type=float)  # TD3 Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5, type=float)    # TD3 Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # TD3 Frequency of delayed policy updates
    parser.add_argument("--plan_length", default=5, type=int)       # PIP plan's length
    parser.add_argument("--c1", default=1.4, type=float)            # PIP PSO c1 coef
    parser.add_argument("--c2", default=1.4, type=float)            # PIP PSO c2 coef
    parser.add_argument("--w", default=0.7, type=float)             # PIP PSO w coef
    parser.add_argument("--neighbour", default=20, type=int)        # PIP PSO number of neighbour
    parser.add_argument("--particles", default=50, type=int)        # PIP PSO number of particles
	parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = init_arguments()
	file_name = f"{args.policy}_{args.env}_expl_noise_{args.expl_noise}_policy_noise_{args.policy_noise}_noise_clip_{args.noise_clip}_tau_{args.tau}_steps_{args.n_steps}_buffer_{args.buffer_size}_seed_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)

	# Set seeds
	env.seed(args.seed)
	tf.random.set_seed(args.seed)
	np.random.seed(args.seed)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)
	elif args.policy == "OurDDPG":
		policy = OurDDPG.DDPG(**kwargs)
	elif args.policy == "DDPG":
		policy = DDPG.DDPG(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=args.buffer_size)

	# Evaluate untrained policy

	evaluations = [eval_policy(policy, args.env, args.seed, n_steps=args.n_steps)]
	train_rewards = []

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	df_eval = pd.DataFrame(columns =  ["episode", "cum_rewards", "eval"])

	for t in tqdm(range(args.max_timesteps)):
		state, done = env.reset(), False
		episode_reward = 0

        while not done:
			if t < args.start_timesteps:
				action = env.action_space.sample()
			else:
				action = policy.select_action(np.array(state))

			next_state, reward, done, _ = env.step(action)
			done_bool = 0

			replay_buffer.add(state, action, next_state, reward)

			state = next_state
			episode_reward += reward

		train_rewards.append(episode_reward)
		policy.train(replay_buffer, args.batch_size)

		# Evaluate and save data frames
		if (t + 1) % args.eval_freq == 0:
			reward = eval_policy(policy, args.env, args.seed, args.n_steps)
			df_eval = df_eval.append({"episode": t+1, "cum_rewards": reward}, ignore_index=True)
			df_eval.to_pickle(f"./results/{file_name}_df_eval.pkl")

			df_train = pd.DataFrame()
			df_train["episode"] = np.arange(len(train_rewards))
			df_train["train_cum_reward"] = train_rewards
			df_train.to_pickle(f"./results/{file_name}_df_train.pkl")

	plt.plot(train_rewards)
	plt.savefig(f"results/{file_name}.png")
