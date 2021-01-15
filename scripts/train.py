import numpy as np
import pandas as pd
import seaborn as sns
import gym
import argparse
import os
import envs
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from pathlib import Path
from distutils.dir_util import copy_tree

from cacla import CACLA
from pip import PIP
from pso_planner import PSOPlanner
from world_model import WorldModel
from network_with_transformer import NetworkWithTransformer
from rbf_transformer import RadialBasisTransformer, RadialBasisTransformerConcat
from replay_buffer import ReplayBuffer
from plotter.rbf_pendulum import visualize_pendulum


# Runs agent for X episodes and returns average reward
# A fixed seed is used for the eval environment
def evaluate_agent(agent, env_name, seed, n_eval=1, transformer=None):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(n_eval):
		state, done = eval_env.reset(), False
		while not done:
			action = agent(state.reshape(1,-1))
			if tf.is_tensor(action):
				action = action.numpy()

			action = action.reshape(-1)

			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

			avg_reward /= n_eval
			return avg_reward

def build_actor(n_obs, n_action, lr):
	actor = Sequential()
	actor.add(Dense(64, activation='tanh', input_shape=(n_obs,)))
	actor.add(Dense(64, activation='tanh'))
	actor.add(Dense(32, activation='tanh'))
	actor.add(Dense(n_action))
	actor.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=lr))

	return actor

def plot_evaluation(df_eval, window, path):
	fig, ax = plt.subplots(1, figsize=(15,8))
	df_eval = df_eval.astype("float32")
	sns.lineplot(x="episode", y="cum_rewards", data=df_eval, ax=ax)
	fig.savefig("{}/eval.png".format(path))

def plot_train(df_train, window, path):
	fig, ax = plt.subplots(1, figsize=(15,8))
	df_train["train cum reward average"] = df_train.train_cum_reward.rolling(window=window).mean()
	sns.lineplot(x="episode", y="train cum reward average", data=df_train, ax=ax)
	fig.savefig("{}/train_cum_reward.png".format(path))

def build_critic(n_obs, lr):
	critic = Sequential()
	critic.add(Dense(64, activation='relu', input_shape=(n_obs,)))
	critic.add(Dense(64, activation='relu'))
	critic.add(Dense(32, activation='relu'))
	critic.add(Dense(1))
	critic.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=lr))

	return critic

def init_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument("--agent", default="PIP")                          # agent name (CACLA, TD3, PIP)
	parser.add_argument("--env", default="SparsePendulum-v0")              # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)                     # Sets Gym, TF and Numpy seeds
	parser.add_argument("--start_episodes", default=0, type=int)       # Episodes initial random agent is used
	parser.add_argument("--eval_freq", default=1000, type=int)             # How often (time steps) we evaluate
	parser.add_argument("--episodes", default=50000, type=int)        # Max time steps to run environment
	parser.add_argument("--n_steps", default=50, type=int)   		       # 1 episode how many steps in the environment
	parser.add_argument("--learning_rate", default=0.5, type=float)
	parser.add_argument("--expl_noise", default=0.5, type=float)           # CACLA and TD3 std of Gaussian exploration noise
	parser.add_argument("--buffer_size", default=1000000, type=int)            # replay buffer size
	parser.add_argument("--batch_size", default=256, type=int)             # Batch size for both actor and critic
	parser.add_argument("--save_replay", action="store_true")             # Batch size for both actor and critic
	parser.add_argument("--gamma", default=0.95, type=float)                           # Discount factor
	parser.add_argument("--tau", default=0.005, type=float)                # TD3 Target network update rate
	parser.add_argument("--agent_noise", default=0.2, type=float)          # TD3 Noise added to target agent during critic update
	parser.add_argument("--noise_clip", default=0.5, type=float)           # TD3 Range to clip target agent noise
	parser.add_argument("--agent_freq", default=2, type=int)               # TD3 Frequency of delayed agent updates
	parser.add_argument("--plan_length", default=5, type=int)              # PIP plan's length
	parser.add_argument("--c1", default=1.4, type=float)                   # PIP PSO c1 coef
	parser.add_argument("--c2", default=1.4, type=float)                   # PIP PSO c2 coef
	parser.add_argument("--w", default=0.7, type=float)                    # PIP PSO w coef
	parser.add_argument("--k", default=20, type=int)               		   # PIP PSO number of neighbour
	parser.add_argument("--particles", default=50, type=int)               # PIP PSO number of particles
	parser.add_argument("--iterations", default=3, type=int)               # PIP PSO number of iterations
	parser.add_argument("--name", default="default", required=True)  	   # Save model and optimizer parameters
	parser.add_argument("--world_model", default="pendulum")               # World model directory for pendulum env
	parser.add_argument('--rbf', nargs='*', type=int)               	   # Radial basis units applied to networks
	parser.add_argument('--rbf_space', nargs='*', type=int)
	parser.add_argument('--rbf_concatenate', default=False, type=bool)
	parser.add_argument('--start_train_wm', default=10, type=int)

	args = parser.parse_args()
	return args

def create_rbf_space(val, shape):
	return gym.spaces.Box(-val, val, shape, dtype=np.float32)

if __name__ == "__main__":
	args = init_arguments()
	file_name = f"{args.agent}_{args.env}_expl_noise_{args.expl_noise}_agent_noise_{args.agent_noise}_noise_clip_{args.noise_clip}_tau_{args.tau}_steps_{args.n_steps}_buffer_{args.buffer_size}_seed_{args.seed}"
	model_path = Path(f"./results/{args.name}")

	if not model_path.exists():
		os.makedirs(model_path)

	env = gym.make(args.env)
	# Set seeds
	env.seed(args.seed)
	tf.random.set_seed(args.seed)
	np.random.seed(args.seed)

	transformer = None
	if args.rbf:
		rbf_space =  env.observation_space
		if args.rbf_space:
			rbf_space = create_rbf_space(
				np.array(args.rbf_space),
				env.observation_space.shape
			)

		if args.rbf_concatenate:
			transformer = RadialBasisTransformerConcat(rbf_space, args.rbf)
		else:
			transformer = RadialBasisTransformer(rbf_space, args.rbf)

		with open(f"{model_path}/transformer.pkl", "wb") as f:
			pickle.dump(transformer, f)

	n_obs = transformer.rbf_dim if transformer else env.observation_space.shape[0]
	n_action = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	critic = build_critic(n_obs, args.learning_rate)
	if transformer:
		critic = NetworkWithTransformer(critic, transformer)

	# Initialize agent
	if args.agent == "PIP":
		copy_tree(f"world_models/{args.world_model}", f"{model_path}/world_model")
		world_model = WorldModel.load(model_path)
		planner = PSOPlanner(
			env.action_space,
			world_model,
			critic,
			length=args.plan_length,
			gamma=args.gamma,
			particles=args.particles,
			iterations=args.iterations,
			c1=args.c1,
			c2=args.c2,
			w=args.w,
			k=args.k,
		)

		agent = PIP(planner, critic, gamma=args.gamma)
	elif args.agent == "CACLA":
		actor = build_actor(n_obs, n_action, args.learning_rate)
		if transformer:
			actor = NetworkWithTransformer(actor, transformer)
		agent = CACLA(
			actor,
			critic,
			env.action_space,
			gamma=args.gamma,
			exploration_noise_std=args.expl_noise
		)

	dump_dir = model_path if args.save_replay else None
	replay_buffer = ReplayBuffer(n_obs, n_action, max_size=args.buffer_size, dump_dir=dump_dir)

	# Evaluate untrained agent
	evaluations = [evaluate_agent(agent, args.env, args.seed, transformer=transformer)]
	train_rewards = []

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	df_eval = pd.DataFrame(columns =  ["episode", "cum_rewards", "eval"])

	for t in tqdm(range(args.episodes)):
		state, done = env.reset(), False
		episode_reward = 0

		episode_buffer = []

		while not done:
			if t < args.start_episodes:
				action = env.action_space.sample()
			else:
				action = agent.behavior_policy(state.reshape(1,-1))

			next_state, reward, done, _ = env.step(action.reshape(-1))
			done_bool = 0

			state_temp = state
			next_state_temp = next_state
			if transformer:
				state_temp = transformer.transform(state.reshape(1,-1))
				next_state_temp = transformer.transform(next_state.reshape(1,-1))

			episode_buffer.append([state_temp, action, next_state_temp, reward, done])

			state = next_state
			episode_reward += reward

		train_rewards.append(episode_reward)

		for row in episode_buffer:
			replay_buffer.add(*row, episode_reward)

		state_transition = replay_buffer.sample()
		agent.train(state_transition)

		# Train world model after n episodes
		if t >= args.start_train_wm:
			state_transition_wm = replay_buffer.boltzmann_sampling(1)
			world_model.train(state_transition_wm)

		# Evaluate and save data frames
		if (t + 1) % args.eval_freq == 0:
			reward = evaluate_agent(agent, args.env, args.seed, args.n_steps, transformer=transformer)
			df_eval = df_eval.append({"episode": t+1, "cum_rewards": reward}, ignore_index=True)
			df_eval.to_pickle(f"{model_path}/df_eval.pkl")

			df_train = pd.DataFrame()
			df_train["episode"] = np.arange(len(train_rewards))
			df_train["train_cum_reward"] = train_rewards
			df_train.to_pickle(f"{model_path}/df_train.pkl")
			agent.save(model_path, eps=t)

			plot_evaluation(df_eval, args.eval_freq, model_path)
			plot_train(df_train, args.eval_freq, model_path)

	if args.world_model == "pendulum":
		visualize_pendulum(env, df_train["train_cum_reward"].max(), model_path, transformer)
