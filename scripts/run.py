import argparse
import pickle
import tensorflow as tf
import gym
import envs
import numpy as np
from pip import PIP

def init_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", default="PIP")
    parser.add_argument("--env", default="SparsePendulum-v0")
    parser.add_argument("--load_model", required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = init_arguments()
    env = gym.make(args.env)
    agent = PIP.load(f"results/{args.load_model}")
    agent.planner.env = env.unwrapped
    env.reset()
    state = np.array([np.radians(180), 0])
    env.unwrapped.state = state

    state = env.unwrapped._get_obs()
    done = False
    step = 0
    while not done:
        env.render()
        action = agent(state.reshape(1,-1))
        state, reward, done, _ = env.step(action)
        step += 1
        print(f'step {step} reward {reward} done {done}')
        if reward:
            break
