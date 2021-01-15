import numpy as np
import csv

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6), batch_size=256, dump_dir=None):
        self.max_size = max_size
        self.batch_size = batch_size
        self.ptr = 0
        self.size = 0
        self.dump_dir = dump_dir

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.cum_reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done, cum_reward=0):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        self.cum_reward[self.ptr] = cum_reward

        if self.dump_dir:
            with open(f"{self.dump_dir}/replay_buffer.csv", mode='a') as csv_file:
                writer = csv.writer(csv_file)
                row = np.concatenate((state.reshape(-1), action.reshape(-1), next_state.reshape(-1), reward, done), axis=None)
                writer.writerow(row)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def get_by_index(self, ind):
        return (
            self.state[ind],
            self.action[ind],
            self.next_state[ind],
            self.reward[ind],
            self.not_done[ind]
        )

    def sample(self):
        ind = np.random.randint(0, self.size, size=self.batch_size)

        return self.get_by_index(ind)

    def boltzmann_sampling(self, beta):
        r_min = self.cum_reward[:self.size].min()
        r_max = self.cum_reward[:self.size].max()
        r = self.cum_reward[:self.size].flatten()
        values = np.exp(beta * (r - r_min)/(r_max - r_min))
        total = values.sum()
        weight = values/total
        ind = np.random.choice(np.arange(self.size), self.batch_size, p=weight)

        return self.get_by_index(ind)
