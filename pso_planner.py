import collections
import numpy as np
import tensorflow as tf
import pyswarms as ps
from world_model import WorldModel
import pickle

class PSOPlanner:
    def __init__(self, action_space, world_model, critic, particles=50, iterations=3, length=5 , gamma=0.95, c1=1.4, c2=1.4, w=0.7, k=20):
        self.action_space = action_space
        self.world_model = world_model
        self.critic = critic
        self.particles = particles
        self.iterations = iterations
        self.length = length
        self.gamma = gamma
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.k = k
        self.history = []

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["world_model"]
        del state["history"]
        del state["critic"]
        del state["optimizer"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.world_model = None
        self.critic = None
        self.history = []

    def __str__(self):
        return f"particles_{self.particles}_iterations_{self.iterations}_c1_{self.c1}_c2_{self.c2}_w_{self.w}_k_{self.k}_p_{self.p}"

    def options(self):
        return {
            'c1': self.c1,
            'c2': self.c2,
            'w': self.w,
            'k': self.k,
            'p': 2
        }

    def new_optimizer(self):
        optimizer = ps.single.LocalBestPSO(
                n_particles=self.particles,
                dimensions=self.dimensions(),
                options=self.options(),
                bounds=self.bounds())

        optimizer.bh = ps.backend.handlers.BoundaryHandler('nearest')

        return optimizer

    def bounds(self):
        low = self.action_space.low
        high = self.action_space.high

        min_bound = np.tile(low, self.length)
        max_bound = np.tile(high, self.length)
        return (min_bound, max_bound)

    def action_length(self):
        return self.action_space.shape[0]

    def dimensions(self):
        return self.length * self.action_length()

    def generate(self, state):
        self.state = state
        self.optimizer = self.new_optimizer()
        best_reward, best_plan = self.optimizer.optimize(self.pso_cost_func,
                                    iters=self.iterations,
                                    verbose=False)
        self.history.append({
            "state": state,
            "cost_history": self.optimizer.cost_history,
            "plan": best_plan
        })

        self.plans = self.optimizer.swarm.position
        self.plans_value = self.optimizer.swarm.current_cost

        return best_plan.reshape(self.length, -1)

    def pso_cost_func(self, x):
        value = np.zeros((self.particles, 1))
        state = np.tile(self.state, (self.particles, 1))

        predicted_states = []

        for i in range(self.length):
            from_idx = i * self.action_length()
            to_idx = (i+1) * self.action_length()
            action = x[:,from_idx:to_idx].reshape(self.particles, self.action_length())
            state = self.world_model(state, action).numpy()
            value += self.critic(state)

        cost = value.numpy().reshape(-1) * -1 # min the cost instead of max rewards since pyswarms can only minimize

        return np.squeeze(cost)

    def save(self, dir="./results"):
        with open(f"{dir}/planner.pkl", 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(dir):
        with open(f"{dir}/planner.pkl", 'rb') as f:
            planner = pickle.load(f)

        planner.world_model = WorldModel.load(dir)

        return planner
