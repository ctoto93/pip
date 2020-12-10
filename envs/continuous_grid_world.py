import gym
from gym import spaces
import numpy as np
from envs.transformer import RadialBasisTransformer

class ContinuousGridWorldEnv(gym.Env):
    def __init__(self, width, height, max_step, goal_x, goal_y, goal_side, radial_basis_fn, x_dim, y_dim, cumulative):

        self.width = width
        self.height = height
        self.max_step = max_step
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_side = goal_side
        self.radial_basis_fn = radial_basis_fn
        self.cumulative = cumulative

        obs_low = np.array([0., 0.], dtype=np.float32)
        obs_high = np.array([width, height], dtype=np.float32)
        act_high = np.array([max_step, max_step])
        self.action_space = spaces.Box(low=-act_high, high=act_high, dtype=np.float32)
        self.original_observation_space = spaces.Box(low=-obs_low, high=obs_high, dtype=np.float32)

        if self.radial_basis_fn:
            n_dim = x_dim * y_dim
            obs_low = np.array([0.] * n_dim, dtype=np.float32)
            obs_high = np.array([1.] * n_dim, dtype=np.float32)
            self.observation_space = spaces.Box(low=-obs_low, high=obs_high, dtype=np.float32)
            self.transformer = RadialBasisTransformer(self.original_observation_space, [x_dim, y_dim], 'world_models/continuous_grid_world/inverse_rbf.h5')
        else:
            self.observation_space = self.original_observation_space

    @staticmethod
    def calculate_state(state, action, max_step, observation_space):
        action = np.clip(action, -max_step, max_step).reshape(state.shape)
        state += action
        return np.clip(state, observation_space.low, observation_space.high)

    def create_world_model(self):
        def world_model(state_action):
            state = state_action[:,:-self.action_space.shape[0]]
            action = state_action[:,-self.action_space.shape[0]:]
            state = ContinuousGridWorldEnv.calculate_state(state, action, self.max_step, self.original_observation_space)
            return state
        return world_model

    def step(self, action):
        self.state = ContinuousGridWorldEnv.calculate_state(self.state, action, self.max_step, self.original_observation_space)

        reward = 0
        done = False

        if self.in_box():
            reward = 1
            if not self.cumulative:
                done = True


        if self.radial_basis_fn:
            return self.radial_basis_state(), np.array([reward]), done, {}

        return self.state, np.array([reward]), done, {}

    def reset(self):
        self.state = self.original_observation_space.sample()

        if self.radial_basis_fn:
            return self.radial_basis_state()

        return self.state

    def render(self, mode='human', close=False):
        print(f'state: {self.state}')

    def in_box(self):
        x, y = self.state

        return ((x >= self.goal_x and x <= (self.goal_x + self.goal_side))
                and (y >= self.goal_y and y <= (self.goal_y + self.goal_side)))

    def radial_basis_state(self):
        return self.transformer.transform(self.state.reshape(1,-1))[0]
