import gym
from gym import spaces
import numpy as np

class ContinuousGridWorldEnv(gym.Env):
    def __init__(self, width, height, max_step, goal_x, goal_y, goal_side, cumulative):

        self.width = width
        self.height = height
        self.max_step = max_step
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.goal_side = goal_side
        self.cumulative = cumulative

        obs_low = np.array([0., 0.], dtype=np.float32)
        obs_high = np.array([width, height], dtype=np.float32)
        act_high = np.array([max_step, max_step])
        self.action_space = spaces.Box(low=-act_high, high=act_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_low, high=obs_high, dtype=np.float32)

    @staticmethod
    def calculate_state(state, action, max_step, observation_space):
        action = np.clip(action, -max_step, max_step).reshape(state.shape)
        state += action
        return np.clip(state, observation_space.low, observation_space.high)

    def create_world_model(self):
        def world_model(state_action):
            state = state_action[:,:-self.action_space.shape[0]]
            action = state_action[:,-self.action_space.shape[0]:]
            state = ContinuousGridWorldEnv.calculate_state(state, action, self.max_step, self.observation_space)
            return state
        return world_model

    def step(self, action):
        self.state = ContinuousGridWorldEnv.calculate_state(self.state, action, self.max_step, self.observation_space)

        reward = 0
        done = False

        if self.in_box():
            reward = 1
            if not self.cumulative:
                done = True

        return self.state, np.array([reward]), done, {}

    def reset(self):
        self.state = self.observation_space.sample()
        return self.state

    def render(self, mode='human', close=False):
        print(f'state: {self.state}')

    def in_box(self):
        x, y = self.state

        return ((x >= self.goal_x and x <= (self.goal_x + self.goal_side))
                and (y >= self.goal_y and y <= (self.goal_y + self.goal_side)))
