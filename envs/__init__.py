from gym.envs.registration import register
from envs.sparse_pendulum import REWARD_MODE_BINARY

register(
    id='SparsePendulum-v0',
    entry_point='envs.sparse_pendulum:SparsePendulum',
    max_episode_steps=100,
    kwargs={
         'max_speed': 8.0,
         'max_torque': 2.0,
         'reward_angle_limit': 2.0,
         'reward_speed_limit': 1.0,
         'balance_counter': 5,
         'reward_mode': REWARD_MODE_BINARY,
         'cumulative': True
    }

)

register(
    id='ContinuousGridWorld-v0',
    entry_point='envs.continuous_grid_world:ContinuousGridWorldEnv',
    max_episode_steps=15,
    kwargs={
         'width': 1,
         'height': 1,
         'max_step': 0.1,
         'goal_x': 0.45,
         'goal_y': 0.45,
         'goal_side': 0.1,
         'cumulative': True
    }
)
