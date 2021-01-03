from tensorflow import keras
from tensorflow.keras import optimizers, losses
import tensorflow as tf
import numpy as np
import pickle as pickle
from pso_planner import PSOPlanner
from network_with_transformer import NetworkWithTransformer
from pathlib import Path

class PIP:

    def __init__(self,
                planner,
                critic,
                gamma=0.95):

        self.planner = planner
        self.critic = critic
        self.gamma = gamma

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["critic"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.critic = None

    def __call__(self, state):
        """ PIP's target and behavior policy are the same"""

        return self.behavior_policy(state)

    def train(self, replay_buffer):
        state, action, next_state, reward, _ = replay_buffer.sample()
        next_state_value = self.critic(next_state).numpy()
        v_target = reward + self.gamma * next_state_value
        critic_loss = self.critic.fit(x=state, y=v_target, verbose=0).history["loss"][0]
        return critic_loss

    def behavior_policy(self, state):
        plan = self.planner.generate(state)
        return plan[:1].reshape(1,-1)

    def save(self, dir="./results", eps=None):
        fname = f"critic_{eps}.h5" if eps else "critic.h5"
        self.critic.save(f"{dir}/{fname}")
        self.planner.save(dir)
        with open(f"{dir}/agent.pkl", 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(dir):
        with open(f"{dir}/agent.pkl", 'rb') as f:
            pip = pickle.load(f)

        pip.planner = PSOPlanner.load(dir)

        critic = tf.keras.models.load_model(f"{dir}/critic.h5")

        transformer = None
        transformer_path = Path(f"{dir}/transformer.pkl")
        if transformer_path.exists():
            with open(transformer_path, 'rb') as fp:
                transformer = pickle.load(fp)
            critic = NetworkWithTransformer(critic, transformer)

        pip.critic = critic
        pip.planner.critic = critic

        return pip
