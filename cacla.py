import tensorflow as tf
import numpy as np
import pickle
from pathlib import Path
from network_with_transformer import NetworkWithTransformer

class CACLA:

    def __init__(self,
                actor,
                critic,
                action_space,
                gamma=0.9,
                exploration_noise_std=0.1):

        self.actor = actor # seq model
        self.critic = critic # seq model
        self.action_space = action_space
        self.exploration_noise_std = exploration_noise_std
        self.gamma = gamma

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["actor"]
        del state["critic"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.actor = None
        self.critic = None

    def __call__(self, state):
        """ Called during evaluation. It fully exploits the policy."""
        
        return self.actor(state)

    def train(self, replay_buffer):
        states, actions, rewards , values, next_state_values = replay_buffer.sample()
        v_targets = rewards + self.gamma * next_state_values

        critic_loss = self.critic.fit(x=states, y=v_targets, verbose=0).history["loss"][0]

        index = (next_state_values > values).reshape(-1) # only update action next state leads to good values

        actor_loss = None
        if index.any():
            actor_loss = self.actor.fit(x=states[index], y=actions[index], verbose=0).history["loss"][0]

        return actor_loss, critic_loss

    def behavior_policy(self, state):
        """ Policy used for exploration. CACLA adds a gaussian noise as its exploration method."""

        action = self.actor(state)
        noise = np.random.normal(scale=self.exploration_noise_std, size=(action.shape))
        action += noise
        action = np.clip(action, self.action_space.low, self.action_space.high)

        return action

    def save(self, dir="./logs"):
        with open(f"{dir}/agent.pkl", 'wb') as f:
            pickle.dump(self, f)

        self.actor.save(f"{dir}/actor.h5")
        self.critic.save(f"{dir}/critic.h5")

    @staticmethod
    def load(dir):
        with open(f"{dir}/agent.pkl", 'rb') as f:
            cacla = pickle.load(f)

        actor = tf.keras.models.load_model(f"{dir}/actor.h5")
        critic = tf.keras.models.load_model(f"{dir}/critic.h5")

        transformer = None
        transformer_path = Path(f"{dir}/transformer.pkl")
        if transformer_path.exists():
            with open(transformer_path, 'rb') as fp:
                transformer = pickle.load(fp)
            actor = NetworkWithTransformer(actor, transformer)
            critic = NetworkWithTransformer(critic, transformer)

        cacla.actor = actor
        cacla.critic = critic

        return cacla
