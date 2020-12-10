import numpy as np

class CACLA(Agent):

    def __init__(self,
                env,
                actor,
                critic,
                n_steps=5,
                gamma=0.9,
                exploration_noise_std=0.1,
                target_noise_clip=0.5):

        self.env = env
        self.actor = actor # seq model
        self.critic = critic # seq model
        self.exploration_noise_std = exploration_noise_std
        self.gamma = gamma
        self.target_noise_clip = target_noise_clip
        self.runner = Runner(env, self, n_steps, gamma)

    def train(self, replay_buffer):
        states, actions, rewards , values, next_state_values = replay_buffer.sample()
        v_targets = rewards + self.gamma * next_state_values

        critic_loss = self.critic.fit(states, v_targets, verbose=0).history["loss"][0]

        index = (next_state_values > values).reshape(-1) # only update action next state leads to good values

        actor_loss = None
        if index.any():
            actor_loss = self.actor.fit(states[index], actions[index], verbose=0).history["loss"][0]

        return actor_loss, critic_loss


    def call(self, state):
        return self.actor(state)

    def select_action(self, state):
        action = self.actor(state)
        noise = np.random.normal(scale=self.exploration_noise_std, size=(action.shape))
        action += noise
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

        return action

    def save(self, dir="./logs"):
        self.actor.save("{}/actor_model_eps_{}.h5".format(dir, self.current_episode))
        self.critic.save("{}/critic_model_eps_{}.h5".format(dir, self.current_episode))
