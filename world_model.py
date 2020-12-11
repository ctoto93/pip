from tensorflow.keras import models
import numpy as np
import pickle
import tensorflow as tf

class WorldModel:

    def __init__(self, model, scaler, transformer=None):
        self.model = model
        self.scaler = scaler
        self.transformer = transformer

    def __call__(self, state, action):
        state_action = self.to_neural_input(state, action)
        neural_state = self.model(state_action)
        return self.to_actual_state(neural_state)

    def to_neural_input(self, state, action):
        input = np.concatenate((state, action), axis=1)
        input = self.scaler.transform(input)
        return input

    def to_actual_state(self, neuron_state):
        n_particle, n_obs = neuron_state.shape
        n_action = self.scaler.scale_.shape[0] - n_obs

        dummy_action = np.zeros((n_particle, n_action))
        scaler_in = np.hstack((neuron_state, dummy_action))
        scaler_out = self.scaler.inverse_transform(scaler_in)
        state = scaler_out[:,:n_obs]
        return state

    def save(self, dir):
        self.model.save(f"{dir}/world_model/model.h5")
        with open(f"{dir}/world_model/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)

    @staticmethod
    def load(dir):
        model = models.load_model(f"{dir}/world_model/model.h5", compile=False)
        with open(f"{dir}/world_model/scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)

        return WorldModel(model, scaler)
