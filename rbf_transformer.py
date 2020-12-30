import numpy as np
from sklearn.preprocessing import MinMaxScaler

def check_space_inf(space):
    if np.isinf(space.low).any() or np.isinf(space.high).any():
        raise Exception("Observation space contains infinite value")

class Base:
    def __init__(self, observation_space, n_rbf=[]):
        check_space_inf(observation_space)
        self.observation_space = observation_space
        self.env_dim = observation_space.shape[0]
        self.n_rbf = np.array(n_rbf)

        self.sigma = self.create_sigma()
        self.mu, self.rbf_dim = self.create_mu()


    def create_sigma(self):
        obs_dim = self.observation_space.shape[0]
        sigma = []
        for i in range(obs_dim):
            low = self.observation_space.low[i]
            high = self.observation_space.high[i]
            s = 0.5 * (np.abs(low) + high) / self.n_rbf[i]
            sigma.append(s)

        return np.array(sigma)

    def calculate_distance(self, X, X_mu, sigma):
        return (X - X_mu)**2 / (2 * (sigma**2))

    def transform(self, A):
        pass

    def create_mu(self):
        pass

class RadialBasisTransformer(Base):

    def create_mu(self):
        obs_dim = self.observation_space.shape[0]
        linspaces = []
        for i in range(obs_dim):
            low = self.observation_space.low[i]
            high = self.observation_space.high[i]
            linspace = np.linspace(low, high, self.n_rbf[i])
            linspaces.append(linspace)

        rbf_dim = np.prod(self.n_rbf)
        mu = np.asarray(np.meshgrid(*linspaces)).reshape(obs_dim, rbf_dim)

        return mu, rbf_dim

    def transform(self, A):
        D = np.zeros((A.shape[0], self.rbf_dim))
        for i in range(self.env_dim):
            X_mu = np.broadcast_to(self.mu[i], (A.shape[0], self.rbf_dim))
            X = np.broadcast_to(A[:,i].reshape(-1,1), (A.shape[0], self.rbf_dim))
            D += self.calculate_distance(X, X_mu, self.sigma[i])

        return np.exp(-D)

class RadialBasisTransformerConcat(Base):
    def create_mu(self):
        obs_dim = self.observation_space.shape[0]
        linspaces = []
        for i in range(obs_dim):
            low = self.observation_space.low[i]
            high = self.observation_space.high[i]
            linspace = np.linspace(low, high, self.n_rbf[i])
            linspaces.append(linspace)

        mu = np.concatenate(linspaces)
        rbf_dim = np.sum(self.n_rbf)

        return mu, rbf_dim

    def transform(self, A):
        D = np.zeros((A.shape[0], self.rbf_dim))
        for i in range(self.env_dim):
            X = np.broadcast_to(A[:,i].reshape(-1,1), (A.shape[0], self.rbf_dim))
            D += self.calculate_distance(X, self.mu, self.sigma[i])

        return np.exp(-D)
