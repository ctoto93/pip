import numpy as np
from sklearn.preprocessing import MinMaxScaler

class RadialBasisTransformer:
    def __init__(self, observation_space, n_rbf=[], inverse_model_path=''):
        self.observation_space = observation_space
        self.env_dim = observation_space.shape[0]
        self.n_rbf = self.check_n_rbf(n_rbf)
        self.rbf_dim = np.prod(self.n_rbf)
        self.inverse_model = None

        self.init()

    def check_n_rbf(self, val):
        if isinstance(val, int):
            val = [val] * self.env_dim

        if len(val) != self.env_dim:
            raise Exception("n_rbf dimmensions not same with observation space")

        return np.asarray(val)

    def init(self):
        linspaces = []
        sigma = []
        for i in range(self.env_dim):
            low = self.handle_space_inf(self.observation_space.low[i])
            high = self.handle_space_inf(self.observation_space.high[i])
            linspace = np.linspace(low, high, self.n_rbf[i])
            s = 0.5 * (np.abs(low) + high) / self.n_rbf[i]
            linspaces.append(linspace)
            sigma.append(s)

        self.meshgrid = np.asarray(np.meshgrid(*linspaces))
        self.sigma = np.array(sigma)
        self.init_scaler()

    def init_scaler(self):
        self.scaler = MinMaxScaler()
        data = np.array([
            self.observation_space.low,
            self.observation_space.high
        ])
        self.scaler.fit(data)

    def handle_space_inf(self, space):
        if not np.isinf(space):
            return space
        return space * self.sigma

    def transform(self, A):
        D = np.zeros((A.shape[0], self.rbf_dim))
        for i in range(self.env_dim):
            X_mu = np.broadcast_to(self.meshgrid[i].flatten(), (A.shape[0], self.rbf_dim))
            X = np.broadcast_to(A[:,i].reshape(-1,1), (A.shape[0], self.rbf_dim))
            D += (X - X_mu)**2 / (2 * (self.sigma[i]**2))

        return np.exp(-D)

    def inverse_transform(self, A):
        neural_output = self.inverse_model.predict(A)
        return self.scaler.inverse_transform(neural_output)
