import numpy as np
from scipy import linalg
from src.utils import sample


class MarkovChain:
    """
    Represents a gamma-discounted Markov chain.
    """
    def __init__(self, rho, P, gamma, seed=None, rng=None):
        """
        Initialize the Markov chain.

        Args:
            rho: Initial distribution.
            P: Transition matrix.
            gamma: Discount factor.
        """
        self.S = rho.shape[0]
        self.rho = rho
        self.P = P
        self.gamma = gamma
        self.seed = int(seed) if seed is not None else np.random.randint(1e5)
        self.rng = np.random.RandomState(seed) if rng is None else rng

    #___________________________________________________________________________
    # Simulation

    def run(self):
        "Simulate the Markov chain with (1-gamma)-resetting dynamics"
        s = self.start()
        while True:
            sp = self.step(s)
            yield s, sp
            s = sp

    def start(self):
        """
        Get the initial state.
        """
        return sample(self.rho)

    def step(self, s):
        """
       Move to the next state based on the current state.

       Args:
           s: Current state.

       Returns:
           Next state.
       """
        if self.rng.uniform(0,1) <= 1-self.gamma:
            return self.start()
        return sample(self.P[s,:])

    #___________________________________________________________________________
    # Important quantities

    def get_d_pi(self, rho=None):
        """
        Compute the stationary distribution.

        Args:
            rho: Initial distribution.

        Returns:
            Stationary distribution.
        """
        if rho is None:
            rho = self.rho
        return (1-self.gamma) * self.solve_t(rho)   # note the transpose

    @property
    def M(self):
        """
        Transition matrix with discount factor.
        """
        return np.eye(self.S) - self.gamma * self.P

    @property
    def M_inv(self):
        """
        Transition matrix with discount factor.
        """
        return np.linalg.inv(self.M)
    #___________________________________________________________________________
    # Operators

    def solve(self, b, transpose=False):
        """
        Solve the linear system (I - gamma * P) * x = b or xᵀ * (I - gamma * P) = b.

        Args:
            b: Right-hand side vector.
            transpose: Whether to transpose the matrix before solving.

        Returns:
            Solution vector.
        """
        matrix = self.M.T if transpose else self.M
        try:
            x = linalg.solve(matrix, b)
        except ValueError as err:
            print(f"linalg.solve(matrix, b) failed with error: {err}")
            x = np.zeros_like(b)
        return x

    def successor_representation(self, normalize=False):
        return linalg.solve(self.M, np.eye(self.S) * ((1-self.gamma) if normalize else 1))
    #        return np.linalg.inv(self.M)


    def get_cond_number(self):
        """
        Compute the condition number of the transition matrix.

        Returns:
            Condition number.
        """

        eig = np.sort(np.linalg.eigvals(self.M))
        cond_no = np.abs(eig[1:].max() / eig[1:].min())
        return cond_no

    def get_spectral_number(self):
        """
        Compute the spectral number of the transition matrix.

        Returns:
            Spectral number.
        """
        eig = np.sort(np.linalg.eigvals(self.M))
        spectral_no = np.abs(eig[1:]).max()
        return spectral_no

    def solve_t(self, b):
        """
        Solve the linear system xᵀ * (I - gamma * P) = b.

        Args:
            b: Right-hand side vector.

        Returns:
            Solution vector.
        """
        try:
            x = linalg.solve(self.M.T, b)
        except ValueError as err:
            print(f"linalg.solve(self.M.T, b) fail with err: {err}")
            x = np.zeros_like(self.M.S, self.M.n_actions)
        return x
