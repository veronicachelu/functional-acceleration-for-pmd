import importlib
import numpy as np
from scipy import linalg
from src.markovchain import MarkovChain


class MRP(MarkovChain):
    """
    Markov reward process is Markov chain with a state-dependent reward function.
    """

    def __init__(self, rho, P, R, gamma):
        """
        Initialize the Markov reward process.

        Args:
            rho: Initial distribution.
            P: Transition matrix.
            R: Reward function.
            gamma: Discount factor.
        """
        super().__init__(rho, P, gamma)
        self.R = R
        self.S, _ = P.shape

    def __iter__(self):
        """
        Make the MRP iterable.

        Returns:
            Iterator containing the MRP parameters.
        """
        return iter([self.rho, self.P, self.gamma, self.R])

    #___________________________________________________________________________
    # Simulation

    def run(self):
        """
        Simulate the MRP.

        Yields:
            State, reward, next state.
        """
        s = self.start()
        while True:
            sp = self.step(s)
            yield s, self.R[s], sp
            s = sp

    #___________________________________________________________________________
    # Important functionns

    def J(self):
        """
        Expected value of the MRP.
        """
        return self.rho @ self.get_V()

    def get_cond_no(self):
        """
        Compute the condition number of the transition matrix.
        """
        return self.get_cond_number()

    def get_spectral_no(self):
        """
        Compute the spectral number of the transition matrix.
        """
        return self.get_spectral_number()

    def get_V(self):
        """
        Compute the value function.
        """
        return self.solve(self.R)
