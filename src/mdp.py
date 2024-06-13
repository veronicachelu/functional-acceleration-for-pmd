import numpy as np
import yaml
import os
from src import runner
import importlib

runner = importlib.reload(runner)


class MDP(object):
    """
    Represents a Markov Decision Process (MDP).
    """
    def __init__(self, rho, P, R, seed=0, rng=None, mdp_config=None):
        """
        Initialize the MDP.

        Args:
            rho: Distribution over the initial state.
            P: Transition probabilities.
            R: Reward function.
            seed: Seed for random number generation (default: 0).
            rng: Random number generator (default: None).
            mdp_config: Additional MDP configuration (default: None).
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed) if rng is None else rng
        self.rho = rho
        self.n_states, self.n_actions, _ = P.shape
        self.P = P
        self.R = R

        # Compute `r(s,a)` from `r(s,a,s')`
        self.r = np.einsum('sap,sap->sa', self.P, self.R)

        # Combine provided MDP config with default config
        self.mdp_config = {
            "rho": self.rho,
            "P": self.P,
            "R": self.R,
            "seed": self.seed
        }
        if mdp_config is not None:
            self.mdp_config = {**self.mdp_config, **mdp_config}

    def save_config(self, experiment_name):
        """
        Save the MDP configuration to a YAML file.

        Args:
            experiment_name: Name of the experiment.
        """
        # Define the directory for saving the config file
        logs_dir = os.path.join(os.path.dirname(__file__), "..", "logs", experiment_name)
        os.makedirs(logs_dir, exist_ok=True)

        self.config_path = os.path.join(logs_dir, 'mdp_config.yaml')
        with open(self.config_path, 'w') as file:
            yaml.dump(self.mdp_config, file)

        print(f"MDP configuration saved successfully at: {self.config_path}")
