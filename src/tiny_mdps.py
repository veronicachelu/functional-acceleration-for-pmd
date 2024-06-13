import numpy as np
from src.discounted_mdp import DiscountedMDP

examples = {
    0: (2, 0.9,
        [0.06, 0.38, -0.13, 0.64],
        [[0.01, 0.99], [0.92, 0.08], [0.08, 0.92], [0.70, 0.30]]),

    1: (2, 0.9,
        [0.88, -0.02, -0.98, 0.42],
        [[0.96, 0.04], [0.19, 0.81], [0.43, 0.57], [0.72, 0.28]]),

    3: (2, 0.9,
        [-0.45, -0.1, 0.5, 0.5],
        [[0.7, 0.3], [0.99, 0.01], [0.2, 0.8], [0.99, 0.01]]),

    4: (3, 0.8,
        [-0.1, -1., 0.1, 0.4, 1.5, 0.1],
        [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.05, 0.95], [0.25, 0.75], [0.3, 0.7]]),

}


def get_dynamics(n_states, n_actions, P, R):
    """
    Convert dynamics from paper's format to ndarray format.

    Args:
        n_states (int): Number of states.
        n_actions (int): Number of actions.
        P (list): Transition probabilities encoded as a list.
        R (list): Rewards encoded as a list.

    Returns:
        tuple: Tuple containing transition probabilities (p) and rewards (r) as ndarrays.
    """
    p = np.zeros((n_states, n_actions, n_states))
    r = np.zeros((n_states, n_actions, n_states))
    for i in range(n_states):
        for j in range(n_actions):
            for k in range(n_states):
                # P(s_k | s_i, a_j ) = P[i x |A| + j][k]
                p[i, j, k] = P[i * n_actions + j][k]
                # r(s_i, aj) = r[i x |A| + j]
                r[i, j, k] = R[i * n_actions + j]
    return p, r

def load_example_config(eg_no):
    """
    Load configuration for a specific example.

    Args:
        eg_no (int): Example number.

    Returns:
        dict: Configuration dictionary for the specified example.
    """
    n_states = 2
    n_actions, gamma, R, P = examples[eg_no]
    p, r = get_dynamics(n_states, n_actions, P, R)
    rho = np.ones(n_states)/n_states
    config_mdp = {"gamma": gamma,
                  "rho": rho,
                  "P": p,
                  "R": r}
    return config_mdp

def load_example(eg_no):
    """
    Load an example MDP.

    Args:
       eg_no (int): Example number.

    Returns:
       DiscountedMDP: Discounted MDP instance for the specified example.
    """
    config_mdp = load_example_config(eg_no)
    return DiscountedMDP(**config_mdp)