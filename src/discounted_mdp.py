import numpy as np
import importlib
from src.utils import sample, random_dist
from scipy import linalg
from itertools import product
from src.mrp import MRP
from src import runner
import src.mdp
src.mdp = importlib.reload(src.mdp)
from src.mdp import MDP

runner = importlib.reload(runner)


def run_mdp_sweep(key, values, sweep_params):
    """
    Run a sweep over multiple parameter configurations.
    """

    sweep_results = []
    for val in values:
        mdps = []
        # Iterate over each parameter configuration and run the landscape generator
        for params in list(runner.dict_product(sweep_params)):
            params[key] = val
            mdp = random_MDP(**params)
            mdps.append(mdp)
        sweep_results.append(mdps)

    return sweep_results

class DiscountedMDP(MDP):
    """
    Represents a gamma-discounted, infinite-horizon Markov decision process.

    Attributes:
        rho (numpy.ndarray): Distribution over the initial state.
        P (numpy.ndarray): Transition probability matrix (S x A x S').
        R (numpy.ndarray): Reward function (S x A x S').
        gamma (float): Temporal discount factor.
        seed (int): Seed for random number generation.
        rng (numpy.random.RandomState): Random number generator.
        mdp_config (dict): Configuration dictionary.
    """

    def __init__(self, rho, P, R, gamma, seed=0, rng=None, mdp_config=None):
        # gamma: Temporal discount factor
        super().__init__(rho, P, R, seed=seed, rng=rng, mdp_config=mdp_config)
        self.gamma = gamma
        if mdp_config is not None:
            self.mdp_config = {**mdp_config, **self.mdp_config}
        self.mdp_config = {**self.mdp_config, **{"gamma": self.gamma}}


    def all_det_policies(self):
        """Generate all deterministic policies."""
        return np.eye(self.n_actions)[np.array(list(product(range(self.n_actions), repeat=self.n_states)))]

    def __iter__(self):
        """
        Make the MDP iterable.

        Returns:
            Iterator containing the MDP parameters.
        """
        return iter((self.rho, self.P, self.R, self.gamma))


    # ___________________________________________________________________________
    # Simulation

    def simulate(self, pi, state=None, action=None, mode='direct'):
        """
        Simulate the MDP under various modes.

        Args:
            pi (numpy.ndarray): Policy matrix (S x A).
            state (int): Initial state.
            action (int): Initial action.
            mode (str): Simulation mode ('direct', 'terminate', or 'reset').

        Yields:
            Tuple[int, int, float, bool]: Tuple containing (state, action, reward, done).
        """
        # Coerce arrays into functions
        pi_fn = lambda s, pi=pi: np.random.choice(np.arange(self.n_actions), p=pi[s])
        if state is None:
            s = sample(self.rho)
        if action is None:
            action = pi_fn(state)
        done = False
        while True:
            next_state = sample(self.P[state, action, :])
            reward = self.R[state, action, next_state]  #  r[s,a] = E[R[s,a,s']]
            yield (state, action, reward, done)
            done = False
            if mode != 'direct':
                if self.rng.uniform(0, 1) <= (1 - self.gamma):
                    if mode == 'terminate':
                        return
                    else:
                        next_state = sample(self.rho)
                        done = True
            state = next_state
            action = pi_fn(state)

    def get_return(self, pi, samples=1000, trunc=50):
        """Estimate the advantage function of a policy."""
        n_sa = np.zeros((self.n_states, self.n_actions)) + 1
        q = np.zeros((self.n_states, self.n_actions))
        v = np.zeros(self.n_states)
        sim = iter(self.simulate(pi, mode='reset'))
        trajectory = []

        for _ in range(samples):
            state, action, reward, done = next(sim)
            n_sa[state, action] += 1
            trajectory.append((state, action, reward, done))

            for tt, (state, action, _, _) in enumerate(reversed(trajectory), start=0):
                if tt > trunc:
                    break

                q[state, action] += reward
                v[state] += reward

        return self.qvn(q, v, n_sa)

    def qvn(self, q, v, n_sa):
        """Helper function to build advantage estimation from the pieces passed in."""
        n_s = n_sa.sum(axis=1)
        adv = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                adv[s, a] = q[s, a] / n_sa[s, a] - v[s] / n_s[s]
        return adv

    # ___________________________________________________________________________
    # Conditioning

    def mrp(self, pi):
        """Condition MDP on a policy to obtain an MRP."""
        return MRP(self.rho, np.einsum('sa,sap->sp', pi, self.P), np.einsum('sa,sap,sap->s', pi, self.P, self.R), self.gamma)

    __or__ = mrp  # alias M | pi

    # ___________________________________________________________________________
    # Implicit functions

    def J(self, pi):
        """Compute the expected value of a policy."""
        return (self | pi).J()

    def get_d_pi(self, pi, rho=None):
        """Compute the stationary distribution of a policy."""
        if rho is None:
            rho = self.rho

        return (self | pi).get_d_pi(rho)

    def get_V(self, pi):
        """Compute the value function `V(s)` for a policy."""
        return (self | pi).get_V()

    def sasa_matrix(self, pi, normalize=True):
        """
        Compute the equivalent of the normalized successor reorientation of the Markov chain.

        Args:
            pi (numpy.ndarray): Policy matrix (S x A).
            normalize (bool): Flag to enable/disable normalization.

        Returns:
            numpy.ndarray: Successor reorientation matrix (S x A x S x A).
        """
        n_states, n_actions = self.n_states, self.n_actions
        I = np.eye(n_states * n_actions)
        if normalize:
            H = linalg.solve(I - self.gamma * self.P.reshape(n_states * n_actions, n_states) @ self.policy_matrix(pi),
                             I * (1 - self.gamma))
        else:
            H = linalg.solve(I - self.gamma * self.P.reshape(n_states * n_actions, n_states) @ self.policy_matrix(pi),
                             I)
        H = H.reshape((n_states, n_actions, n_states, n_actions))
        return H


    def get_cond_no(self, pi):
        H = self.successor_representation(pi)
        eig = np.sort(np.linalg.eigvals(H))
        cond_no = np.abs(eig[1:].max() / eig[1:].min())
        return cond_no


    def successor_representation(self, pi, **kwargs):
        return (self | pi).successor_representation(**kwargs)

    def get_Q(self, pi):
        "Compute the action-value function `Q(s,a)` for a pi."
        return self.Q_from_V((self | pi).get_V())

    def Q_by_linalg(self, pi):
        """
        Compute the action-value function `Q(s,a)` for a policy `pi` by solving
        a linear system of equations.

        Args:
            pi (numpy.ndarray): Policy matrix (S x A).

        Returns:
            numpy.ndarray: Action-value function Q(s, a) (S x A).
        """
        r = self.r.ravel()
        return (linalg.solve(self.M(pi), r).reshape((self.n_states, self.n_actions)))

    def M(self, pi):
        """
        Compute the transition matrix `M(pi)` for a given policy `pi`.

        Args:
            pi (numpy.ndarray): Policy matrix (S x A).

        Returns:
            numpy.ndarray: Transition matrix M(pi) (S * A x S).
        """
        P = self.P.reshape((self.n_states * self.n_actions, self.n_states))
        return np.eye(self.n_states * self.n_actions) - self.gamma * P @ self.policy_matrix(pi)


    def policy_matrix(self, pi):
        """
        The policy matrix policy_matrix is an |S| × |S||A| matrix representation of pi.

        It is convenient because,
          policy_matrix P = P_pi = P(s' | s; pi)
          policy_matrix R = R_pi = E[ r | pi ]
          P policy_matrix = Pr[ <s', a'> | s, a ]

        This definition is used in Lagoudakis and Parr (2003) and Wang et al. (2008).

        """

        policy_matrix = np.zeros((self.n_states, self.n_states, self.n_actions))
        for s in range(self.n_states):
            policy_matrix[s, s, :] = pi[s, :]
        return policy_matrix.reshape((self.n_states, self.n_states * self.n_actions))

    # ___________________________________________________________________________
    # Operators

    def B(self, V):
        """
       Bellman operator.

       Args:
           V (numpy.ndarray): Value function (S,)

       Returns:
           tuple: Tuple containing the value of the Bellman operator and the optimal policy matrix.
               (v, pi)
               v (numpy.ndarray): Value of the Bellman operator (S,)
               pi (numpy.ndarray): Optimal policy matrix (S x A)
       """
        Q = self.Q_from_V(V)
        if 1:
            pi = np.zeros((self.n_states, self.n_actions))
            pi[range(self.n_states), Q.argmax(axis=1)] = 1
        else:
            pi = Q == Q.max(axis=1)[:, None]
            pi = pi / pi.sum(axis=1)[:, None]
        v = Q.max(axis=1)
        return v, pi

    def Q_from_V(self, V):
        "Lookahead by a single action from value function estimate `V`."
        r = self.r
        Q = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                Q[s, a] = r[s, a] + self.gamma * self.P[s, a, :] @ V
        return Q

    # ___________________________________________________________________________
    # Algorithms

    def solve_by_policy_iteration(self, max_iter=50):
        """
       Solve the MDP with the policy iteration algorithm.

       Args:
           max_iter (int, optional): Maximum number of iterations. Defaults to 50.

       Returns:
           dict: Dictionary containing the result of policy iteration.
               {
                   'obj': Objective value,
                   'policy': Optimal policy matrix (S x A),
                   'V': Value function (S,)
               }
       """
        V = np.zeros(self.n_states)
        pi_prev = np.zeros((self.n_states, self.n_actions))
        for _ in range(max_iter):
            _, pi = self.B(V)
            V = self.get_V(pi)
            if (pi_prev == pi).all():
                break
            pi_prev = pi
        else:
            print(f'Warning: policy iteration did not converge in {max_iter} iterations.')
        return {
            'obj': V @ self.rho,
            'policy': pi,
            'V': V,
        }

    def solve_by_value_iteration(self, tol=1e-10):
        """
        Solve the MDP with the value iteration algorithm.

        Args:
            tol (float, optional): Tolerance for convergence. Defaults to 1e-10.

        Returns:
            dict: Dictionary containing the result of the value iteration algorithm.
                {
                    'obj': Optimal value of the objective function,
                    'policy': Optimal policy matrix,
                    'V': Optimal value function
                }
        """
        V = np.zeros(self.n_states)
        while True:
            V1, pi = self.B(V)
            if np.abs(V1 - V).max() < tol: break
            V = V1
        return {
            'obj': V @ self.rho,
            'policy': pi,
            'V': V,
        }

    def get_optimal_Q(self, iters=int(1e5)):
        """
        Compute the optimal action-value function Q(s,a).

        Args:
            iters (int, optional): Maximum number of iterations for policy iteration. Defaults to int(1e5).

        Returns:
            array: The optimal action-value function Q(s,a).
        """
        V = self.solve_by_policy_iteration(max_iter=iters)["V"]
        Q = self.Q_from_V(V)
        return Q
        # n_states = P.shape[0]
        # n_actions = r.shape[1]
        # Q = np.zeros((n_states, n_actions))
        # return self.action_value_iteration(P, r, gamma, Q, iters=iters)


def random_MDP(seed, n_states, n_actions, gamma=0.95, n_branches=None, r=None, r_scale=1):
    """"
    Generate a randomly constructed MDP.

    Args:
        seed (int): Random seed for reproducibility.
        n_states (int): Number of states in the MDP.
        n_actions (int): Number of actions in the MDP.
        gamma (float, optional): Discount factor. Defaults to 0.95.
        n_branches (int, optional): Number of possible next states for each state-action pair. Defaults to None.
        r (int, optional): Number of states for which rewards are randomly generated. Defaults to None.
        r_scale (float, optional): Scaling factor for randomly generated rewards. Defaults to 1.

    Returns:
        DiscountedMDP: The randomly generated MDP.
    """
    mdp_config = {
        "mdp_type": "random_mdp",
        "seed": seed,
        "gamma":  gamma,
        "n_states": n_states,
        "n_actions": n_actions,
        "n_branches": n_branches,
        "r": r,
        "r_scale": r_scale
    }
    if n_branches is None:
        n_branches = n_states
    if r is None:
        r = n_states

    rng = np.random.RandomState(seed)
    P = np.zeros((n_states, n_actions, n_states))
    states = np.array(list(range(n_states)))

    for s in range(n_states):
        for a in range(n_actions):
            # pick b states to be connected to.
            connected = rng.choice(states, size=n_branches, replace=False)
            P[s, a, connected] = random_dist(n_branches)

    R = np.zeros((n_states, n_actions, n_states))
    rstates = rng.choice(states, size=r, replace=False)
    R[rstates, :, :] = rng.uniform(0, r_scale, r)

    env = DiscountedMDP(
        seed=seed,
        rng=rng,
        rho=random_dist(n_states),
        R=R,
        P=P,
        gamma=gamma,
        mdp_config=mdp_config,
    )

    return env

