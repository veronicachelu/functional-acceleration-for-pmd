import numpy as np
import jax
import jax.numpy as jnp
from collections import defaultdict
import distrax

class Policy:
    n_states = None
    n_actions = None
    parametric = True
    theta = None
    pi = None

    def set_empty(self, n_states, n_actions, parametric=True):
        """
        Initialize an empty policy.

        Args:
            n_states: Number of states in the environment.
            n_actions: Number of actions in the environment.
            parametric: Whether the policy is parametric or not.
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.parametric = parametric
        self.theta = np.zeros((n_states, n_actions))
        self.pi = jax.nn.softmax(self.theta, axis=1)



    def get(self):
        """
        Get the policy parameters.

        Returns:
            Policy parameters (theta if parametric, pi if not).
        """
        if self.parametric:
            return self.theta
        else:
            return self.pi

    def get_pi(self):
        """
        Get the policy probabilities.

        Returns:
            Policy probabilities.
        """
        return self.pi

    def get_params(self):
        """
        Get the policy parameters.

        Returns:
            Policy parameters (theta).
        """
        return self.theta

    def set_pi(self, pi):
        """
        Set the policy probabilities.

        Args:
            pi: Policy probabilities.
        """
        self.pi = pi

    def set_params(self, theta):
        """
        Set the policy parameters.

        Args:
            theta: Policy parameters.
        """
        self.theta = theta
        self.pi = jax.nn.softmax(self.theta, axis=1)

    def set(self, pi, theta):
        """
        Set both policy parameters and probabilities.

        Args:
            pi: Policy probabilities.
            theta: Policy parameters.
        """
        self.pi = pi
        self.theta = theta

    def set_greedy(self, q_t):
        """
        Set the policy to be greedy based on the Q-values.

        Args:
            q_t: Q-values for the current state.

        Returns:
            Greedy policy probabilities and parameters.
        """
        n_states, n_actions = q_t.shape
        self.n_states = n_states
        self.n_actions = n_actions
        self.parametric = False

        act_greedy_tp1 = np.argmax(q_t, -1)
        pi_greedy_tp1 = np.zeros_like(q_t)
        pi_greedy_tp1[np.arange(n_states), act_greedy_tp1] = 1
        theta_greedy_tp1 = np.zeros((n_states, n_actions))
        theta_greedy_tp1[pi_greedy_tp1 != 1] = -np.infty
        theta_greedy_tp1 = jax.nn.log_softmax(theta_greedy_tp1)

        self.theta = theta_greedy_tp1
        self.pi = pi_greedy_tp1

        return pi_greedy_tp1, theta_greedy_tp1
