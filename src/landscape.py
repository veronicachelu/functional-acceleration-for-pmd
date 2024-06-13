import numpy as np
import jax
import jax.numpy as jnp
from collections import defaultdict
import importlib
from src import utils
from src import policy
from src import logger
from src import mdp
import yaml
import os
mdp = importlib.reload(mdp)
policy = importlib.reload(policy)
logger = importlib.reload(logger)
utils = importlib.reload(utils)

DEFAULT_KWARGS = {
    "noise_type": "normal",
    "k": 0,
    "n": 0,
    "optimism_decay": 1.0,
    "use_d": False,
    "m": 0,
    "initial_policy": "random_uniform",
    "behaviour_policy": "online",
    "trunc": 50,
    "mix_start": "random_uniform",
    "mix_end": "optimal",
    "inner_loop_lr": 0.1,
    "tau_mean1": 0.0,
    "mu": 1.0,
    "tau_mean2": 0.0,
    "tau_scale1": 0.0,
    "tau_scale2": 0.0,
    "eps_0": 1e-4,
    "experiment_name": "untitled",
    "load_logs": False,
    "save_logs": False,
}

class PolicyLandscape:
    def __init__(self, env, **kwargs):
        """
        Initialize PolicyLandscape class.

        Args:
            env: Environment object.
            num_iter: Number of iterations for iterative optimization.
            policy_improvement_type: Type of policy improvement algorithm.
            noise_type: Type of noise used in optimization.
            k: Parameter used in some policy improvement algorithms.
            n: Parameter used in some policy improvement algorithms.
            optimism_decay: Parameter used in some policy improvement algorithms.
            use_d: Parameter used in some policy improvement algorithms.
            inner_loop_lr: Learning rate for inner loop optimization.
            tau_mean1: Mean of Gaussian distribution used in some policy improvement algorithms.
            mu: Parameter used in some policy improvement algorithms.
            tau_mean2: Mean of Gaussian distribution used in some policy improvement algorithms.
            tau_scale1: Scale of Gaussian distribution used in some policy improvement algorithms.
            tau_scale2: Scale of Gaussian distribution used in some policy improvement algorithms.
            eps_0: Small value used to avoid division by zero.
        """
        # Environment setup
        self.env = env
        self.seed = int(kwargs.pop("seed", None))
        kwargs["seed"] = int(self.seed)
        self.kwargs = {**DEFAULT_KWARGS, **kwargs}
        self._initialize_params()

    def _initialize_params(self):
        self.seed = int(self.seed) if self.seed is not None else np.random.randint(1e5)

        # Environment setup
        self.num_iter = self.kwargs["num_iter"]
        self.mix_start = self.kwargs["mix_start"]
        self.mix_end = self.kwargs["mix_end"]
        self.initial_policy = self.kwargs["initial_policy"]
        self.behaviour_policy = self.kwargs["behaviour_policy"]
        self.policy_improvement_type = self.kwargs["policy_improvement_type"]
        self.noise_type = self.kwargs["noise_type"]
        self.k = self.kwargs["k"] # no of steps of inner-loop parameter optimization for the forward update (step 1)
        self.n = self.kwargs["n"] # no of steps of inner-loop parameter optimization for the backward update (step 2)
        self.optimism_decay = self.kwargs["optimism_decay"]
        self.use_d = self.kwargs["use_d"]
        self.inner_loop_lr = self.kwargs["inner_loop_lr"]
        self.tau_mean1 = self.kwargs["tau_mean1"]
        self.mu = self.kwargs["mu"]
        self.tau_mean2 = self.kwargs["tau_mean2"]
        self.tau_scale1 = self.kwargs["tau_scale1"]
        self.tau_scale2 = self.kwargs["tau_scale2"]
        self.m = self.kwargs["m"] # no trajectories
        self.trunc = self.kwargs["trunc"]
        self.eps_0 = self.kwargs["eps_0"] # arbitrary sequence to use in the lower bound of PMD

        self.rng = np.random.RandomState(self.seed)
        self.rho_uniform = np.full_like(self.env.rho, fill_value=1/self.env.n_states)

        # Initial setup for policy and Q-values
        self.initialize_policies()
        # Functions for policy evaluation and improvement
        self.initialize_functions()

        # Log settings
        self.experiment_name = self.kwargs["experiment_name"]
        self.save_logs = self.kwargs["save_logs"]
        self.load_logs = self.kwargs["load_logs"]
        self.folder_path = self._setup_logs_folder()

    def _setup_logs_folder(self):
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Define the logs folder
        folder_name = "logs"
        # Construct the full path to the folder
        folder_path = os.path.join(current_dir, "..", folder_name,
                                   self.experiment_name,
                                   self.policy_improvement_type)
        return folder_path

    def initialize_optimal_policy(self):
        """
        Initialize optimal policy used for
        plotting statistics about policy improvement.

        Args:
            env: Environment object.
        """
        # Compute optimal Q-values and policies for different scenarios
        self.q_star = self.env.get_optimal_Q(iters=int(100))
        self.policy_star = policy.Policy()
        self.policy_star.set_greedy(self.q_star)
        self.v_star = self.env.get_V(self.policy_star.get_pi()) # Q(pi_0)
        self.v_star__rho = np.sum(self.v_star * self.env.rho)


    def initialize_suboptimal_policies(self):
        """
        Initialize optimal policies used for
        plotting statistics about policy improvement.

        Args:
           env: Environment object.
        """
        if type(self.env).__name__ == "CliffwalkMDP":
            import copy
            self.q_subopt1 = np.array(copy.deepcopy(self.q_star))
            for s in [5, 10, 6, 11, 7, 12, 8, 13]:
                self.q_subopt1[s, 3] = np.infty
            self.policy_subopt1 = policy.Policy()
            self.policy_subopt1.set_greedy(self.q_subopt1)
            self.v_subopt1 = self.env.get_V(self.policy_subopt1.get_pi())
            self.v_subopt1__rho = np.sum(self.v_subopt1 * self.env.rho)

            self.q_subopt2 = np.array(copy.deepcopy(self.q_star))
            for s in [5, 6, 7, 8]:
                self.q_subopt2[s, 3] = np.infty
            self.policy_subopt2 = policy.Policy()
            self.policy_subopt2.set_greedy(self.q_subopt2)
            self.v_subopt2 = self.env.get_V(self.policy_subopt2.get_pi())
            self.v_subopt2__rho = np.sum(self.v_subopt2 * self.env.rho)


    def initialize_policies(self):
        """
        Initialize optimal/suboptimal/random/initial/behaviour policies used for
        plotting statistics about policy improvement.

        Args:
            env: Environment object.
        """
        self.initialize_optimal_policy()
        self.initialize_suboptimal_policies()

        self.policy_random = policy.Policy()
        self.policy_random.set_empty(self.env.n_states, self.env.n_actions, parametric=False)
        self.v_random = self.env.get_V(self.policy_random.get_pi())
        self.v_random__rho = np.sum(self.v_random * self.env.rho)

        self.policy_mix = self.initialize_policy_mix()
        self.policy_0 = self.get_policy(self.initial_policy)

        if self.behaviour_policy == "online":
            self.policy_behave = None
        else:
            self.policy_behave = self.get_policy(self.behaviour_policy)

    def get_policy(self, params_type):
        pol = policy.Policy()
        theta = self.get_params(params_type)
        theta = np.nan_to_num(theta, nan=-np.infty)
        pol.set_params(theta)
        return pol

    def get_params(self, params_type):
        if params_type == "random_normal":
            initializer = jax.nn.initializers.normal(1.0)
            theta = initializer(jax.random.key(self.seed),
                  (self.env.n_states, self.env.n_actions),
                          jnp.float32)
        elif params_type in ["random_uniform", "offline"]:
            initializer = jax.nn.initializers.uniform(1.0)
            theta = initializer(jax.random.key(self.seed),
                                  (self.env.n_states, self.env.n_actions), jnp.float32)
        elif params_type in ["random_uniform2", "offline"]:
            initializer = jax.nn.initializers.uniform(10.0)
            theta = initializer(jax.random.key(self.seed),
                                (self.env.n_states, self.env.n_actions), jnp.float32)

        elif params_type == "zeros":
            theta = np.zeros((self.env.n_states, self.env.n_actions))
        elif params_type == "optimal":
            act = np.argmax(self.q_star, -1)
            pi = np.zeros_like(self.q_star)
            pi[np.arange(self.env.n_states), act] = 1
            theta = np.zeros((self.env.n_states, self.env.n_actions))
            theta[pi == 1] = 1
            theta = jax.nn.log_softmax(theta)

        elif params_type == "boundary":
            act = np.argmax(self.q_star, -1)
            pi = np.zeros_like(self.q_star)
            pi[np.arange(self.env.n_states), act] = 1
            theta = np.zeros((self.env.n_states, self.env.n_actions))
            theta[1][pi[1] != 1] = 3
            theta[0][pi[0] != 1] = 1
            theta = jax.nn.log_softmax(theta)

        elif params_type == "hard_adversarial":
            act = np.argmax(self.q_star, -1)
            pi = np.zeros_like(self.q_star)
            pi[np.arange(self.env.n_states), act] = 1
            theta = np.zeros((self.env.n_states, self.env.n_actions))
            theta[pi != 1] = 5
            theta = jax.nn.log_softmax(theta)

        elif params_type == "adversarial":
            act = np.argmax(self.q_star, -1)
            pi = np.zeros_like(self.q_star)
            pi[np.arange(self.env.n_states), act] = 1
            theta = np.zeros((self.env.n_states, self.env.n_actions))
            theta[pi != 1] = 3
            theta = jax.nn.log_softmax(theta)

        elif params_type == "partially_adversarial":
            det_policies = self.env.all_det_policies()
            theta = np.zeros((self.env.n_states, self.env.n_actions))
            theta[det_policies[1] == 1] = 3
            theta = jax.nn.log_softmax(theta)

        return theta

    def initialize_policy_mix(self):
        thetas = np.linspace(self.get_params(self.mix_start),
                             self.get_params(self.mix_end),
                             self.num_iter, dtype=np.float)
        policies = []
        for i in range(self.num_iter):
            pol = policy.Policy()
            theta = np.nan_to_num(thetas[i], nan=-np.infty)
            pol.set_params(theta)
            policies.append(pol)
        return policies


    def initialize_functions(self):
        """
        Initialize functions for policy evaluation and improvement.
        """
        self.surrogate_objective = jax.jit(self.surrogate_objective_function)

    def surrogate_objective_function(self, theta_k, theta_t, d_t, q_t, step_size_t):
        """
        Compute surrogate objective function used in policy improvement.

        Args:
            theta_k: Parameters of the current policy.
            theta_t: Parameters of the previous policy.
            d_t: Stationary distribution of the policy.
            q_t: Q-values of the policy.
            step_size_t: Step size used in optimization.

        Returns:
            Gradient of the surrogate objective function.
        """
        def surrogate_objective(theta):
            return np.sum(jax.lax.stop_gradient(d_t) * (
                    -np.sum(jax.lax.stop_gradient(q_t) * jax.nn.softmax(theta, axis=1), axis=1)
                    + (1 / jax.lax.stop_gradient(step_size_t)) * utils.categorical_kl_divergence(theta,
                                                                                                jax.lax.stop_gradient(
                                                                                                    theta_t))),
                          axis=-1)

        grad_t = jax.grad(surrogate_objective)(theta_k)
        return grad_t / (1 - self.env.gamma)


    def policy_evaluation(self, policy_t):
        """
        Evaluate the policy using Q-values.

        Args:
            policy_t: Policy object.

        Returns:
            Q-values of the policy.
        """
        if self.m == 0:
            q_t = self.env.get_Q(policy_t.get_pi())  # Q(pi_0)
        else:
            q_t = self.env.get_return(policy_t.get_pi(),
                                  samples=self.trunc * self.m,
                                  trunc=self.trunc)
        return q_t

    def performance(self, policy_t):
        # Compute the value function for the given policy using the environment dynamics
        # and the policy's action probabilities

        v_t = self.env.get_V(policy_t.get_pi())  # V(pi_0)

        # Return the computed value function
        return v_t


    def stationary_distribution(self, policy_t, rho=None):
        # Compute the stationary distribution for the given policy using the environment dynamics,
        # discount factor, policy's action probabilities, and the environment's initial distribution
        rho = rho if rho is not None else self.env.rho
        if self.behaviour_policy == "online":
            d_t = jax.lax.stop_gradient(self.env.get_d_pi(policy_t.get_pi(), rho))
        else:
            if self.behaviour_policy == "offline":
                d_t = np.full((self.env.n_states,), fill_value=1/self.env.n_states)
            else:
                d_t = jax.lax.stop_gradient(self.env.get_d_pi(self.policy_behave.get_pi(), rho))
        # Return the computed stationary distribution
        return d_t


    def greedification(self, q_t):
        # Create a new policy object for the greedy policy update
        policy_greedy_tp1 = policy.Policy()

        # Set the greedy action selection for the policy using the given Q-values
        policy_greedy_tp1.set_greedy(q_t)

        # Return the policy with the updated greedy action selection
        return policy_greedy_tp1


    def add_noise(self, q_t, tau_mean, tau_scale):
        # Check the noise type
        if self.noise_type == "normal":
            # Add normal noise to the Q-values
            return utils.add_normal_noise(self.rng, q_t, tau_mean, tau_scale)
        else:
            # Add uniform noise to the Q-values
            return utils.add_uniform_noise(self.rng, q_t, tau_mean)


    def policy_interpolation(self, t, policy_t, history_t=None):
        # Prepare logs for monitoring
        q_t = self.policy_evaluation(policy_t)
        # Add noise to the Q-values
        inexact_q_t = self.add_noise(q_t, self.tau_mean1, self.tau_scale1)

        # Calculate the error in Q-values
        err_q_t = inexact_q_t - q_t

        # Calculate the stationary distribution of the current policy
        d_t = self.stationary_distribution(policy_t)

        # Prepare logs for monitoring
        logs = {
            "q_t": inexact_q_t,
            "err_q_t": np.sum(d_t * np.max(err_q_t, 1)),  # Weighted maximum error
            "step_size_t": 0  # Placeholder for step size (to be filled later)
        }

        return self.policy_mix[t], logs


    def policy_iteration(self, t, policy_t, history_t=None):
        # Evaluate the policy to get Q-values
        q_t = self.policy_evaluation(policy_t)
        # policy_tm1 = history_t["policy_tm1"]
        inexact_q_tm1 = history_t["q_tm1"]

        # Add noise to the Q-values
        inexact_q_t = self.add_noise(q_t, self.tau_mean1, self.tau_scale1)

        # Calculate the error in Q-values
        err_q_t = inexact_q_t - q_t
        # delta_t = (t/(t+1)) * (inexact_q_t - inexact_q_tm1)
        # q = inexact_q_t + delta_t
        # Generate a greedy policy based on the noisy Q-values
        policy_greedy_tp1 = self.greedification(inexact_q_t)

        # Calculate the stationary distribution of the current policy
        d_t = self.stationary_distribution(policy_t)

        # Prepare logs for monitoring
        logs = {
            "d_t": d_t,
            "q_tm1": inexact_q_tm1,
            "q_t": inexact_q_t,
            "err_q_t": err_q_t,  # Weighted maximum error
            "step_size_t": 0  # Placeholder for step size (to be filled later)
        }

        return policy_greedy_tp1, logs


    def policy_mirror_descent(self, t, policy_t, history_t=None):
        # Calculate the stationary distribution of the current policy
        d_t = self.stationary_distribution(policy_t)
        # Exact policy evaluation
        q_t = self.policy_evaluation(policy_t)

        # Add noise to the Q-values
        inexact_q_t = self.add_noise(q_t, self.tau_mean1, self.tau_scale1)

        # Determine optimal step size
        step_size_t = self.optimal_step_size(t, inexact_q_t, policy_t)

        # Calculate the error in Q-values
        err_q_t = inexact_q_t - q_t

        # Define policy gradient function for inexact policy improvement
        policy_gradient_fn = lambda theta: self.inexact_policy_gradient(theta, d_t, inexact_q_t, policy_t, step_size_t)

        # Perform inner loop gradient descent for inexact policy improvement
        policy_tp1, avg_gradient = self.inner_loop_gradient_descent(policy_t, policy_gradient_fn, self.k)
        # Prepare logs for monitoring
        logs = {
            "grad_t": np.linalg.norm(avg_gradient),  # Average gradient during inner loop gradient descent
            "d_t": d_t,
            "step_size_t": step_size_t,  # Weighted step size
            "q_t": inexact_q_t,  # Noisy Q-values
            "err_q_t": err_q_t  # Weighted maximum error in Q-values
        }

        return policy_tp1, logs


    def inexact_policy_mirror_descent(self, t, q, policy_t, param_steps):
        # Exact stationary distribution
        d_t = self.stationary_distribution(policy_t)

        # Adaptive step size
        step_size_t = self.optimal_step_size(t, q, policy_t)

        # Define policy gradient function for inexact policy improvement
        policy_gradient_fn = lambda theta: self.inexact_policy_gradient(theta, d_t, q, policy_t, step_size_t)

        # Perform inner loop gradient descent for inexact policy improvement
        policy_tp1, avg_gradient = self.inner_loop_gradient_descent(policy_t, policy_gradient_fn, param_steps)

        # Prepare logs for monitoring
        logs = {
            "grad_t": np.linalg.norm(avg_gradient),  # Average gradient during inner loop gradient descent
        }

        return policy_tp1, logs


    def lookahead_policy_mirror_descent(self, t, policy_t, history_t=None):
        # Exact policy iteration
        d_t = self.stationary_distribution(policy_t)
        q_t = self.policy_evaluation(policy_t)

        # Add noise to Q-value estimates
        inexact_q_t = self.add_noise(q_t, self.tau_mean1, self.tau_scale1)
        err_q_t = inexact_q_t - q_t

        # Proposal policy for lookahead
        proposal_tp1 = self.greedification(inexact_q_t)

        # Prepare logs for monitoring
        logs = {
            "q_t": inexact_q_t,
            "err_q_t": err_q_t,
            "step_size_t": 0,
            "d_t": d_t,
        }

        # Exact policy evaluation of the lookahead policy
        q_tp1 = self.policy_evaluation(proposal_tp1)

        # Add noise to lookahead Q-value estimates
        inexact_q_tp1 = self.add_noise(q_tp1, self.tau_mean2, self.tau_scale2)
        err_q_tp1 = inexact_q_tp1 - q_tp1
        d_tp1 = self.stationary_distribution(proposal_tp1)

        # Policy Mirror Descent (PMD) with lookahead gradient (keeping the old stationary distribution)
        d_t = self.stationary_distribution(policy_t)
        step_size_tp1 = self.optimal_step_size(t, inexact_q_tp1, policy_t)

        # Define policy gradient function for inexact policy improvement
        policy_gradient_fn = lambda theta: self.inexact_policy_gradient(theta, d_t, inexact_q_tp1, policy_t, step_size_tp1)

        # Perform inner loop gradient descent for inexact policy improvement
        policy_tp1, avg_gradient = self.inner_loop_gradient_descent(policy_t, policy_gradient_fn, self.n)

        # Update logs with information about the lookahead policy
        logs.update({
            "step_size_tp1": step_size_tp1,
            "grad_tp1": np.linalg.norm(avg_gradient),  # Average gradient during inner loop gradient descent
            "d_tp1": d_tp1,
            "q_tp1": inexact_q_tp1,
            "err_q_tp1": err_q_tp1,
            "proposal_tp1": proposal_tp1,
        })

        return policy_tp1, logs


    def optimistic_correction_policy_mirror_descent(self, t, policy_t, history_t=None):
        # Exact Policy Mirror Descent (PMD)
        d_t = self.stationary_distribution(policy_t)
        q_t = self.policy_evaluation(policy_t)

        # Add noise to Q-value estimates
        inexact_q_t = self.add_noise(q_t, self.tau_mean1, self.tau_scale1)
        err_q_t = inexact_q_t - q_t

        # Calculate optimal step size for PMD
        step_size_t = self.optimal_step_size(t, inexact_q_t, policy_t)

        # Define policy gradient function for inexact policy improvement
        policy_gradient_fn = lambda theta: self.inexact_policy_gradient(theta, d_t, inexact_q_t, policy_t, step_size_t)

        # Perform inner loop gradient descent for inexact policy improvement
        proposal_tp1, avg_gradient = self.inner_loop_gradient_descent(policy_t, policy_gradient_fn, self.k)

        # Prepare logs for monitoring
        logs = {
            "step_size_t": step_size_t,
            "q_t": inexact_q_t
        }

        # Inexact PMD using the correction and bootstrapping on the future PMD policy at timestep t+1
        # Exact stationary distribution for t+1
        d_tp1 = self.stationary_distribution(proposal_tp1)
        q_tp1 = self.policy_evaluation(proposal_tp1)

        # Add noise to Q-value estimates for t+1
        inexact_q_tp1 = self.add_noise(q_tp1, self.tau_mean2, self.tau_scale2)
        err_q_tp1 = inexact_q_tp1 - q_tp1

        # Calculate optimal step size for t+1
        step_size_tp1 = self.optimal_step_size(t, inexact_q_tp1, policy_t)

        # Calculate step size ratio
        if self.use_d == 1:
            d = d_tp1
            d_ip = d_t / d_tp1
            step_size_ratio = ((step_size_t / step_size_tp1) * d_ip)[..., None]
        else:
            d = d_t
            step_size_ratio = (step_size_t / step_size_tp1)[..., None]

        # Compute the corrected Q-values for t+1
        delta_t = inexact_q_tp1 - step_size_ratio * inexact_q_t
        delta_t *= self.optimism_decay

        # Update logs with information about t+1
        logs.update({
            "step_size_tp1": step_size_tp1,
            "step_ratio": np.sum(step_size_t / step_size_tp1 * d_t),
            "q_tp1": q_tp1,
            "proposal_tp1": proposal_tp1,
            "delta_t": np.sum(np.sum(delta_t * proposal_tp1.get_pi(), axis=1) * d_tp1),
        })

        # Define policy gradient function for inexact policy improvement for t+1
        policy_gradient_fn = lambda theta: self.inexact_policy_gradient(theta, d, delta_t, proposal_tp1, step_size_tp1)

        # Perform inner loop gradient descent for inexact policy improvement for t+1
        policy_tp1, avg_gradient = self.inner_loop_gradient_descent(proposal_tp1, policy_gradient_fn, self.n)

        return policy_tp1, logs

    def optimistic_correction_policy_mirror_descent_v2(self, t, policy_t, history_t=None):
        # Exact Policy Mirror Descent (PMD)
        d_t = self.stationary_distribution(policy_t)
        q_t = self.policy_evaluation(policy_t)

        # Add noise to Q-value estimates for t
        inexact_q_t = self.add_noise(q_t, self.tau_mean1, self.tau_scale1)
        err_q_t = inexact_q_t - q_t

        # Calculate optimal step size for PMD for t
        step_size_t = self.optimal_step_size(t, inexact_q_t, policy_t)

        # Define policy gradient function for inexact policy improvement for t
        policy_gradient_fn = lambda theta: self.inexact_policy_gradient(theta, d_t, inexact_q_t, policy_t, step_size_t)

        # Perform inner loop gradient descent for inexact policy improvement for t
        proposal_tp1, avg_gradient = self.inner_loop_gradient_descent(policy_t, policy_gradient_fn, self.k)

        # Prepare logs for monitoring for t
        logs = {
            "step_size_t": step_size_t,
            "q_t": inexact_q_t
        }

        # Calculate stationary distribution for t+1
        d_tp1 = self.stationary_distribution(proposal_tp1)
        q_tp1 = self.policy_evaluation(proposal_tp1)

        # Add noise to Q-value estimates for t+1
        inexact_q_tp1 = self.add_noise(q_tp1, self.tau_mean2, self.tau_scale2)
        err_q_tp1 = inexact_q_tp1 - q_tp1

        # Calculate optimal step size for PMD for t+1
        step_size_tp1 = self.optimal_step_size(t, inexact_q_tp1, policy_t)

        # Calculate step size ratio for t+1
        step_size_ratio = (1 / step_size_tp1)[..., None]

        # Compute correction for t+1
        theta_t = policy_t.get_params()
        proposal_theta_t = proposal_tp1.get_params()
        log_t = utils.log_diff(proposal_theta_t, theta_t)
        delta_t = q_tp1 - step_size_ratio * self.mu * log_t
        delta_t *= self.optimism_decay

        # Update logs with information about t+1
        logs.update({
            "step_size_tp1": step_size_tp1,
            "step_ratio": np.sum(step_size_t / step_size_tp1 * d_tp1),
            "q_tp1": q_tp1,
            "proposal_tp1": proposal_tp1,
            "delta_t": np.sum(np.sum(delta_t * proposal_tp1.get_pi(), axis=1) * d_tp1),
        })

        # Define policy gradient function for inexact policy improvement for t+1
        policy_gradient_fn = lambda theta: self.inexact_policy_gradient(theta, d_t, delta_t, proposal_tp1, step_size_tp1)

        # Perform inner loop gradient descent for inexact policy improvement for t+1
        policy_tp1, avg_gradient = self.inner_loop_gradient_descent(proposal_tp1, policy_gradient_fn, self.n)

        return policy_tp1, logs


    def lazy_optimistic_correction_policy_mirror_descent(self, t, policy_t, history_t=None):
        # Previous policy at timestep t-1
        policy_tm1 = history_t["policy_tm1"]
        q_tm1 = history_t["q_tm1"]

        # Exact policy evaluation of previous policy at timestep t-1
        # Compute correction for t-1
        d_t = self.stationary_distribution(policy_t)
        step_size_tm1 = self.optimal_step_size(t - 1, q_tm1, policy_tm1)

        # Policy evaluation and noise addition for current timestep t
        q_t = self.policy_evaluation(policy_t)
        inexact_q_t = self.add_noise(q_t, self.tau_mean1, self.tau_scale1)
        err_q_t = inexact_q_t - q_t
        step_size_t = self.optimal_step_size(t, inexact_q_t, policy_t)
        # Calculate step size ratio
        if self.use_d == 0:
            # d_ip = d_t / d_tp1
            step_size_ratio = 1#((step_size_t / step_size_tp1) * d_ip)[..., None]
        else:
            step_size_ratio = (step_size_tm1 / step_size_t)[..., None]

        delta_tm1 = (inexact_q_t - step_size_ratio*q_tm1)
        delta_tm1 *= self.optimism_decay

        # Inexact policy mirror descent with correction for timestep t-1
        # applied at timestep t to compute a "half-step" proposal
        policy_gradient_fn = lambda theta: self.inexact_policy_gradient(theta, d_t, delta_tm1, policy_t, step_size_tm1)
        proposal_tp1, avg_gradient = self.inner_loop_gradient_descent(policy_t, policy_gradient_fn, self.n)

        # Inexact policy mirror descent for timestep t
        # applied by bootstrapping on the half-step proposal
        policy_gradient_fn = lambda theta: self.inexact_policy_gradient(theta, d_t, inexact_q_t, proposal_tp1, step_size_t)
        policy_tp1, avg_gradient = self.inner_loop_gradient_descent(proposal_tp1, policy_gradient_fn, self.k)

        # Prepare logs for monitoring
        logs = {
            "step_size_tm1": np.sum(step_size_tm1 * d_t),
            "step_size_t": step_size_t,
            "q_tm1": q_tm1,
            "proposal_tp1": proposal_tp1,
            "delta_tm1": np.sum(np.sum(delta_tm1 * policy_t.get_pi(), axis=1) * d_t),
            "q_t": inexact_q_t
        }

        return policy_tp1, logs

    def lazy_optimistic_correction_policy_mirror_descent_v2(self, t, policy_t, history_t=None):
        # Previous policy at timestep t-1
        policy_tm1 = history_t["policy_tm1"]

        # Exact policy evaluation of previous policy at timestep t-1
        q_tm1 = self.policy_evaluation(policy_tm1)

        # Compute correction for t-1
        d_t = self.stationary_distribution(policy_t)
        step_size_tm1 = self.optimal_step_size(t - 1, q_tm1, policy_tm1)

        # Policy evaluation and noise addition for current timestep t
        q_t = self.policy_evaluation(policy_t)
        inexact_q_t = self.add_noise(q_t, self.tau_mean1, self.tau_scale1)
        err_q_t = inexact_q_t - q_t
        step_size_t = self.optimal_step_size(t, inexact_q_t, policy_t)
        step_size_ratio = (1 / step_size_t)[..., None]

        # Compute correction for timestep t-1
        theta_t = policy_t.get_params()
        theta_tm1 = policy_tm1.get_params()
        delta_tm1 = utils.log_diff(theta_t, theta_tm1)
        delta_tm1 *= self.optimism_decay

        # Inexact policy mirror descent with correction for timestep t-1
        # applied at timestep t to compute a "half-step" proposal
        policy_gradient_fn = lambda theta: self.inexact_policy_gradient(theta, d_t, delta_tm1, policy_t, step_size_tm1)
        proposal_tp1, avg_gradient = self.inner_loop_gradient_descent(policy_t, policy_gradient_fn, self.n)

        # Inexact policy mirror descent for timestep t
        # applied by bootstrapping on the half-step proposal
        policy_gradient_fn = lambda theta: self.inexact_policy_gradient(theta, d_t, q_t, proposal_tp1, step_size_t)
        policy_tp1, avg_gradient = self.inner_loop_gradient_descent(proposal_tp1, policy_gradient_fn, self.k)

        # Prepare logs for monitoring
        logs = {
            "step_size_tm1": np.sum(step_size_tm1 * d_t),
            "step_size_t": step_size_t,
            "q_tm1": q_tm1,
            "proposal_tp1": proposal_tp1,
            "delta_tm1": np.sum(np.sum(delta_tm1 * policy_t.get_pi(), axis=1) * d_t),
            "q_t": q_t
        }

        return policy_tp1, logs


    def lazy_momentum(self, t, policy_t, history_t=None):
        # Previous policy at timestep t-1
        policy_tm1 = history_t["policy_tm1"]
        q_tm1 = history_t["q_tm1"]

        # Exact policy evaluation of previous policy at timestep t-1
        # Compute correction
        d_t = self.stationary_distribution(policy_t)
        step_size_tm1 = self.optimal_step_size(t - 1, q_tm1, policy_tm1)

        # Policy evaluation and noise addition for current timestep t
        q_t = self.policy_evaluation(policy_t)
        inexact_q_t = self.add_noise(q_t, self.tau_mean1, self.tau_scale1)
        err_q_t = inexact_q_t - q_t
        step_size_t = self.optimal_step_size(t, inexact_q_t, policy_t)
        step_size_ratio = (step_size_tm1 / step_size_t)[..., None]
        delta_tm1 = (inexact_q_t - q_tm1)
        delta_tm1 *= self.optimism_decay

        # Inexact policy improvement
        q_mom = inexact_q_t + delta_tm1
        policy_gradient_fn = lambda theta: self.inexact_policy_gradient(theta, d_t, q_mom, policy_t, step_size_t)
        policy_tp1, avg_gradient = self.inner_loop_gradient_descent(policy_t, policy_gradient_fn, self.n + self.k)

        # Prepare logs for monitoring
        logs = {
            "grad_t": np.linalg.norm(avg_gradient),
            "step_size_tm1": np.sum(step_size_tm1 * d_t),
            "step_size_t": step_size_t,
            "q_tm1": q_tm1,
            "delta_tm1": np.sum(np.sum(delta_tm1 * policy_t.get_pi(), axis=1) * d_t),
            "q_t": inexact_q_t,
            "q_mom": q_mom,
            "step_ratio": step_size_ratio
        }

        return policy_tp1, logs


    def lazy_momentum_v2(self, t, policy_t, history_t=None):
        # Previous policy at timestep t-1
        policy_tm1 = history_t["policy_tm1"]
        # Exact policy evaluation of previous policy at timestep t-1
        q_tm1 = self.policy_evaluation(policy_tm1)
        # Exact policy evaluation of current policy at timestep t
        # Compute correction
        d_t = self.stationary_distribution(policy_t)
        step_size_tm1 = self.optimal_step_size(t - 1, q_tm1, policy_tm1)

        q_t = self.policy_evaluation(policy_t)
        step_size_t = self.optimal_step_size(t, q_t, policy_t)
        step_size_ratio = (1 / step_size_t)[..., None]
        # Compute correction
        theta_tm1 = policy_tm1.get_params()
        theta_t = policy_t.get_params()
        delta_tm1 = step_size_ratio * utils.log_diff(theta_t, theta_tm1)
        delta_tm1 *= self.optimism_decay

        # Inexact policy improvement
        q_mom = q_t + delta_tm1
        policy_gradient_fn = lambda theta: self.inexact_policy_gradient(theta, d_t, q_mom, policy_t, step_size_t)
        policy_tp1, avg_gradient = self.inner_loop_gradient_descent(policy_t, policy_gradient_fn, self.n + self.k)

        # Prepare logs for monitoring
        logs = {
            "step_size_tm1": np.sum(step_size_tm1 * d_t),
            "step_size_t": step_size_t,
            "q_tm1": q_tm1,
            "delta_tm1": np.sum(np.sum(delta_tm1 * policy_t.get_pi(), axis=1) * d_t),
            "q_t": q_t,
            "q_mom": q_mom,
            "step_ratio": step_size_ratio
        }

        return policy_tp1, logs


    def extragradient_policy_mirror_descent(self, t, policy_t, history_t=None):
        # Exact PMD
        d_t = self.stationary_distribution(policy_t)
        q_t = self.policy_evaluation(policy_t)

        # Add noise to the Q-values
        inexact_q_t = self.add_noise(q_t, self.tau_mean1, self.tau_scale1)
        err_q_t = inexact_q_t - q_t

        # Calculate the optimal step size
        step_size_t = self.optimal_step_size(t, inexact_q_t, policy_t)

        # Compute the policy gradient for the current policy
        policy_gradient_fn = lambda theta: self.inexact_policy_gradient(theta, d_t, inexact_q_t, policy_t, step_size_t)

        # Perform inner loop gradient descent to obtain a proposal policy
        proposal_tp1, avg_gradient_t = self.inner_loop_gradient_descent(policy_t, policy_gradient_fn, self.k)

        # Prepare logs for monitoring
        logs = {
            "d_t": d_t,
            "step_size_t": step_size_t,
            "q_t": inexact_q_t,
            "err_q_t": err_q_t,
            "grad_t": np.linalg.norm(avg_gradient_t),
        }

        # Exact policy evaluation for the proposal policy
        q_tp1 = self.policy_evaluation(proposal_tp1)

        # Add noise to the Q-values of the proposal policy
        inexact_q_tp1 = self.add_noise(q_tp1, self.tau_mean2, self.tau_scale2)
        err_q_tp1 = inexact_q_tp1 - q_tp1
        d_tp1 = self.stationary_distribution(proposal_tp1)

        # Calculate the optimal step size for the next iteration
        step_size_tp1 = self.optimal_step_size(t, inexact_q_tp1, policy_t)
        if self.use_d == 1:
            d = d_tp1
        else:
            d = d_t
        # Compute the policy gradient for the proposal policy
        policy_gradient_fn = lambda theta: self.inexact_policy_gradient(theta, d, inexact_q_tp1, policy_t, step_size_tp1)

        # Perform inner loop gradient descent for the proposal policy
        policy_tp1, avg_gradient_tp1 = self.inner_loop_gradient_descent(policy_t, policy_gradient_fn, self.n)

        # Update logs with information for the proposal policy
        logs.update({
            "d_tp1": d_tp1,
            "q_tp1": inexact_q_tp1,
            "proposal_tp1": proposal_tp1,
            "step_size_tp1": step_size_tp1,
            "step_ratio": step_size_t / step_size_tp1,
            "err_q_tp1": err_q_tp1,
            "grad_tp1": np.linalg.norm(avg_gradient_tp1),
        })

        return policy_tp1, logs


    def optimal_step_size(self, t, q_t, policy_t):
        # Greedify the policy based on the current Q-function
        policy_greedy_tp1 = self.greedification(q_t)

        # Get the parameters of the current policy and the greedified policy
        theta_t = policy_t.get_params()
        theta_greedy_tp1 = policy_greedy_tp1.get_params()

        # Compute the KL-divergence between the greedified policy and the current policy
        RKL_greedy_t = utils.categorical_kl_divergence(theta_greedy_tp1, jax.lax.stop_gradient(theta_t))  # KL(pi_greedy, pi_t)

        # Compute the regularization decay parameter epsilon_t
        eps_t = self.env.gamma ** (2*t+1) * self.eps_0

        # Compute the optimal step size using the KL-divergence and epsilon_t
        step_size_t = RKL_greedy_t / eps_t

        # Return the optimal step size
        return step_size_t


    def exact_policy_gradient(self, theta_k, policy_t, step_size_t):

        # Exact stationary distribution
        d_t = self.stationary_distribution(policy_t)

        # Exact policy evaluation
        q_t = self.policy_evaluation(policy_t)

        pg = self.surrogate_objective(theta_k, policy_t.get_params(), d_t, q_t, step_size_t)

        return pg


    def inexact_policy_gradient(self, theta_k, d_t, q_t, policy_t, step_size_t):
        """
        Computes the inexact policy gradient.

        Args:
        - theta_k: Parameters of the policy at the current iteration.
        - d_t: Stationary distribution for the current policy.
        - q_t: Q-values for the current policy.
        - policy_t: Current policy object.
        - step_size_t: Optimal step size for the current iteration.

        Returns:
        - Gradient of the surrogate objective function.
        """
        # Calculate the surrogate objective function
        return self.surrogate_objective(theta_k, policy_t.get_params(), d_t, q_t, step_size_t)


    def inner_loop_gradient_descent(self, policy_t, policy_gradient_fn, param_steps):
        # Retrieve the current parameters of the policy and stop gradient computation
        theta_k = jax.lax.stop_gradient(policy_t.get_params())

        # Initialize mean gradient
        mean_grad = np.zeros_like(policy_gradient_fn(theta_k))

        # Iterate for param_steps to perform gradient descent updates
        for k in range(param_steps):
            # Compute gradient of the policy with respect to current parameters
            grad_k = policy_gradient_fn(theta_k)

            # Update parameters using gradient descent update rule
            theta_kp1 = theta_k - self.inner_loop_lr * grad_k

            # Stop gradient computation for the updated parameters
            theta_k = jax.lax.stop_gradient(theta_kp1)

            # Accumulate gradient for computing mean gradient
            mean_grad += grad_k

        # Create a new policy instance for the updated parameters
        policy_tp1 = policy.Policy()
        policy_tp1.set_params(theta_kp1)

        # Compute mean gradient by averaging accumulated gradients
        mean_grad /= param_steps

        # Return the updated policy and mean gradient
        return policy_tp1, mean_grad


    def policy_improvement(self, t, policy_t, history_t):
        """
        Perform policy improvement based on the selected algorithm.

        Args:
            t: Current iteration.
            policy_t: Current policy.
            history_t: Dictionary to store history.

        Returns:
            Parameters of the updated policy and step size.
        """
        # Policy improvement types
        policy_improvement_types = {
            "interpolation": self.policy_interpolation,
            "PI": self.policy_iteration,
            "PMD": self.policy_mirror_descent,
            "PMD(+lookahead)": self.lookahead_policy_mirror_descent,
            "PMD(+lazy_correction)": self.lazy_optimistic_correction_policy_mirror_descent,
            "PMD(+extragradient)": self.extragradient_policy_mirror_descent,
            "PMD(+correction)": self.optimistic_correction_policy_mirror_descent,
            "PMD(+correction_v2)": self.optimistic_correction_policy_mirror_descent_v2,
            "PMD(+lazy_correction_v2)": self.lazy_optimistic_correction_policy_mirror_descent_v2,
            "PMD(+lazy_momentum)": self.lazy_momentum,
            "PMD(+lazy_momentum_v2)": self.lazy_momentum_v2,
        }

        if self.policy_improvement_type not in policy_improvement_types:
            raise ValueError(f"Unknown policy_improvement_type: {self.policy_improvement_type}")

        return policy_improvement_types[self.policy_improvement_type](t, policy_t, history_t)

   #  def get_return(self, pi, max_traj):
   #      for m,(s,a,r,sp) in enumerate(self.monte_carlo(pi), start=1):
   #          if m >= max_traj:
   #              break
   #          if learner.update(s, a, r, sp):
   #             break
   #          callback(t, learner)
   #
   # def monte_carlo(self, pi, s=None, a=None):
   #     yield from self.simulate(pi, s=s, a=a, mode='reset')
   #
   # def step(self, s, a):
   #     sp = sample(self.P[s,a,:])
   #     if self.rng.uniform(0,1) <= 1-self.gamma:
   #         sp = self.start()
   #     r = self.R[s,a,sp]
   #     return r, sp
   #
   # def start(self):
   #     return sample(self.rho)



    def iterative_optimization(self, base_log_dir):
        """
        Perform iterative optimization over the specified number of iterations.

        Returns:
        - logs: Dictionary containing logged metrics during optimization.
        """
        if self.load_logs:
            folder_path = self.folder_path
            if base_log_dir is not None:
                folder_path = os.path.join(folder_path, base_log_dir)

            if os.path.exists(folder_path):
                data_path = os.path.join(folder_path, f"data.npz")
                logs = np.load(data_path, allow_pickle=True)
                print(f" Loaded data logs from '{base_log_dir}'")
                return logs


        logs = defaultdict(list)

        # Initialize policies and Q-values

        policy_t = self.policy_0
        policy_tm1 = self.policy_0
        q_tm1 = q_t = self.policy_evaluation(policy_t)

        # Iterate over the specified number of iterations
        for t in range(self.num_iter):

            # Prepare history for current iteration
            history_t = {"policy_tm1": policy_tm1, "q_tm1": q_tm1}

            # Perform policy improvement for the current iteration
            policy_tp1, logs_t = self.policy_improvement(t, policy_t, history_t)

            # Compute and log metrics for the current iteration
            # Update policies and Q-values for next iteration
            policy_tm1 = policy_t
            q_t = logs_t["q_t"]
            q_tm1 = q_t
            # print(f"t:{t}\n q_t:{q_t} \n pi_tp1:{policy_tp1.get_pi()}")

            logs = logger.log_metrics(logs, self, t, policy_tp1, policy_t, history_t, logs_t)
            policy_t = policy_tp1

        folder_path = self.folder_path
        if base_log_dir is not None:
            folder_path = os.path.join(folder_path, base_log_dir)

        if self.load_logs:
            print(f"Folder '{folder_path}' does not exist.")

        if self.save_logs:
            # Check if folder exists
            if not os.path.exists(folder_path):
                # If not, create it
                os.makedirs(folder_path)
                print(f"Folder '{folder_path}' created successfully.")

            data_path = os.path.join(folder_path, f"data.npz")
            np.savez(data_path, **logs)
            print(f"Saved data logs")
                  # f" in '{data_path}'.")

            config_path = os.path.join(folder_path, 'config.yaml')

            with open(config_path, 'w') as file:
                yaml.dump(self.kwargs, file)
            print(f"Saved config in '{config_path}'.")

        return logs

    def optimization_landscape(self):
        """
        Plot the optimization landscape

        Returns:
        - logs: Dictionary containing logged metrics during optimization.
        """
        logs = defaultdict(list)

        # Initialize policies and Q-values

        policy_tm1 = self.policy_mix[0]
        q_tm1 = self.policy_evaluation(policy_tm1)

        # Iterate over the specified number of iterations
        for t in range(self.num_iter):

            # Prepare history for current iteration
            history_t = {"policy_tm1": policy_tm1, "q_tm1": q_tm1}

            policy_t = self.policy_mix[::-1][t]

            # Perform policy improvement for the current iteration
            policy_tp1, logs_t = self.policy_improvement(t, policy_t, history_t)

            # Compute and log metrics for the current iteration
            logs = logger.log_metrics(logs, self, t, policy_tp1, policy_t, history_t, logs_t)
            # Update policies and Q-values for next iteration
            policy_tm1 = policy_t
            q_t = logs_t["q_t"]
            q_tm1 = q_t

        return logs


