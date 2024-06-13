import numpy as np
import distrax
from collections import defaultdict

state_metric_keys = ["v_*", "v_t", "v_tp1", "<q_t,tp1>", "<q_tp1,tp1>",
                     "<q_t,greedy_tp1-tp1>",  "<q_t,greedy_tp1-t>",
                     "<q_t,tp1-t>", "BFY_tp1", "BFY_greedy",
                     'D(greedy,t)', "<delta_t,tp1>", "step_size_t",
                     "FY1_tp1", "FY2_tp1", "v_proposal_tp1", "BFY_tp1/2",
                     "BFY_tp1,tp1/2", "improvement_tp1/2", "improvement_tp1",
                     "grad_t", "grad_tp1", "improvement_t", "suboptimality_t",
                     "suboptimality_tp1", "suboptimality_tp1/2",
                     "(1/step_size_t)D(tp1, t)",
                     "D(tp1, t)", "D(t,tp1)", "lower_bound",
                     "surrogate_proposal_tp1", "surrogate_tp1",
                     "D(*,tp1)", "D(*,t)", "upper_bound"]

def compute_atomic_metrics(landscape, metrics):
    """
    Compute atomic metrics based on landscape and other provided metrics.

    Args:
        landscape: Instance of the landscape.
        metrics (dict): Dictionary containing various metrics.

    Returns:
        dict: Updated dictionary with computed atomic metrics.
    """
    t = metrics["t"]
    proposal_tp1 = metrics["proposal_tp1"]
    policy_tp1 = metrics["policy_tp1"]
    policy_t = metrics["policy_t"]
    history_t = metrics["history_t"]

    # Compute greedy policy, Q-values, and performances
    metrics["greedy_tp1"], _ = landscape.policy_iteration(t, policy_t, history_t)
    metrics["q__pi_tp1"] = landscape.policy_evaluation(policy_tp1)
    metrics["q__pi_t"] = landscape.policy_evaluation(policy_t)
    metrics["v_t"] = landscape.performance(policy_t)  # V(pi_0)
    metrics["v_tp1"] = landscape.performance(policy_tp1)
    metrics["v_*"] = landscape.v_star
    metrics["v_proposal_tp1"] = landscape.performance(proposal_tp1)

    return metrics

def compute_composed_metrics(landscape, metrics):
    """
    Compute composed metrics based on the atomic metrics.

    Args:
        metrics (dict): Dictionary containing various metrics.

    Returns:
        dict: Updated dictionary with computed composed metrics.
    """
    policy_tp1 = metrics["policy_tp1"]
    proposal_tp1 = metrics["proposal_tp1"]
    policy_t = metrics["policy_t"]
    step_size_t = metrics["step_size_t"]
    step_size_tp1 = metrics["step_size_tp1"]
    policy_star = metrics["policy_*"]
    greedy_policy_tp1 = metrics["greedy_tp1"]

    # Compute various composed metrics based on policies and Q-values
    metrics["<q_t,tp1-t>"] = np.sum(metrics["q_t"] * (policy_tp1.get_pi() - policy_t.get_pi()), -1)
    metrics["<q_t,tp1>"] = np.sum(metrics["q_t"] * policy_tp1.get_pi(), -1)
    metrics["<q_tp1,tp1>"] = np.sum(metrics["q_tp1"] * policy_tp1.get_pi(), -1)

    delta_tp1 = metrics["q_tp1"] - metrics["q_t"]
    metrics["<delta_t,tp1>"] = np.sum(delta_tp1 * policy_tp1.get_pi(), -1)
    metrics["<q_t,greedy_tp1-tp1>"] = np.sum(metrics["q_t"] * (greedy_policy_tp1.get_pi() - policy_tp1.get_pi()), -1)
    metrics["<q_t,greedy_tp1-t>"] = np.sum(metrics["q_t"] * (greedy_policy_tp1.get_pi() - policy_t.get_pi()), -1)

    theta_star = policy_star.get_params()
    theta_tp1 = policy_tp1.get_params()
    theta_proposal_tp1 = proposal_tp1.get_params()
    theta_t = policy_t.get_params()
    theta_greedy_tp1 = greedy_policy_tp1.get_params()

    # Compute KL divergences between policies
    def categorical_kl_divergence(p_logits, q_logits):
        return distrax.Softmax(p_logits, 1.).kl_divergence(distrax.Softmax(q_logits, 1.))

    metrics["D(greedy,t)"] = categorical_kl_divergence(theta_greedy_tp1, theta_t)  # KL(pi_greedy, pi_t)
    metrics["D(*,t)"] = categorical_kl_divergence(theta_star, theta_t)  # KL(pi_greedy, pi_t)
    metrics["D(proposal_tp1, t)"] = categorical_kl_divergence(theta_proposal_tp1, theta_t)  # KL(pi_t+1, pi_t)
    metrics["D(tp1, proposal_tp1)"] = categorical_kl_divergence(theta_tp1, theta_proposal_tp1)  # KL(pi_t+1, pi_t)
    metrics["D(tp1, t)"] = categorical_kl_divergence(theta_tp1, theta_t)  # KL(pi_t+1, pi_t)

    metrics["D(t,tp1)"] = categorical_kl_divergence(theta_t, theta_tp1)  # KL(pi_t, pi_t+1)
    metrics["D(greedy,tp1)"] = categorical_kl_divergence(theta_greedy_tp1, theta_tp1)  # KL(pi_greedy, pi_t+1)
    metrics["D(*,tp1)"] = categorical_kl_divergence(theta_star, theta_tp1)  # KL(pi_greedy, pi_t+1)

    # Compute bounds, surrogates, and other related metrics
    # step_size = np.array(metrics["D(greedy, t)"]) / (landscape.eps_0 * landscape.env.gamma**(2*metrics["t"]+1))
    metrics["lower_bound"] = (metrics["D(tp1, t)"] + metrics["D(t,tp1)"])
    if landscape.policy_improvement_type not in ["PI", "PMD(+lookahead)"]:
        metrics["(1/step_size_t)D(tp1, t)"] = (1/step_size_t) * metrics["D(tp1, t)"]

        metrics["BFY_tp1"] = (np.sum(metrics["q_t"] * (policy_tp1.get_pi() - policy_t.get_pi()), -1) -
                              (1/step_size_t) * metrics["D(tp1, t)"])
        metrics["BFY_greedy"] = (np.sum(metrics["q_t"] * (greedy_policy_tp1.get_pi() - policy_t.get_pi()), -1) -
                                (1/step_size_t) * metrics["D(greedy,t)"])
        metrics["BFY_tp1/2"] = (np.sum(metrics["q_t"] * (proposal_tp1.get_pi() - policy_t.get_pi()), -1) -
                              (1/step_size_t) * metrics["D(proposal_tp1, t)"])
        if landscape.policy_improvement_type not in ["PMD", "PMD(+lazy_correction)", "PMD(+lazy_momentum)"]:
            metrics["BFY_tp1,tp1/2"] = (np.sum(metrics["q_tp1"] * (policy_tp1.get_pi() - proposal_tp1.get_pi()), -1) -
                                  (1/step_size_tp1) * metrics["D(tp1, proposal_tp1)"])
    else:
        metrics["(1/step_size_t)D(tp1, t)"] = metrics["D(tp1, t)"]
        metrics["BFY_tp1"] = np.sum(metrics["q_t"] * (policy_tp1.get_pi() - policy_t.get_pi()), -1)
        metrics["BFY_greedy"] = np.sum(metrics["q_t"] * (greedy_policy_tp1.get_pi() - policy_t.get_pi()), -1)
        metrics["BFY_tp1/2"] = np.sum(metrics["q_t"] * (proposal_tp1.get_pi() - policy_t.get_pi()), -1)
        metrics["BFY_tp1,tp1/2"] = np.sum(metrics["q_tp1"] * (policy_tp1.get_pi() - proposal_tp1.get_pi()), -1)

    metrics["upper_bound"] = (metrics["D(greedy,t)"] - metrics["D(greedy,tp1)"] - metrics["D(tp1, t)"])
    metrics["surrogate_proposal_tp1"] = (np.sum(metrics["q_t"] * proposal_tp1.get_pi(), -1) -
                                         metrics["D(proposal_tp1, t)"])
    metrics["surrogate_tp1"] = np.sum(metrics["q_tp1"] * policy_tp1.get_pi(), -1) - metrics["D(tp1, t)"]
    metrics["FY1_tp1"] = metrics["surrogate_tp1"] - metrics["surrogate_proposal_tp1"]
    metrics["FY2_tp1"] = (np.sum(delta_tp1 * policy_tp1.get_pi(), -1) - metrics["D(tp1, proposal_tp1)"])

    return metrics

def expand_state_metrics(metrics, rho, d_t, d_tp1):
    """
    Expand state metrics by applying a weighting factor.

    Args:
        metrics (dict): Dictionary containing various metrics.
        rho (float): Weighting factor.

    Returns:
        dict: Updated dictionary with expanded state metrics.
    """

    expanded_metrics = defaultdict(list)
    for k, v in metrics.items():
        if k in state_metric_keys:
            expanded_metrics.update({f"{k}__rho": np.sum(v * rho)})
            expanded_metrics.update({f"{k}__d_t": np.sum(v * d_t)})
            expanded_metrics.update({f"{k}__d_tp1": np.sum(v * d_tp1)})
            expanded_metrics.update({f"{k}__uniform": np.linalg.norm(v)})
    metrics.update(expanded_metrics)
    return metrics

def compute_metrics(landscape, t, policy_tp1, policy_t, history_t, logs_t):
    """
    Compute various metrics based on the current state of the RL environment and policies.

    Args:
        landscape: Instance of the landscape.
        t (int): Current timestep.
        policy_tp1: Policy at timestep t+1.
        policy_t: Policy at timestep t.
        history_t (dict): History data at timestep t.
        logs_t (dict): Logs data at timestep t.

    Returns:
        dict: Dictionary containing computed metrics.
    """
    metrics = defaultdict(list)
    metrics["t"] = t
    metrics["prob_right"] = policy_tp1.get_pi()[0][0]
    metrics["policy_*"] = landscape.policy_star
    metrics["policy_tp1"] = policy_tp1
    metrics["proposal_tp1"] = logs_t["proposal_tp1"] if "proposal_tp1" in logs_t.keys() else policy_tp1

    metrics["policy_t"] = policy_t
    # d_t = landscape.stationary_distribution(policy_t)
    # d_t = logs_t["d_t"] if "d_t" in logs_t.keys() else dpi

    metrics["history_t"] = history_t
    metrics["grad_t"] = logs_t["grad_t"] if "grad_t" in logs_t.keys() else 0
    metrics["grad_tp1"] = logs_t["grad_tp1"] if "grad_tp1" in logs_t.keys() else 0
    metrics["step_size_t"] = logs_t["step_size_t"] if "step_size_t" in logs_t.keys() else 0
    metrics["step_ratio"] = logs_t["step_ratio"] if "step_ratio" in logs_t.keys() else 0
    metrics["step_size_tp1"] = logs_t["step_size_tp1"] if "step_size_tp1" in logs_t.keys() else 0
    metrics["step_size_tm1"] = logs_t["step_size_tm1"] if "step_size_tm1" in logs_t.keys() else 0

    metrics["q_t"] = logs_t["q_t"] if "q_t" in logs_t.keys() else 0
    metrics["q_tp1"] = logs_t["q_tp1"] if "q_tp1" in logs_t.keys() else 0
    metrics["q_tm1"] = logs_t["q_tm1"] if "q_tm1" in logs_t.keys() else 0
    metrics["delta_tm1"] = logs_t["delta_tm1"] if "delta_tm1" in logs_t.keys() else 0
    metrics["delta_t"] = logs_t["delta_t"] if "delta_t" in logs_t.keys() else 0

    metrics["err_q_t"] = logs_t["err_q_t"] if "err_q_t" in logs_t.keys() else 0
    metrics["err_q_tp1"] = logs_t["err_q_tp1"] if "err_q_tp1" in logs_t.keys() else 0

    metrics = compute_atomic_metrics(landscape, metrics)
    metrics = compute_composed_metrics(landscape, metrics)

    metrics["improvement_t"] = metrics["v_tp1"] - metrics["v_t"]
    metrics["improvement_tp1/2"] = metrics["v_proposal_tp1"] - metrics["v_t"]
    metrics["improvement_tp1"] = metrics["v_tp1"] - metrics["v_proposal_tp1"]
    metrics["suboptimality_t"] = metrics["v_*"] - metrics["v_t"]
    metrics["suboptimality_tp1"] = metrics["v_*"] - metrics["v_tp1"]
    metrics["suboptimality_tp1/2"] = metrics["v_*"] - metrics["v_proposal_tp1"]
    d_t = landscape.stationary_distribution(policy_t)
    d_tp1 = landscape.stationary_distribution(metrics["proposal_tp1"])
    d_t = logs_t["d_t"] if "d_t" in logs_t.keys() else d_t
    d_tp1 = logs_t["d_tp1"] if "d_tp1" in logs_t.keys() else d_tp1
    metrics = expand_state_metrics(metrics, landscape.env.rho, d_t, d_tp1)

    return metrics


def log_metrics(logs, *kwargs):
    """
    Log the computed metrics for later analysis or visualization.

    Args:
        logs (dict): Dictionary containing logs data.
        metrics (dict): Dictionary containing computed metrics.

    Returns:
        dict: Updated logs dictionary.
    """
    metrics = compute_metrics(*kwargs)
    # List of metrics keys to log
    metrics_to_log = ["t",  "policy_tp1",  "prob_right", "grad_t",
                      "q_t", "q_tp1", "q_tm1", "err_q_t", "err_q_tp1"]
    metrics_to_log.extend(metrics.keys())
    # Loop through metrics and log only the required ones
    for k, v in metrics.items():
        if k in metrics_to_log:
            logs[k].append(v)


    return logs

