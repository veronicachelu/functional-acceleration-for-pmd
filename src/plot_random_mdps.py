import matplotlib as mpl
import matplotlib.pyplot as plt
import colormaps as cmaps
import numpy as np
import importlib
import os
import src.utils as utils
from collections import defaultdict
import matplotlib.ticker as mticker

# '#DE8F05','#CC78BC','#CA9161',
all_colors_style = ['#0173B2', '#B20173', '#73B201', '#4001B2', '#B24001', '#B2011B', '#01B298', '#B29801', 'k']
BACKGROUND_COLOR = '#808080'
BACKGROUND_ALPHA = 0.
BACKGROUND_ALPHA_POLYTOPE = 0.5
BACKGROUND_LW = 1
LEGEND_SIZE = 10
LEGEND_LOCATION = "best"
LEGEND_MARKER_SCALE = 0.5
BACKGROUND_S = 10
BACKGROUND_FONTSIZE = 6
TICKLABELSIZE = 8
LABELSIZE = 10
PILABELSIZE = 12
CMAP="Grays"
LEVELS=3
def plot_learning_curve_per_ax(data=None, no_seeds= None, alg_cmaps=None, labels=None, initial_policy=None,
                               with_k=None, kk=None, mdps=None, T=None, ax=None):
    ts = range(T)
    for alg in range(len(data)):
        vals_seeds = []

        for mdp_no, mdp in enumerate(mdps):
            vals = []
            for i in range(no_seeds):
                if with_k[alg]:
                    m = data[alg][kk][2][mdp_no][initial_policy][i]["suboptimality_tp1__rho"][:T]
                else:
                    m = data[alg][2][mdp_no][initial_policy][i]["suboptimality_tp1__rho"][:T]
                vals.append(m)

            if no_seeds > 1:
                m = np.mean(vals, axis=0)

            vals_seeds.append(m)

        ax.plot(ts, np.mean(vals_seeds, axis=0), color=alg_cmaps[alg].colors[-1],
                lw=2, ls='-', label=labels[kk][alg], alpha=0.5, zorder=1000)
        ax.fill_between(ts, np.mean(vals_seeds, axis=0) - np.std(vals_seeds, axis=0),
                        np.mean(vals_seeds, axis=0) + np.std(vals_seeds, axis=0),
                        color=alg_cmaps[alg].colors[-1], alpha=0.2)

def plot_joint_learning_curve(k_list, mdp_sweep, with_k,
                        initial_policy, labels, annotations,
                        data, alg_cmaps, T, savefig=False,
                        figname="output"):
    no_seeds = len(data[0][0][2][0][initial_policy])
    fig, axes = plt.subplots(nrows=len(k_list),
                             ncols=len(mdp_sweep),
                             figsize=(3*len(mdp_sweep), len(k_list)*3),
                             squeeze=False, sharex="row")


    for kk, k in enumerate(k_list):
        for mdps_no, mdps in enumerate(mdp_sweep):
            ax = axes[kk][mdps_no]
            plot_learning_curve_per_ax(data=[data[alg][mdps_no] for alg in range(len(data))],
                                       no_seeds=no_seeds, alg_cmaps=alg_cmaps, labels=labels,
                                       initial_policy=initial_policy,
                                       with_k=with_k, kk=kk, mdps=mdps, T=T, ax=ax)

            ymin, ymax = ax.get_ylim()
            xmin, xmax = ax.get_xlim()
            ax.text(xmin+(xmax-xmin)*1/8, ymax-(ymax-ymin)/8,
                    f"$b:{annotations[kk][mdps_no]}, k:{k}$",
                    fontsize=10, color='gray',
                    bbox=dict(facecolor='gray',
                              alpha=0.1,
                              edgecolor='gray',
                              boxstyle='round,pad=0.1'))
            if mdps_no == 1 and kk == 2:
                ax.legend(
                    prop={"size": 12},
                    labelcolor="gray",
                    markerscale=0.1,
                    bbox_to_anchor=(0.8, -0.3),
                    loc='upper center',
                    labelspacing=0.1,
                    borderaxespad=-0.1,
                    ncol=3,
                    columnspacing=2.,
                    borderpad=-0.1,
                    handletextpad=0.1)
            if mdps_no == 0:
                ax.set_ylabel(r'$V^\rho_{\pi}$', fontsize=12)
            if kk == 2:
                ax.set_xlabel(r'#iterations($T$)', fontsize=12)

            ax.xaxis.set_tick_params(pad=-1)
            ax.yaxis.set_tick_params(pad=-1)
            ax.tick_params(axis='x', labelsize=10)
            ax.tick_params(axis='y', labelsize=10)

    if savefig:
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Define the figures folder
        folder_name = "figures"

        # Construct the full path to the folder
        current_dir = os.path.join(current_dir, "..")
        folder_path = os.path.join(current_dir, folder_name)

        # Check if folder exists
        if not os.path.exists(folder_path):
            # If not, create it
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created successfully.")
        # else:
        #     print(f"Folder '{folder_path}' already exists.")
        # Define the figure files
        pathname = os.path.join(folder_path, f"{figname}.png")
        # Save it
        plt.savefig(pathname, bbox_inches='tight', dpi=300)

    plt.show()



def plot_learning_curve(k_list, mdps, with_k, legend_bbox,
                        initial_policy, labels, annotations,
                        data, alg_cmaps, T, savefig=False,
                        figname="output"):

    fig, axes = plt.subplots(nrows=1,
                             ncols=len(k_list),
                             figsize=(3*len(k_list), 3),
                             squeeze=False, sharex="row",
                             sharey="col")
    no_seeds = len(data[0][2][0][initial_policy])

    for kk, k in enumerate(k_list):
        ax = axes[0][kk]
        plot_learning_curve_per_ax(data=data,
                                   no_seeds=no_seeds, alg_cmaps=alg_cmaps, labels=labels,
                                   initial_policy=initial_policy,
                                   with_k=with_k, kk=kk, mdps=mdps, T=T, ax=ax)

        ymin, ymax = ax.get_ylim()
        xmin, xmax = ax.get_xlim()
        ax.text(xmin+(xmax-xmin)*1/8, ymax-(ymax-ymin)/8,
                f"$b:{annotations[kk]}, k:{k}$",
                fontsize=10, color='gray',
                bbox=dict(facecolor='gray',
                          alpha=0.1,
                          edgecolor='gray',
                          boxstyle='round,pad=0.1'))
        if kk == int(len(k_list)/2):
            ax.legend(
                prop={"size": 12},
                labelcolor="gray",
                markerscale=0.1,
                bbox_to_anchor=legend_bbox,
                loc='upper right',
                labelspacing=0.1,
                borderaxespad=-0.1,
                ncol=3,
                columnspacing=2.,
                borderpad=-0.1,
                handletextpad=0.1)
        if kk == 0:
            ax.set_ylabel(r'$V^\rho_{\pi}$', fontsize=12)
        ax.set_xlabel(r'#iterations($T$)', fontsize=12)

        ax.xaxis.set_tick_params(pad=-1)
        ax.yaxis.set_tick_params(pad=-1)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)

    if savefig:
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Define the figures folder
        folder_name = "figures"

        # Construct the full path to the folder
        current_dir = os.path.join(current_dir, "..")
        folder_path = os.path.join(current_dir, folder_name)

        # Check if folder exists
        if not os.path.exists(folder_path):
            # If not, create it
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created successfully.")
        # else:
        #     print(f"Folder '{folder_path}' already exists.")
        # Define the figure files
        pathname = os.path.join(folder_path, f"{figname}.png")
        # Save it
        plt.savefig(pathname, bbox_inches='tight', dpi=300)

    plt.show()

def calculate_metrics_sweep_k(data, mdps, no_seeds, with_k, initial_policy, T, sweep_dim_list,final=False):
    mean_mdp_seeds = defaultdict(list)
    std_mdp_seeds = defaultdict(list)
    mean_cond_no_seeds = defaultdict(list)
    std_cond_no_seeds = defaultdict(list)
    beginning = T-1 if final else 0
    for kk, k in enumerate(sweep_dim_list):
        for alg, alg_datum in enumerate(data):
            vals_seeds = []
            cond_no_seeds = []
            for mdp_no, mdp in enumerate(mdps):
                vals = []
                cond_no = []
                for i in range(no_seeds):
                    if with_k[alg]:
                        subopt_traj = alg_datum[kk][2][mdp_no][initial_policy][i]["suboptimality_tp1__rho"][beginning:T]
                        policies_t = alg_datum[kk][2][mdp_no][initial_policy][i]["policy_t"][beginning:T]
                        # kappa_traj = alg_datum[kk][2][mdp_no][initial_policy][i]["improvement_t__d_t"][:T]
                        # kappa_traj = [utils.entropy(policy.get_params()).mean(-1) for policy in policies_t]
                    else:
                        subopt_traj = alg_datum[2][mdp_no][initial_policy][i]["suboptimality_tp1__rho"][beginning:T]
                        policies_t = alg_datum[2][mdp_no][initial_policy][i]["policy_t"][beginning:T]
                        # kappa_traj = alg_datum[2][mdp_no][initial_policy][i]["improvement_t__d_t"][:T]
                        # kappa_traj = [utils.entropy(policy.get_params()).mean(-1) for policy in policies_t]

                    m = np.cumsum(subopt_traj)[-1]
                    kappa_traj = [mdp.get_cond_no(policy.get_pi()) for policy in policies_t]

                    # kappa_traj = [mdp.mrp(policy.get_pi()).get_cond_no() for policy in policies_t]
                    kappa = np.mean(kappa_traj, axis=0)
                    # kappa = kappa_traj[-1]
                    vals.append(m)
                    cond_no.append(kappa)
                if no_seeds > 1:
                    m = np.mean(vals, axis=0)
                    kappa = np.mean(cond_no, axis=0)
                cond_no_seeds.append(kappa)
                vals_seeds.append(m)
            mean_mdp_seeds[alg].append(np.mean(vals_seeds, axis=0))
            std_mdp_seeds[alg].append(np.std(vals_seeds, axis=0))
            mean_cond_no_seeds[alg].append(np.mean(cond_no_seeds, axis=0))
            std_cond_no_seeds[alg].append(np.std(cond_no_seeds, axis=0))
    return mean_mdp_seeds, std_mdp_seeds, mean_cond_no_seeds, std_cond_no_seeds


def plot_metrics_sweep_k(mean_mdp_seeds, std_mdp_seeds, mean_cond_no_seeds,
                 std_cond_no_seeds, x, labels, alg_cmaps, markers,
                 legend_bbox, annotation, y_twin_label, y_lims=None,
                 labelsize=LABELSIZE, ticklabelsize=TICKLABELSIZE,
                 legendsize=LABELSIZE, textsize=TICKLABELSIZE,
                 labelcolor='gray', y_twin_lims=None,
                 log_scale=False, twin_log_scale=False, savefig=False,
                 offset_annot_x=1/20, offset_annot_y=1/20, legend_loc=None,
                 figname="output", x_label=None, y_label=None,
                 ):
    fig = plt.figure(figsize=(3, 3))
    ax = fig.gca()
    ax_twin = ax.twinx()
    for alg in range(len(mean_mdp_seeds)):
        mean_mdp_seed = np.array(mean_mdp_seeds[alg])
        std_mdp_seed = np.array(std_mdp_seeds[alg])
        mean_cond_no_seed = np.array(mean_cond_no_seeds[alg])
        std_cond_no_seed = np.array(std_cond_no_seeds[alg])

        ax.plot(x, mean_mdp_seed, linestyle='-', marker=markers[alg], markersize=6,
                label=labels[alg], color=alg_cmaps[alg].colors[int(len(alg_cmaps[alg].colors)/2)])
        ax.fill_between(x, mean_mdp_seed - std_mdp_seed, mean_mdp_seed + std_mdp_seed,
                        color=alg_cmaps[alg].colors[int(len(alg_cmaps[alg].colors)/2)], alpha=0.2)
        # if alg == 1:
        #     ax_twin.plot(x, mean_cond_no_seed, linestyle='--', marker=markers[alg],
        #                  markersize=3, alpha=0.5, label=labels[alg], color=labelcolor)
        ax_twin.plot(x, mean_cond_no_seed, linestyle='--', marker=markers[alg], markersize=3,
                     alpha=0.5, label=labels[alg], color=alg_cmaps[alg].colors[int(len(alg_cmaps[alg].colors)/2)])
        ax_twin.fill_between(x, mean_cond_no_seed - std_cond_no_seed,
                             mean_cond_no_seed + std_cond_no_seed,
                             color=alg_cmaps[alg].colors[int(len(alg_cmaps[alg].colors)/2)], alpha=0.2)

    ax.set_xlabel(x_label, fontsize=labelsize)
    ax.tick_params(axis='x', labelsize=ticklabelsize)
    ax.tick_params(axis='y', labelsize=ticklabelsize)
    ax.set_ylabel(y_label, fontsize=labelsize)
    ax.set_xticks(x)
    ax_twin.set_ylabel(y_twin_label, fontsize=labelsize, color=labelcolor)
    ax_twin.tick_params(axis='y', labelsize=ticklabelsize, labelcolor=labelcolor)

    if log_scale:
        ax.set_yscale("log")
    if twin_log_scale:
        ax_twin.set_yscale("log")
        ax_twin.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax_twin.yaxis.get_major_formatter().set_scientific(False)
        ax_twin.yaxis.get_major_formatter().set_useOffset(False)

    if y_lims is not None:
        ax.set_ylim(y_lims[0], y_lims[1])

    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()

    ax.text(xmin + (xmax - xmin) * offset_annot_x, ymax - (ymax - ymin) * offset_annot_y,
            annotation, fontsize=textsize, color=labelcolor, bbox=dict(facecolor='gray', alpha=0.1, edgecolor='gray', boxstyle='round,pad=0.1'))
    ax.legend(prop={"size": legendsize}, labelcolor=labelcolor, markerscale=1.,
              bbox_to_anchor=legend_bbox, loc=legend_loc, labelspacing=0.1,
              borderaxespad=-0.1, columnspacing=2., borderpad=-0.1,
              handletextpad=0.1)

    ax_twin.spines['right'].set_linestyle("--")
    ax_twin.spines['right'].set_visible(True)
    ax_twin.minorticks_off()
    ax.minorticks_off()

    if y_twin_lims is not None:
        ax_twin.set_ylim(y_twin_lims[0], y_twin_lims[1])
    if savefig:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
        os.makedirs(save_path, exist_ok=True)
        pathname = os.path.join(save_path, f"{figname}.png")
        plt.savefig(pathname, bbox_inches='tight', dpi=300)

    plt.show()

def calculate_metrics_sweep_mdp_param(data, no_seeds, initial_policy, T, sweep_mdp_dim_list, final=False):
    mean_mdp_seeds = defaultdict(list)
    mean_cond_no_seeds = defaultdict(list)
    mean_entropies_seeds = defaultdict(list)
    std_mdp_seeds = defaultdict(list)
    std_cond_no_seeds = defaultdict(list)
    std_entropies_seeds = defaultdict(list)
    beginning = T-1 if final else 0
    for alg in range(len(data)):
        alg_data = data[alg]
        for mdps_no, mdps in enumerate(sweep_mdp_dim_list):
            vals_seeds = []
            cond_no_seeds = []
            entropies_seeds = []
            for mdp_no, mdp in enumerate(mdps):
                vals = []
                cond_no = []
                entropies = []
                for i in range(no_seeds):
                    subopt_traj = alg_data[mdps_no][2][mdp_no][initial_policy][i]["suboptimality_tp1__rho"][beginning:T]
                    m = np.cumsum(subopt_traj)[-1]
                    vals.append(m)
                    policies_t = alg_data[mdps_no][2][mdp_no][initial_policy][i]["policy_t"][beginning:T]
                    # kappa_traj = alg_data[mdps_no][2][mdp_no][initial_policy][i]["grad_t"][:T]
                    kappa_traj = [mdp.get_cond_no(policy.get_pi()) for policy in policies_t]
                    ent_traj = [utils.entropy(policy.get_params()).mean(-1) for policy in policies_t]
                    kappa = kappa_traj[0]
                    ent = ent_traj[0]
                    cond_no.append(kappa)
                    entropies.append(ent)
                if no_seeds > 1:
                    m = np.mean(vals, axis=0)
                    kappa = np.mean(cond_no, axis=0)
                    ent = np.mean(entropies, axis=0)
                vals_seeds.append(m)
                cond_no_seeds.append(kappa)
                entropies_seeds.append(ent)
            mean_mdp_seeds[alg].append(np.mean(vals_seeds, axis=0))
            mean_cond_no_seeds[alg].append(np.mean(cond_no_seeds, axis=0))
            mean_entropies_seeds[alg].append(np.mean(entropies_seeds, axis=0))
            std_mdp_seeds[alg].append(np.std(vals_seeds, axis=0))
            std_cond_no_seeds[alg].append(np.std(cond_no_seeds, axis=0))
            std_entropies_seeds[alg].append(np.std(entropies_seeds, axis=0))
    return (mean_mdp_seeds, std_mdp_seeds,
            mean_cond_no_seeds, std_cond_no_seeds,
            mean_entropies_seeds, std_entropies_seeds)

def plot_metrics_sweep_mdp_param(mean_mdp_seeds, std_mdp_seeds, mean_cond_no_seeds,
                                 std_cond_no_seeds, x, labels, alg_cmaps, markers,
                                 legend_bbox, annotation,  y_twin_label,
                                 y_lims=None, y_twin_lims=None, mean_entropies_seeds=None,
                                 std_entropies_seeds=None,
                                 labelsize=LABELSIZE, ticklabelsize=TICKLABELSIZE,
                                 legendsize=LABELSIZE, textsize=TICKLABELSIZE, labelcolor='gray',
                                 log_scale=False, twin_log_scale=False, savefig=False,
                                 offset_annot_x=1/20, offset_annot_y=1/20, legend_loc=None,
                                 figname="output", x_label=None, y_label=None):
    fig = plt.figure(figsize=(3, 3))
    ax = fig.gca()
    ax_twin = ax.twinx()

    for alg in range(len(mean_mdp_seeds)):
        mean_mdp_seed = np.array(mean_mdp_seeds[alg])
        std_mdp_seed = np.array(std_mdp_seeds[alg])
        mean_cond_no_seed = np.array(mean_cond_no_seeds[alg])
        std_cond_no_seed = np.array(std_cond_no_seeds[alg])
        if mean_entropies_seeds is not None and std_entropies_seeds is not None:
            mean_entropies_seed = np.array(mean_entropies_seeds[alg])
            std_entropies_seed = np.array(std_entropies_seeds[alg])

        ax.plot(x, mean_mdp_seed, linestyle='-', marker=markers[alg],
                markersize=6, label=labels[alg],
                color=alg_cmaps[alg].colors[int(len(alg_cmaps[alg].colors)/2)])
        ax.fill_between(x, mean_mdp_seed - std_mdp_seed,
                        mean_mdp_seed + std_mdp_seed,
                        color=alg_cmaps[alg].colors[int(len(alg_cmaps[alg].colors)/2)], alpha=0.2)
        if alg == 2:
            if mean_entropies_seeds is not None and std_entropies_seeds is not None:
                ax_twin.plot(x, mean_entropies_seed, linestyle='--', marker=markers[alg], markersize=3,
                             alpha=0.5, label=labels[alg], color=labelcolor)
                ax_twin.fill_between(x, mean_entropies_seed - std_entropies_seed,
                                     mean_entropies_seed + std_entropies_seed,
                                     color=labelcolor, alpha=0.2)
            else:
                ax_twin.plot(x, mean_cond_no_seed, linestyle='-.', marker=markers[alg], markersize=3,
                             alpha=0.5, label=labels[alg], color=labelcolor)
                ax_twin.fill_between(x, mean_cond_no_seed - std_cond_no_seed,
                                     mean_cond_no_seed + std_cond_no_seed,
                                     color=labelcolor, alpha=0.2)

        # ax_twin.plot(x, mean_cond_no_seed, linestyle='--', marker=markers[alg], markersize=3,
        #              alpha=0.5, label=labels[alg], color=alg_cmaps[alg].colors[int(len(alg_cmaps[alg].colors)/2)])
        # ax_twin.fill_between(x, mean_cond_no_seed - std_cond_no_seed,
        #                      mean_cond_no_seed + std_cond_no_seed,
        #                      color=alg_cmaps[alg].colors[int(len(alg_cmaps[alg].colors)/2)], alpha=0.2)

    if log_scale:
        ax.set_yscale("log")
    if twin_log_scale:
        ax_twin.set_yscale("log")
        ax_twin.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax_twin.yaxis.get_major_formatter().set_scientific(False)
        ax_twin.yaxis.get_major_formatter().set_useOffset(False)

    ax.set_xlabel(x_label, fontsize=labelsize)
    ax.set_ylabel(y_label, fontsize=labelsize)
    ax.tick_params(axis='x', labelsize=ticklabelsize)
    ax.tick_params(axis='y', labelsize=ticklabelsize)
    ax.set_xticks(x)
    ax_twin.set_ylabel(y_twin_label, fontsize=labelsize, color=labelcolor)
    ax_twin.tick_params(axis='y', labelsize=ticklabelsize, labelcolor=labelcolor)
    ax.minorticks_off()
    if y_lims is not None:
        ax.set_ylim(y_lims[0], y_lims[1])

    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    ax.text(xmin + (xmax - xmin) * offset_annot_x, ymax - (ymax - ymin) * offset_annot_y,
            annotation, fontsize=textsize, color=labelcolor, bbox=dict(facecolor='gray', alpha=0.1, edgecolor='gray', boxstyle='round,pad=0.1'))
    ax.legend(prop={"size": legendsize}, labelcolor=labelcolor,
              markerscale=1., bbox_to_anchor=legend_bbox,
              loc=legend_loc, labelspacing=0.1, borderaxespad=-0.1,
              columnspacing=2., borderpad=-0.1, handletextpad=0.1)

    ax_twin.spines['right'].set_linestyle("--")
    ax_twin.spines['right'].set_visible(True)
    ax_twin.minorticks_off()

    if y_twin_lims is not None:
        ax_twin.set_ylim(y_twin_lims[0], y_twin_lims[1])
    if savefig:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
        os.makedirs(save_path, exist_ok=True)
        pathname = os.path.join(save_path, f"{figname}.png")
        plt.savefig(pathname, bbox_inches='tight', dpi=300)


    plt.show()
