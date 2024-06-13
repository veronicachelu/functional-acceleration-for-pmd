import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import src.utils
import importlib
import os


src.utils = importlib.reload(src.utils)
from src.utils import Halfspaces

# '#DE8F05','#CC78BC','#CA9161',
all_colors_style = ['#0173B2', '#B20173', '#73B201', '#4001B2', '#B24001', '#B2011B', '#01B298', '#B29801', 'k']
BACKGROUND_COLOR = '#808080'
BACKGROUND_ALPHA = 0.
BACKGROUND_ALPHA_POLYTOPE = 0.7
BACKGROUND_LW = 1
LEGEND_SIZE = 10
LEGEND_LOCATION = "best"
LEGEND_MARKER_SCALE = 0.5
BACKGROUND_S = 20
BACKGROUND_FONTSIZE = 6
TICKLABELSIZE = 8
LABELSIZE = 10
PILABELSIZE = 12
CMAP="Grays"
LEVELS=3


def plot_cons(P, R, gamma, xscale, yscale, alpha=1., lw=1,
              color='k', ax=None):
    S, A = R.shape
    assert S == 2

    C = [];
    b = []
    for s in range(S):
        for a in range(A):
            c = np.zeros(2)
            C.append(c)
            b.append(R[s, a])
            for sp in range(S):
                c[sp] = (s == sp) - gamma * P[s, a, sp]

    C = np.array(C);
    b = np.array(b)
    Halfspaces(-C, -b).viz(xscale[:-1], yscale[:-1], alpha=alpha,
                           color=color, lw=lw, ax=ax)


def contour_plot(f, xdomain, ydomain, plot_contour=True, color='viridis', fontsize=8, alpha=0.5, levels=None, ax=None):
    "Contour plot of a function of two variables."

    if ax is None: ax = plt.gca()
    [xmin, xmax, _] = xdomain
    [ymin, ymax, _] = ydomain
    X, Y = np.meshgrid(np.linspace(*xdomain), np.linspace(*ydomain))
    Z = np.array(
        [f(np.array([x, y])) for (x, y) in zip(X.flat, Y.flat)]).reshape(
        X.shape)
    contours = ax.contour(X, Y, Z, colors='gray', levels=5, alpha=alpha/4)
    # ax.clabel(contours, contours.levels, inline=True, fontsize=fontsize)
    if color is not None and plot_contour:
        ax.imshow(Z, extent=[xmin, xmax, ymin, ymax],
                  origin='lower', cmap=color,
                  alpha=0.)
    ax.spines['top'].set_visible(False)
    ax.spines['top'].set_color('gray')
    ax.spines['right'].set_visible(False)
    ax.spines['right'].set_color('gray')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)



def plot_policy_path(mdp, data, ax=None, cmap=None, plot_lines=True,
                     supress_policy_annotation=False,
                     label=None, color=None, T=None, fontsize=PILABELSIZE,
                     marker=".", last_marker="*", ls="-", s=30, lw=1,
                     alpha=0.5, zorder=1000, policy_label="policy_t"):
    for j, seed_data in enumerate(data):
        pis = []
        T = len(seed_data[policy_label]) if T is None else T
        for policy in seed_data[policy_label][:T]:
            pi = policy.get_pi()
            pis.append(pi)
        vpis = np.array([mdp.mrp(pi).get_V() for pi in pis])
        v_pi_T = vpis[-1]

        n_points = len(pis)
        if color is not None:
            colors = [color] * n_points
            alphas = np.linspace(0.3, 1., n_points)
        elif cmap is not None:
            colors = [cmap(1. * i / (n_points)) for i in range(n_points)]
            alphas = [alpha] * n_points

        for i in range(len(pis)):
            ax.scatter(vpis[i, 0], vpis[i, 1], s=s/3, color=colors[i], marker=marker, alpha=alphas[i], zorder=zorder)

            if not supress_policy_annotation:
                if i == len(pis) - 1:
                    if policy_label == "policy_t":
                        pol = r'$\pi_T$'
                    elif policy_label == "policy_tp1":
                        pol = r'$\pi_{T+1}$'
                    else:
                        pol = r'$\breve{\pi}_{T+1}$'
                    ax.text(vpis[i, 0]-0.6, vpis[i, 1]+0.2, pol,color=colors[i],fontsize=fontsize)
                if i == 0:
                    if policy_label == "policy_t":
                        start_pi = "0"
                        ax.text(vpis[i, 0]-0.4, vpis[i, 1]-0.4, '$\pi_{'+start_pi+'}$', color=colors[i],fontsize=fontsize)
                if i == int(len(pis)/2) and policy_label == "policy_t":
                    ax.text(vpis[i, 0]+0.1, vpis[i, 1]-0.4, '$\pi_{'+"T/2"+'}$',color=colors[i], fontsize=fontsize)

        if plot_lines:
            for i in range(n_points - 1):
                ax.plot(vpis[i:i + 2, 0], vpis[i:i + 2, 1], color=colors[i], lw=lw, ls=ls, alpha=alphas[i]/3, zorder=zorder)

        if j > 0:
            label = None
        ax.scatter(*v_pi_T, label=label, s=s/1.5, marker=last_marker, color=colors[-1], alpha=alphas[i], zorder=zorder)


def plot_vector_path(mdp, data, ax=None, cmaps=None, ext=False,
                     label=None,  T=None,  arrowsize=8, arrowscale=1.1,
                     tp1=True, plot_points=False, s=100, textsize=LABELSIZE,
                     alpha=0.5, zorder=1000, add_extra_label=True):
    for j, seed_data in enumerate(data):
        T = len(seed_data["policy_t"]) if T is None else T

        pis_t = [pi.get_pi() for pi in seed_data["policy_t"][:T]]
        vpis_t = np.array([mdp.mrp(pi).get_V() for pi in pis_t])
        v_pi_T = vpis_t[-1]

        pis_half = [pi.get_pi() for pi in seed_data["proposal_tp1"][:T]]
        vpis_half = np.array([mdp.mrp(pi).get_V()for pi in pis_half])

        pis_tp1 = [pi.get_pi() for pi in seed_data["policy_tp1"][:T]]
        vpis_tp1 = np.array([mdp.mrp(pi).get_V()for pi in pis_tp1])


        n_points = len(pis_t)
        if plot_points:
            colors = [[cmap(1. * i / (n_points)) for i in range(n_points)] for cmap in cmaps]
        else:
            colors = [[cmap.colors[-1] for i in range(n_points)] for cmap in cmaps]
        alphas = [alpha] * n_points
        for i in range(len(pis_t)):
            if plot_points:
                ax.scatter(vpis_t[i, 0], vpis_t[i, 1], s=s/3, color=colors[0][i],
                           marker='.', alpha=alphas[i], zorder=zorder)
                if i == len(pis_t) - 1:
                    pol = r'$\pi_T$'
                    ax.text(vpis_t[i, 0]-0.6, vpis_t[i, 1]+0.2,
                            pol,
                            color=colors[0][i],
                            fontsize=textsize)
                if i == 0:
                    pol = r'$\pi_0$'
                    ax.text(vpis_t[i, 0]-0.4, vpis_t[i, 1]-0.4,
                            pol,
                            color=colors[0][i],
                            fontsize=textsize)

            else:
                if tp1:
                    _ = ax.quiver(vpis_t[i, 0], vpis_t[i, 1],
                                  vpis_half[i, 0]-vpis_t[i, 0],
                                  vpis_half[i, 1]-vpis_t[i, 1],
                                  units='xy', scale=arrowscale,
                                  color=colors[0][i],
                                  headwidth=arrowsize, headlength=arrowsize,
                                  headaxislength=arrowsize, alpha=alphas[i],
                                  zorder=zorder)
                    extra_label = r'$[\breve{\pi}_{t}]$' if add_extra_label else r''
                    ax.scatter([], [], marker=r'$\nearrow$',
                               color=colors[0][int(len(colors)/2)], s=2,
                               label=(label+ extra_label if label is not None else None))

                if ext:
                    _ = ax.quiver(vpis_t[i, 0], vpis_t[i, 1],
                                  vpis_tp1[i, 0]-vpis_t[i, 0],
                                  vpis_tp1[i, 1]-vpis_t[i, 1],
                                  units='xy', scale=arrowscale, headwidth=arrowsize,
                                  headlength=arrowsize, headaxislength=arrowsize,
                                  color=colors[1][i],  alpha=alphas[i],
                                  zorder=zorder)

                else:
                    _ = ax.quiver(vpis_half[i, 0], vpis_half[i, 1],
                                  vpis_tp1[i, 0]-vpis_half[i, 0],
                                  vpis_tp1[i, 1]-vpis_half[i, 1],
                                  units='xy', scale=arrowscale, headwidth=arrowsize,
                                  headlength=arrowsize, headaxislength=arrowsize,
                                  color=colors[1][i],  alpha=alphas[i],
                                  zorder=zorder)
            extra_label = r'$[{\pi}_{t+1}]$' if add_extra_label else r''
            ax.scatter([], [], marker=r'$\nearrow$',
                       color=colors[1][int(len(colors)/2)], s=2,
                       label=(label + extra_label if label is not None else None))
            if label is not None:
                label = None
        if plot_points:
            ax.scatter(*v_pi_T, s=s/2, marker='*',
                       color=colors[0][-1], alpha=alphas[i], zorder=zorder)



def plot_background_stuff(mdp, v, r, ax,
                          fontsize=BACKGROUND_FONTSIZE,
                          s=BACKGROUND_S, color=BACKGROUND_COLOR,
                          lw=BACKGROUND_LW,
                          alpha_polytope=BACKGROUND_ALPHA_POLYTOPE,
                          alpha=BACKGROUND_ALPHA,
                          batch_param_kv=None,
                          text_pos=None,
                          plot_contour=True,
                          textsize=LABELSIZE,
                          T=None):
    offset = v[:, 0].ptp() * .05
    xs = [min(v[:, 0]) - offset, max(v[:, 0]) + offset, 100]
    ys = [min(v[:, 1]) - offset, max(v[:, 1]) + offset, 100]
    plot_cons(mdp.P, r, mdp.gamma, xs, ys,
              alpha=alpha_polytope, lw=lw,
              color=color, ax=ax)
    ax.grid(False)
    contour_plot(
        lambda v: (1 - mdp.gamma) * mdp.rho @ v,
        xs,
        ys,
        plot_contour=plot_contour,
        levels=LEVELS,
        alpha=alpha,
        color=CMAP,
        fontsize=fontsize,
        ax=ax
    )
    # offset = v[:, 0].ptp() * .05
    if T is not None:
        inverval_size_y = max(v[:, 0]) - min(v[:, 0])
        inverval_size_x = max(v[:, 1]) - min(v[:, 1])

        if text_pos is not None:
            offset_y = inverval_size_y/text_pos[1]
            offset_x = inverval_size_x/text_pos[0]
        else:
            offset_y = inverval_size_y/4
            offset_x = inverval_size_x/4

        ax.text(min(v[:, 0]), max(v[:, 1])- offset_y,
                f"{batch_param_kv}",
                fontsize=textsize, color="gray",
                bbox=dict(facecolor='gray',
                          alpha=0.1,
                          edgecolor='gray',
                          boxstyle='round,pad=0.1'))

        ax.text(max(v[:, 0]) - offset_x, min(v[:, 1]),
                f"$T={T}$", color="gray", fontsize=textsize,
                bbox=dict(facecolor='gray',
                          alpha=0.1,
                          edgecolor='gray',
                          boxstyle='round,pad=0.1'))
    ax.scatter(v[:, 0], v[:, 1], s=s, c=color,
               zorder=100, alpha=alpha_polytope)


def plot_colorbar(fig, ax, data, cmap, bbox_to_anchor, no_bounds, N, label, fontsize=TICKLABELSIZE):
    cax = inset_axes(ax,  # here using axis of the lowest plot
                     width="5%",  # width = 5% of parent_bbox width
                     height="100%",  # height : 340% good for a (4x4) Grid
                     loc='lower left',
                     bbox_to_anchor=bbox_to_anchor,
                     bbox_transform=ax.transAxes,
                     borderpad=0,
                     )

    if no_bounds:
        sm = plt.cm.ScalarMappable(cmap=cmap)
    else:
        if int(len(data[0]["policy_t"])) > 10:
            bounds = range(0, int(len(data[0]["policy_t"])/10))
        else:
            bounds = range(0, int(len(data[0]["policy_t"])))

        norm = mpl.colors.BoundaryNorm(bounds, cmap.N,
                                       extend='both',
                                       clip=False)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(bounds)

    cbar = fig.colorbar(sm, cax=cax,
                        drawedges=False,
                        extend='both',
                        spacing='uniform',
                        label=label)
    cbar.outline.set_visible(False)
    cbar.set_label(label, labelpad=-0.1)
    cbar.ax.minorticks_off()
    cbar.ax.xaxis.label.set_color('gray')
    cbar.ax.yaxis.label.set_color('gray')
    cbar.ax.tick_params(axis='x', colors='gray')
    cbar.ax.tick_params(axis='y', colors='gray')
    if not no_bounds:
        N = len(data[0]["policy_t"]) if N is None else N
        if N > 10:
            cbar.ax.set_yticklabels([b*10 for b in bounds])
        else:
            cbar.ax.set_yticklabels(bounds)
    else:
        cbar.ax.set_yticklabels([])

    cbar.ax.axes.tick_params(length=0)
    cbar.ax.yaxis.label.set_size(fontsize)
    cbar.ax.tick_params(labelsize=fontsize)



def plot_vector_fields(mdp, batch_param_kv_rows=None, data=None,  labels=None,
                                cmaps=None, T=None, text_pos=None, s=100, textsize=LABELSIZE,
                                alpha=None, legend_bbox=None, arrowsize=8, arrowscale=1.1,
                               savefig=False,ext=False,tp1=True,labelsize=LABELSIZE, ticklabelsize=TICKLABELSIZE,
                       legendsize=LEGEND_SIZE,titlesize=LABELSIZE,
                       labelcolor='gray',title=None,
                               figname="output",):
    no_rows = len(data)
    no_cols = len(data[0])
    fig, axes = plt.subplots(no_rows, no_cols,
                             figsize=(2*no_cols,4*no_rows),
                             subplot_kw={'aspect': 1},
                             sharey="col", sharex="row",
                             squeeze=False)
    fig.subplots_adjust(hspace=0., wspace=0.)
    det_policies = mdp.all_det_policies()
    v = np.array([mdp.mrp(pi).get_V() for pi in det_policies])
    r = mdp.r
    for row_no in range(no_rows):
        for col_no in range(no_cols):
            batch_param_kv = batch_param_kv_rows[row_no][col_no] if batch_param_kv_rows is not None else ''
            ax = axes[row_no][col_no]

            plot_background_stuff(mdp, v, r, ax, T=T, batch_param_kv=batch_param_kv, text_pos=text_pos, textsize=textsize)

            label = labels[row_no][col_no]
            data_vec = data[row_no][col_no]
            plot_vector_path(mdp, data_vec, label=label, alpha=alpha, T=T, arrowsize=arrowsize,
                             arrowscale=arrowscale, s=s, textsize=textsize,
                             cmaps=cmaps, ax=ax, zorder=1000, ext=ext, tp1=tp1)

            if col_no == int(no_cols/2):

                ax.legend(
                    prop={"size": legendsize},
                    labelcolor=labelcolor,
                    markerscale=5,
                    bbox_to_anchor=legend_bbox,
                    loc='upper center',
                    labelspacing=0.1,
                    borderaxespad=0.1,
                    ncol=3,
                    framealpha=0.5,
                    columnspacing=0.,
                    borderpad=0.1, handletextpad=-0.1)

            if row_no == len(data) - 1:
                ax.set_xlabel('$V_{\pi}(s_0)$', fontsize=labelsize)
            if col_no == 0:
                ax.set_ylabel('$V_{\pi}(s_1)$', fontsize=labelsize, labelpad=0)
            ax.tick_params(axis='x', labelsize=ticklabelsize)
            ax.tick_params(axis='y', labelsize=ticklabelsize)
            ax.xaxis.label.set_color(labelcolor)
            ax.yaxis.label.set_color(labelcolor)
            ax.tick_params(axis='x', colors=labelcolor)
            ax.tick_params(axis='y', colors=labelcolor)
            if col_no > 0:
                ax.tick_params(axis="y", labelleft=False)

    if title is not None:
        fig.suptitle(title, fontsize=titlesize)

    if savefig:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
        os.makedirs(save_path, exist_ok=True)
        pathname = os.path.join(save_path, f"{figname}.png")
        plt.savefig(pathname, bbox_inches='tight', dpi=300)

    plt.show()

def compare_vector_fields(mdp, batch_param_kv_rows=None, data=None,  labels=None,
                       cmaps=None, T=None, ext=False,tp1=True,
                       alpha=None, legend_bbox=None,text_pos=None,
                       savefig=False, arrowsize=8, arrowscale=1.1,
                          plot_points=None,legend_ncols=2,s=100,
                          labelsize=LABELSIZE, ticklabelsize=TICKLABELSIZE,
                          legendsize=LEGEND_SIZE, textsize=LABELSIZE,
                          labelcolor='gray', add_extra_label=True,
                       figname="output",):
    no_rows = len(data)
    no_cols = len(data[0])
    fig, axes = plt.subplots(no_rows, no_cols,
                             figsize=(2*no_cols,4*no_rows),
                             subplot_kw={'aspect': 1},
                             sharey="col", sharex="row",
                             squeeze=False)
    fig.subplots_adjust(hspace=0.,wspace=0.)
    det_policies = mdp.all_det_policies()
    v = np.array([mdp.mrp(pi).get_V() for pi in det_policies])
    r = mdp.r
    # legend_handles = []
    # legend_labels = []
    for row_no in range(no_rows):
        for col_no in range(no_cols):
            batch_param_kv = batch_param_kv_rows[row_no][col_no] if batch_param_kv_rows is not None else ''
            ax = axes[row_no][col_no]

            plot_background_stuff(mdp, v, r, ax, T=T, batch_param_kv=batch_param_kv,
                                  text_pos=text_pos, textsize=textsize)

            label = labels[row_no][col_no]
            data_vec = data[row_no][col_no]
            for l, inner_data_vec in enumerate(data_vec):
                inner_label = label[l]
                inner_cmaps = cmaps[l]
                plot_vector_path(mdp, inner_data_vec, label=inner_label, ext=ext, tp1=tp1,
                                 alpha=alpha, T=T, arrowscale=arrowscale,arrowsize=arrowsize,
                                 plot_points=plot_points,s=s,textsize=textsize, add_extra_label=add_extra_label,
                                 cmaps=inner_cmaps, ax=ax, zorder=1000-l)
            if col_no == int(no_cols/2):
                ax.legend(
                    prop={"size": legendsize},
                    labelcolor=labelcolor,
                    markerscale=5,
                    bbox_to_anchor=legend_bbox,
                    loc='upper center',
                    labelspacing=0.1,
                    borderaxespad=0.1,
                    ncol=legend_ncols,
                    framealpha=0.5,
                    columnspacing=0.1,
                    borderpad=0.1, handletextpad=0)

            if row_no == len(data) - 1:
                ax.set_xlabel('$V_{\pi}(s_0)$', fontsize=labelsize)
            if col_no == 0:
                ax.set_ylabel('$V_{\pi}(s_1)$', fontsize=labelsize, labelpad=0)
            ax.tick_params(axis='x', labelsize=ticklabelsize)
            ax.tick_params(axis='y', labelsize=ticklabelsize)
            ax.xaxis.label.set_color(labelcolor)
            ax.yaxis.label.set_color(labelcolor)
            ax.tick_params(axis='x', colors=labelcolor)
            ax.tick_params(axis='y', colors=labelcolor)
            if col_no > 0:
                ax.tick_params(axis="y", labelleft=False)

    if savefig:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
        os.makedirs(save_path, exist_ok=True)
        pathname = os.path.join(save_path, f"{figname}.png")
        plt.savefig(pathname, bbox_inches='tight', dpi=300)

    plt.show()

def compare_algs_batch_per_mdp(mdp, batch_param_kv_rows=None, data_rows1=None, labels_rows3=None,
                               data_rows2=None, data_rows3=None, labels_rows1=None, cmap1=None,
                               cmap2=None, cmap3=None, Ts=None, invert_axes=False,
                               lw=1, s=100, labels_rows2=None, marker='.', last_marker="*",
                               cmap4=None, alpha=None, legend_bbox=None, supress_policy_annotation=False,
                               data_rows4=None, plot_colorbar4=True, labels_rows4=None,
                               plot_colorbar1=True, plot_colorbar2=True, text_pos=None,
                               plot_colorbar3=True, savefig=False, plot_lines=True,
                               figname="output", policies1=None, policies2=None, policies3=None):
    no_rows = len(data_rows1)
    no_cols = len(data_rows1[0])
    # fig, axes = plt.subplots(no_rows, no_cols,
    #                          subplot_kw={'aspect': 1},
    #                          squeeze=False)
    # fig.subplots_adjust(hspace=0.01)
    fig, axes = plt.subplots(no_rows, no_cols,
                             figsize=(2*no_cols,4*no_rows),
                             subplot_kw={'aspect': 1},
                             sharey="col",
                             squeeze=False)
    fig.subplots_adjust(hspace=0.)
    det_policies = mdp.all_det_policies()
    v = np.array([mdp.mrp(pi).get_V() for pi in det_policies])
    r = mdp.r
    legend_handles = []
    legend_labels = []
    for row_no in range(no_rows):
        for col_no in range(no_cols):
            batch_param_kv = batch_param_kv_rows[row_no][col_no] if batch_param_kv_rows is not None else ''
            ax = axes[row_no][col_no]
            if invert_axes:
                ax = axes[col_no][row_no]
            T = Ts[row_no][col_no]

            plot_background_stuff(mdp, v, r, ax, T=T, batch_param_kv=batch_param_kv, text_pos=text_pos)


            if data_rows4 is not None:
                label4 = labels_rows4[row_no][col_no]
                data4 = data_rows4[row_no][col_no]
                plot_policy_path(mdp, data4, label=label4, alpha=alpha, T=T,
                                 plot_lines=plot_lines,
                                 supress_policy_annotation=supress_policy_annotation,
                                 marker=marker, last_marker=last_marker,
                                 cmap=cmap3,  ax=ax, s=s, lw=lw, zorder=1000)


            if data_rows3 is not None:
                label3 = labels_rows3[row_no][col_no]
                data3 = data_rows3[row_no][col_no]
                policy_label = policies3[row_no][col_no] if policies3 is not None else "policy_t"
                plot_policy_path(mdp, data3, label=label3, alpha=alpha, T=T,
                                 supress_policy_annotation=supress_policy_annotation,
                                 plot_lines=plot_lines, policy_label=policy_label,
                                 marker=marker, last_marker=last_marker,
                                 cmap=cmap3,  ax=ax, s=s, lw=lw, zorder=1000)

            if data_rows2 is not None:
                label2 = labels_rows2[row_no][col_no]
                data2 = data_rows2[row_no][col_no]
                policy_label = policies2[row_no][col_no] if policies2 is not None else "policy_t"
                plot_policy_path(mdp, data2, label=label2, alpha=alpha, T=T,
                                 marker=marker, last_marker=last_marker,
                                 supress_policy_annotation=supress_policy_annotation,
                                 plot_lines=plot_lines, policy_label=policy_label,
                                 cmap=cmap2,  ax=ax, s=s, lw=lw, zorder=1000)

            label1 = labels_rows1[row_no][col_no]
            data1 = data_rows1[row_no][col_no]
            policy_label = policies1[row_no][col_no] if policies1 is not None else "policy_t"
            plot_policy_path(mdp, data1, label=label1, alpha=alpha, T=T,
                             supress_policy_annotation=supress_policy_annotation,
                             plot_lines=plot_lines, policy_label=policy_label,
                             marker=marker, last_marker=last_marker,
                             cmap=cmap1, ax=ax, s=s, lw=lw, zorder=1000)
            if col_no == int(no_cols/2):
                h, l = ax.get_legend_handles_labels()
                legend_handles.extend(h)
                legend_labels.extend(l)
                ax.legend(
                    prop={"size": LEGEND_SIZE},
                    labelcolor="gray",
                    markerscale=LEGEND_MARKER_SCALE,
                    bbox_to_anchor=legend_bbox,
                    loc='upper center',
                    labelspacing=-0.1,
                    borderaxespad=0.,
                    ncol=3,
                    framealpha=0.5,
                    columnspacing=-0.2,
                    borderpad=0., handletextpad=-0.4)
                # ax.legend(
                #        prop={"size": LEGEND_SIZE},
                #        # loc=LEGEND_LOCATION,
                #        labelcolor="gray",
                #        markerscale=LEGEND_MARKER_SCALE,
                #        # bbox_to_anchor=(-1, 1.02, 1, 0.2),
                #        # loc="lower left",
                #        # mode="expand", borderaxespad=0,
                #     bbox_to_anchor=(0.5, -0.3),
                #     loc='upper center',
                #        labelspacing=-0.1,
                #        borderaxespad=-0.1,
                #        ncol=3,
                #     columnspacing=0.1,
                #        borderpad=-0.1, handletextpad=-0.4)


            if row_no == len(data_rows1) - 1:
                ax.set_xlabel('$V_{\pi}(s_0)$', fontsize=LABELSIZE)
            if col_no == 0:
                ax.set_ylabel('$V_{\pi}(s_1)$', fontsize=LABELSIZE)
            ax.xaxis.set_tick_params(pad=-5)
            ax.yaxis.set_tick_params(pad=-5)
            ax.tick_params(axis='x', labelsize=TICKLABELSIZE)
            ax.tick_params(axis='y', labelsize=TICKLABELSIZE)
            ax.xaxis.label.set_color('gray')
            ax.yaxis.label.set_color('gray')
            ax.tick_params(axis='x', colors='gray')
            ax.tick_params(axis='y', colors='gray')

    label_colormap = f'#updates ($T$)' if not plot_colorbar2 else None
    if plot_colorbar1:
        plot_colorbar(fig, ax, data1, cmap=cmap1, label=label_colormap,
                      no_bounds=True,
                      N=T,bbox_to_anchor=(1.25, 0.5, 0.8, 0.5))
    label_colormap = f'#updates ($T$)' if not plot_colorbar3 else None
    if plot_colorbar2:
        plot_colorbar(fig, ax, data2, cmap=cmap2, label=label_colormap,
                      no_bounds=True,
                      N=T, bbox_to_anchor=(1.31, 0.5, 0.8, 0.5))
    label_colormap = f'#updates ($T$)' if not plot_colorbar4 else None
    if plot_colorbar3:
        plot_colorbar(fig, ax, data3, cmap=cmap3, label=label_colormap,
                      no_bounds=True,
                      N=T,bbox_to_anchor=(1.37, 0.5, 0.8, 0.5))
    label_colormap = f'#updates ($T$)'
    if plot_colorbar4:
        plot_colorbar(fig, ax, data4, cmap=cmap4, label=label_colormap,
                      no_bounds=True,
                      N=T,bbox_to_anchor=(1.42, 0.5, 0.8, 0.5))

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
    #
    # fig.legend(handles=legend_handles,
    #            labels=legend_labels,
    #            prop={"size": LEGEND_SIZE},
    #          # loc=LEGEND_LOCATION,
    #          labelcolor="gray",
    #          markerscale=LEGEND_MARKER_SCALE,
    #          # bbox_to_anchor=(-1, 1.02, 1, 0.2),
    #          # loc="lower left",
    #          # mode="expand", borderaxespad=0,
    #            bbox_to_anchor=(0.5, 1.),
    #            loc="lower center",
    #            bbox_transform=fig.transFigure,
    #          labelspacing=0.1,
    #          # borderaxespad=-0.1,
    #          ncol=3,
    #          borderpad=-0.1, handletextpad=-0.4)
    plt.show()


def compare_inexact(mdp,  datas=None, labels=None, cmaps=None, Ns=None,
                    lw=1, s=100, marker='.', last_marker="*",
                    legend_bbox=None,textsize=LABELSIZE,
                     alpha=None, batch_param_kv_rows=None, supress_policy_annotation=None,
                    savefig=False, plot_colormaps=True,text_pos=None,
                    labelsize=LABELSIZE, ticklabelsize=TICKLABELSIZE,
                    legendsize=LEGEND_SIZE,titlesize=LABELSIZE,
                    labelcolor='gray',
                    plot_lines=True, figname="output"):
    no_cols = len(datas[0])
    no_rows = len(datas)
    fig, axes = plt.subplots(no_rows, no_cols,
                             figsize=(no_rows*5,no_cols*5),
                             subplot_kw={'aspect': 1},
                             sharey="col", sharex="row",
                             squeeze=False)
    fig.subplots_adjust(hspace=0., wspace=0.)
    # fig_width, fig_height = fig.get_size_inches()
    # fig, axes = plt.subplots(no_rows, no_cols,
    #                         figsize=(fig_width/3, fig_height*3),
    #                          subplot_kw={'aspect': 1},
    #                          squeeze=False)
    legend_handles = []
    legend_labels = []

    # fig.subplots_adjust(hspace=0.4, wspace=0.4)
    det_policies = mdp.all_det_policies()
    v = np.array([mdp.mrp(pi).get_V() for pi in det_policies])
    r = mdp.r
    for row_no in range(no_rows):
        for col_no in range(no_cols):
            batch_param_kv = batch_param_kv_rows[row_no][col_no] if batch_param_kv_rows is not None else ''
            ax = axes[row_no][col_no]
            N = Ns[row_no][col_no]
            # ax.text(-2.5, 1.7, f"$T={N}$", fontsize=13,
            #         bbox=dict(facecolor='gray',
            #                   alpha=0.1,
            #                   edgecolor='gray',
            #                   boxstyle='round,pad=0.1'))
            plot_background_stuff(mdp, v, r, ax, T=N, text_pos=text_pos,
                                  batch_param_kv=batch_param_kv,
                                  textsize=textsize)

            # plot_background_stuff(mdp, v, r, ax)
            label = labels[row_no][col_no]
            data = datas[row_no][col_no]
            cmap = cmaps[row_no][col_no]

            # ax.text(-1.8, -1.7, f"{label}",
            #     fontsize=13,
            #     bbox=dict(facecolor='gray',
            #               alpha=0.1,
            #               edgecolor='gray',
            #               boxstyle='round,pad=0.1'))

            plot_policy_path(mdp, data, label=label, alpha=alpha, T=N,
                             supress_policy_annotation=supress_policy_annotation,
                             plot_lines=plot_lines, fontsize=PILABELSIZE,
                             marker=marker, last_marker=last_marker,
                             cmap=cmap, ax=ax, s=s, lw=lw, zorder=1000)


            ax.set_xlabel('$V_{\pi}(s_0)$', fontsize=labelsize, labelpad=0)
            if col_no == 0:
                ax.set_ylabel('$V_{\pi}(s_1)$', fontsize=labelsize, labelpad=0)
            else:
                ax.tick_params(axis='y', labelleft=False)
            ax.tick_params(axis='x', labelsize=ticklabelsize)
            ax.xaxis.label.set_color(labelcolor)
            ax.tick_params(axis='x', colors=labelcolor)

            ax.yaxis.label.set_color(labelcolor)
            ax.tick_params(axis='y', colors=labelcolor)
            ax.tick_params(axis='y', labelsize=ticklabelsize)



            h, l = ax.get_legend_handles_labels()
            legend_handles.extend(h)
            legend_labels.extend(l)

            # ax.xaxis.set_tick_params(pad=-5)
            # ax.yaxis.set_tick_params(pad=-5)

            # if plot_colormaps:
            #     label_colormap = f'#updates ($T$)'
            #     plot_colorbar(fig, ax, data, cmap=cmap, label=label_colormap,
            #               no_bounds=True, fontsize=ticklabelsize,
            #               N=N, bbox_to_anchor=(1., 0.3, 0.8, 0.5))

    axes[0][int(no_cols/2)].legend(handles=legend_handles,
                  labels=legend_labels,
                  prop={"size": legendsize},
                  # loc=LEGEND_LOCATION,
                  labelcolor=labelcolor,
                  markerscale=LEGEND_MARKER_SCALE,
                  # bbox_to_anchor=(-1, 1.02, 1, 0.2),
                  # loc="lower left",
                  # mode="expand", borderaxespad=0,
                  bbox_to_anchor=legend_bbox,
                  # (0.5, -0.3),
                  loc='upper center',
                   labelspacing=0.1,
                   borderaxespad=0.1,
                   ncol=3,
                   framealpha=0.5,
                   columnspacing=0.,
                   borderpad=0.1, handletextpad=-0.1)
    if savefig:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
        os.makedirs(save_path, exist_ok=True)
        pathname = os.path.join(save_path, f"{figname}.png")
        plt.savefig(pathname, bbox_inches='tight', dpi=300)

    plt.show()


def plot_bars(data, tau1_list, colors, hatches,
              y_label=None, x_label=None,
              figname="output", savefig=False):

    width = 0.1 * len(data)  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(figsize=(2, 2))
    for k in data.keys():
        v = data[k]
        # print(data[tau_ind]["pmd"])
        offset = width * multiplier
        # ind = list(np.a)
        # print(v[ind])
        # print(x + offset)
        mean = [v[i][0] for i in range(len(v))]
        std = [v[i][1] for i in range(len(v))]
        x = np.arange(len(mean))  # the label locations

        ax.bar(x + offset, mean,
               width, yerr=std, label=k, color=colors[k].colors[int(len(colors[k].colors)/2)],
               edgecolor=colors[k].colors[-1],
               hatch=hatches[k])
        # print(data[tau_ind]["mom"])
        # ax.bar_label(rects, padding=3)
        # ax.bar(x + offset, data[tau_ind]["mom"], width, label=("mom" if tau_ind == 0 else None), color=newcmap3.colors[-1])
        # ax.bar_label(rects, padding=3)
        multiplier += 1


    # Add some text for labels, title and custom x-axis tick labels, etc.
    if y_label is not None:
        ax.set_ylabel(y_label)

    ax.set_xticks(x + width*17/16, tau1_list)
    if x_label is not None:
        ax.set_xlabel(x_label)
    ax.legend(loc='upper left', ncols=3)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(0, ymax + (ymax- ymin) * 1/3)

    ax.legend(
        prop={"size": LABELSIZE},
        labelcolor="gray",
        markerscale=0.01,
        # markersize=1,
        # bbox_to_anchor=legend_bbox,
        loc='best',
        labelspacing=0.1,
        borderaxespad=0.1,
        # ncol=3,
        framealpha=0.5,
        columnspacing=0.1,
        borderpad=0.1, handletextpad=0)
    # ax.xaxis.set_tick_params(pad=-5)
    # ax.yaxis.set_tick_params(pad=-5)
    ax.tick_params(axis='x', labelsize=TICKLABELSIZE)
    ax.tick_params(axis='y', labelsize=TICKLABELSIZE)
    ax.xaxis.label.set_color('gray')
    ax.yaxis.label.set_color('gray')
    ax.tick_params(axis='x', colors='gray', length=0)
    ax.tick_params(axis='y', colors='gray', length=0)
    ax.spines['top'].set_visible(False)
    ax.spines['top'].set_color('gray')
    ax.spines['right'].set_visible(False)
    ax.spines['right'].set_color('gray')
    ax.yaxis.label.set_size(LABELSIZE)
    ax.xaxis.label.set_size(LABELSIZE)

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


def plot_multiple_bars(data_list, x_list, colors, hatches,
              y_label=None, x_label_list=None, labelcolor='gray',
                       labelsize=LABELSIZE, ticklabelsize=TICKLABELSIZE,
                       legendsize=LEGEND_SIZE, textsize=LABELSIZE,
              figname="output", savefig=False):

    width = 0.1 * len(data_list[0])  # the width of the bars
    multiplier = 0
    fig, axes = plt.subplots(1, len(data_list),
                             figsize=(2 * len(data_list), 2),
                             squeeze=False,
                             # subplot_kw={'aspect': 1},
                             sharey=True)
    fig.subplots_adjust(hspace=0., wspace=0.)
    for j in range(len(data_list)):
        data = data_list[j]
        ax = axes[0][j]
        multiplier = 0
        x_label = x_label_list[j]
        for k in data.keys():
            v = data[k]
            # print(data[tau_ind]["pmd"])
            offset = width * multiplier
            # ind = list(np.a)
            # print(v[ind])
            # print(x + offset)
            mean = [v[i][0] for i in range(len(v))]
            std = [v[i][1] for i in range(len(v))]
            x = np.arange(len(mean))  # the label locations

            ax.bar(x + offset, mean,
                   width, yerr=(np.zeros_like(std), std), label=k if j == len(data_list) - 1 else None,
                   color=colors[k].colors[int(len(colors[k].colors)/2)],
                   edgecolor=colors[k].colors[-1], alpha=0.8,
                   hatch=hatches[k])
            # print(data[tau_ind]["mom"])
            # ax.bar_label(rects, padding=3)
            # ax.bar(x + offset, data[tau_ind]["mom"], width, label=("mom" if tau_ind == 0 else None), color=newcmap3.colors[-1])
            # ax.bar_label(rects, padding=3)
            multiplier += 1


        # Add some text for labels, title and custom x-axis tick labels, etc.
        if y_label is not None and j == 0:
            ax.set_ylabel(y_label, labelpad=0, fontsize=labelsize)

        ax.set_xticks(x + width*17/16, x_list)
        if x_label is not None:
            ax.set_xlabel(x_label, labelpad=0, fontsize=labelsize)
        # ax.legend(loc='upper left', ncols=3)
        # ymin, ymax = ax.get_ylim()
        # ax.set_ylim(0, ymax + (ymax- ymin) * 1/3)
        if j == len(data_list) - 1:
            ax.legend(
                prop={"size": legendsize},
                labelcolor=labelcolor,
                markerscale=0.01,
                # markersize=1,
                # bbox_to_anchor=legend_bbox,
                loc='best',
                labelspacing=0.5,
                borderaxespad=0.1,
                ncol=3,
                framealpha=0.5,
                columnspacing=0.1,
                borderpad=0.1, handletextpad=0)
        # ax.xaxis.set_tick_params(pad=-5)
        # ax.yaxis.set_tick_params(pad=-5)
        ax.tick_params(axis='x', labelsize=ticklabelsize)
        ax.tick_params(axis='y', labelsize=ticklabelsize)
        ax.xaxis.label.set_color(labelcolor)
        ax.yaxis.label.set_color(labelcolor)
        ax.tick_params(axis='x', colors=labelcolor, length=0)
        ax.tick_params(axis='y', colors=labelcolor, length=0)
        ax.spines['top'].set_visible(False)
        ax.spines['top'].set_color(labelcolor)
        ax.spines['right'].set_visible(False)
        ax.spines['right'].set_color(labelcolor)
        ax.yaxis.label.set_size(labelsize)
        ax.xaxis.label.set_size(labelsize)

    if savefig:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
        os.makedirs(save_path, exist_ok=True)
        pathname = os.path.join(save_path, f"{figname}.png")
        plt.savefig(pathname, bbox_inches='tight', dpi=300)

    plt.show()

def plot_learning_curves(data_list, labels_list, axline_data_list,x_list, legend_bbox, y_list, n_seeds,cmaps,Ts,
                         labelsize=LABELSIZE, ticklabelsize=TICKLABELSIZE,
                         legendsize=LEGEND_SIZE, textsize=LABELSIZE,
                         figname="output", savefig=False):
    fig, axes = plt.subplots(len(y_list), len(x_list),
                             figsize=[3*len(x_list),len(y_list)*3],
                             sharey="col", squeeze=False)
    for tau_ind, tau in enumerate(y_list):
        ymins = []
        ymaxs = []
        for kk, k in enumerate(x_list):
            ax = axes[tau_ind][kk]
            data = data_list[tau_ind][kk]
            labels = labels_list[tau_ind][kk]
            T = Ts[tau_ind][kk]
            axline_data = axline_data_list[tau_ind][kk]
            for alg, label in enumerate(labels):
                vals = []
                for i in range(n_seeds):
                    vals.append(data[alg][i]["v_t__rho"][:T])
                v_rho_std = np.std(vals, axis=0)
                v_rho_mean = np.mean(vals, axis=0)

                ts = data[alg][0]["t"][:T]
                tau_label = r'\tau'

                ax.plot(ts, v_rho_mean, color=cmaps[alg].colors[-1], lw=2, ls='-', label=label, alpha=0.8, zorder=1000)
                ax.fill_between(ts, v_rho_mean - v_rho_std,
                                v_rho_mean + v_rho_std, color=cmaps[alg].colors[-1], alpha=0.2)

            # ax.plot(ts, v_rho_ext_mean, color='g', lw=2, ls='--',label=("ext"  if nn == 2 and kk == 2 else None), alpha=0.8, zorder=1000)
            # ax.fill_between(ts, v_rho_ext_mean - v_rho_ext_std,
            #                 v_rho_ext_mean + v_rho_ext_std, color='g', alpha=0.2)
            ax.axhline(y=axline_data, xmin=0, xmax=1, linestyle='--', color='k', alpha=0.5)


            if tau_ind == len(y_list) - 1 and\
                kk == int(len(x_list) / 2) :
                ax.legend(
                    prop={"size": legendsize},
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
                ax.set_ylabel(r'$V^\rho_{\pi_t}$ (performance)', fontsize=labelsize)
            if tau_ind == len(y_list) - 1:
                ax.set_xlabel(r'$t$ (#iterations)', fontsize=labelsize)

            ax.tick_params(axis='x', labelsize=labelsize)
            ax.tick_params(axis='y', labelsize=labelsize)
            ymin, ymax = ax.get_ylim()
            ymins.append(ymin)
            ymaxs.append(ymax)

        ymin, ymax = min(ymins), max(ymaxs)
        for ax in axes[tau_ind]:
            # ax.set_ylim(-1, 1.1)
            ax.set_ylim(ymin, ymax)


        for kk, k in enumerate(x_list):
            ax = axes[tau_ind][kk]
            ymin, ymax = ax.get_ylim()
            xmin, xmax = ax.get_xlim()
            ax.text(xmin+(xmax-xmin)*3/8, ymin+(ymax-ymin)/8,
                    f"$k:{tau},{tau_label}:{k}$",
                    fontsize=textsize,
                    bbox=dict(facecolor='gray',
                              alpha=0.1,
                              edgecolor='gray',
                              boxstyle='round,pad=0.1'))
            # ax.set_yscale("log")

    if savefig:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
        os.makedirs(save_path, exist_ok=True)
        pathname = os.path.join(save_path, f"{figname}.png")
        plt.savefig(pathname, bbox_inches='tight', dpi=300)

    plt.show()
# def plot_learning_curves2(data_list, labels_list, axline_data_list,x_list, legend_bbox, y_list, n_seeds, cmaps,Ts,
#                          labelsize=LABELSIZE, ticklabelsize=TICKLABELSIZE,
#                          legendsize=LEGEND_SIZE, textsize=LABELSIZE,
#                          figname="output", savefig=False):
#     fig, axes = plt.subplots(len(y_list), len(x_list),
#                              figsize=[3*len(x_list),len(y_list)*3],
#                              sharey="col", squeeze=False)
#     for tau_ind, tau in enumerate(y_list):
#         ymins = []
#         ymaxs = []
#         for kk, k in enumerate(x_list):
#             ax = axes[tau_ind][kk]
#             data = data_list[tau_ind][kk]
#             labels = labels_list[tau_ind][kk]
#             T = Ts[tau_ind][kk]
#             axline_data = axline_data_list[tau_ind][kk]
#             for alg, label in enumerate(labels):
#                 vals = []
#                 for i in range(n_seeds):
#                     vals.append(data[alg][i][0][0]["v_t__rho"][:T])
#                 v_rho_std = np.std(vals, axis=0)
#                 v_rho_mean = np.mean(vals, axis=0)
#
#                 ts = data[alg][0][0][0]["t"][:T]
#                 tau_label = r'\tau'
#
#                 ax.plot(ts, v_rho_mean, color=cmaps[alg].colors[-1], lw=2, ls='-', label=label, alpha=0.8, zorder=1000)
#                 ax.fill_between(ts, v_rho_mean - v_rho_std,
#                                 v_rho_mean + v_rho_std, color=cmaps[alg].colors[-1], alpha=0.2)
#
#             # ax.plot(ts, v_rho_ext_mean, color='g', lw=2, ls='--',label=("ext"  if nn == 2 and kk == 2 else None), alpha=0.8, zorder=1000)
#             # ax.fill_between(ts, v_rho_ext_mean - v_rho_ext_std,
#             #                 v_rho_ext_mean + v_rho_ext_std, color='g', alpha=0.2)
#             ax.axhline(y=axline_data, xmin=0, xmax=1, linestyle='--', color='k', alpha=0.5)
#
#
#             if tau_ind == len(y_list) - 1 and \
#                     kk == int(len(x_list) / 2) :
#                 ax.legend(
#                     prop={"size": legendsize},
#                     labelcolor="gray",
#                     markerscale=0.1,
#                     bbox_to_anchor=legend_bbox,
#                     loc='upper right',
#                     labelspacing=0.1,
#                     borderaxespad=-0.1,
#                     ncol=3,
#                     columnspacing=2.,
#                     borderpad=-0.1,
#                     handletextpad=0.1)
#             if kk == 0:
#                 ax.set_ylabel(r'$V^\rho_{\pi_t}$ (performance)', fontsize=labelsize)
#             if tau_ind == len(y_list) - 1:
#                 ax.set_xlabel(r'$t$ (#iterations)', fontsize=labelsize)
#
#             ax.tick_params(axis='x', labelsize=labelsize)
#             ax.tick_params(axis='y', labelsize=labelsize)
#             ymin, ymax = ax.get_ylim()
#             ymins.append(ymin)
#             ymaxs.append(ymax)
#
#         ymin, ymax = min(ymins), max(ymaxs)
#         for ax in axes[tau_ind]:
#             # ax.set_ylim(-1, 1.1)
#             ax.set_ylim(ymin, ymax)
#
#
#         for kk, k in enumerate(x_list):
#             ax = axes[tau_ind][kk]
#             ymin, ymax = ax.get_ylim()
#             xmin, xmax = ax.get_xlim()
#             ax.text(xmin+(xmax-xmin)*3/8, ymin+(ymax-ymin)/8,
#                     f"$k:{tau},{tau_label}:{k}$",
#                     fontsize=textsize,
#                     bbox=dict(facecolor='gray',
#                               alpha=0.1,
#                               edgecolor='gray',
#                               boxstyle='round,pad=0.1'))
#             # ax.set_yscale("log")
#
#     if savefig:
#         save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
#         os.makedirs(save_path, exist_ok=True)
#         pathname = os.path.join(save_path, f"{figname}.png")
#         plt.savefig(pathname, bbox_inches='tight', dpi=300)
#
#     plt.show()

def plot_learning_curves2(data_list, labels_list, axline_data_list,x_list, legend_bbox, y_list, n_seeds,cmaps,Ts,
                         labelsize=LABELSIZE, annot_label=None, ticklabelsize=TICKLABELSIZE,
                         legendsize=LEGEND_SIZE, textsize=LABELSIZE,
                          ylim_list=None, figname="output", savefig=False):
    # print(len(x_list[0]))
    fig, axes = plt.subplots(len(y_list), len(x_list),
                             figsize=[3*len(x_list),len(y_list)*3],
                             sharey=True, squeeze=False)
    fig.subplots_adjust(hspace=0., wspace=0.)

    for kk, k in enumerate(y_list):
        ymins = []
        ymaxs = []

        for mdps_no, mdps in enumerate(x_list):
            ax = axes[kk][mdps_no]
            data = data_list[kk][mdps_no]
            labels = labels_list[kk][mdps_no]
            T = Ts[kk][mdps_no]
            # axline_data = axline_data_list[kk][tau_ind]
            for alg, label in enumerate(labels):
                vals = []
                for i in range(n_seeds):
                    vals.append(data[alg][i][0][0]["suboptimality_t__rho"][:T])
                v_rho_std = np.std(vals, axis=0)
                v_rho_mean = np.mean(vals, axis=0)

                ts = data[alg][0][0][0]["t"][:T]

                ax.plot(ts, v_rho_mean, color=cmaps[alg].colors[-1], lw=2, ls='-', label=label, alpha=0.8, zorder=1000)
                ax.fill_between(ts, v_rho_mean - v_rho_std,
                                v_rho_mean + v_rho_std, color=cmaps[alg].colors[-1], alpha=0.2)

            # ax.plot(ts, v_rho_ext_mean, color='g', lw=2, ls='--',label=("ext"  if nn == 2 and kk == 2 else None), alpha=0.8, zorder=1000)
            # ax.fill_between(ts, v_rho_ext_mean - v_rho_ext_std,
            #                 v_rho_ext_mean + v_rho_ext_std, color='g', alpha=0.2)
            #     vals = []
                # for i in range(n_seeds):
                #     vals.append(axline_data)
                # v_rho_std = np.std(vals, axis=0)
                # v_rho_mean = np.mean(vals, axis=0)
                # ax.axhline(y=v_rho_mean, xmin=0, xmax=1, linestyle='--', color='gray', alpha=0.5)
                # ax.fill_between(ts, v_rho_mean - v_rho_std,
                #                 v_rho_mean + v_rho_std, color='gray', alpha=0.2)


            if mdps_no == len(x_list) - 1 and \
                    kk == int(len(y_list) / 2) :
                ax.legend(
                    prop={"size": legendsize},
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
            if mdps_no == 0:
                ax.set_ylabel(r'$V^\rho_{\pi_t}$ (performance)', fontsize=labelsize)
            if kk == len(y_list) - 1:
                ax.set_xlabel(r'$t$ (#iterations)', fontsize=labelsize)

            ax.tick_params(axis='x', labelsize=labelsize)
            ax.tick_params(axis='y', labelsize=labelsize)
            ymin, ymax = ax.get_ylim()
            ymins.append(ymin)
            ymaxs.append(ymax)

        ymin, ymax = min(ymins), max(ymaxs)
        if ylim_list is not None:
            ymin, ymax = ylim_list[kk]
        for ax in axes[kk]:
            ax.set_ylim(ymin, ymax)

        for xx_ind, xx in enumerate(x_list):
            ax = axes[kk][xx_ind]
            ymin, ymax = ax.get_ylim()
            xmin, xmax = ax.get_xlim()
            ax.text(xmin+(xmax-xmin)*4/8, ymin+(ymax-ymin)*5/8,
                    f"${annot_label}:{xx}$",
                    fontsize=textsize,
                    bbox=dict(facecolor='gray',
                              alpha=0.1,
                              edgecolor='gray',
                              boxstyle='round,pad=0.1'))
            # ax.set_yscale("log")

    if savefig:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
        os.makedirs(save_path, exist_ok=True)
        pathname = os.path.join(save_path, f"{figname}.png")
        plt.savefig(pathname, bbox_inches='tight', dpi=300)

    plt.show()


