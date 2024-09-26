# -*- coding: utf-8 -*-

import numpy as np
import copy

colors = {
    'forward_euler': [1, 0, 0],
    'backward_euler': [0, 0.4, 0.7],
    'runge_kutta4': [0.2, 0.5, 0.6],
    'trapezoidal_rule': [0.5, 0.9, 0.2],
    'backward_differential_o2': [0.2, 0.8, 0.2],
    'backward_differential_o3': [0.5, 0.5, 0.5],
    'backward_differential_o4': [0.5, 0.5, 0.5],
    'adams_moulton_o2': 0.0 + 1.1 * np.array([0.7, 0.9, 0.45]),
    'adams_moulton_o3': 0.0 + 1.0 * np.array([0.7, 0.9, 0.45]),
    'adams_moulton_o4': 0.3 + 0.7 * np.array([0.7, 0.9, 0.45]),
    'adams_moulton_o5': 0.6 + 0.4 * np.array([0.7, 0.9, 0.45]),
    'adams_bashforth_o2': np.array([37/255, 48/255, 49/255]),
    'adams_bashforth_o3': 0.4 + 0.6 * np.array([37/255, 48/255, 49/255]),
    'adams_bashforth_o4': 0.6 + 0.4 * np.array([37/255, 48/255, 49/255]),

    '1e-05': [0.4, 0.5, 0.6],
    '1e-04': [0.5, 0.5, 0.5],
    '1e-03': [0.7, 0.5, 0.3],
    '2e-03': [0.7, 0.5, 0.9],
    '5e-03': [0.2, 0.2, 0.9],
    '1e-01': [0.2, 0.4, 0.9],

    }
# Adams-Bashforth methods (cluster 1: shades of blue)
colors['adams_bashforth_o2'] = [0.0, 0.45, 0.7]   # Blue
colors['adams_bashforth_o3'] = [0.1, 0.5, 0.8]  # Lighter Blue
colors['adams_bashforth_o4'] = [0.2, 0.55, 0.85]  # Lighter Blue

# Adams-Moulton methods (cluster 2: shades of green)
colors['adams_moulton_o2'] = [0.0, 0.6, 0.5]    # Green
colors['adams_moulton_o3'] = [0.0, 0.6, 0.5]    # Green
colors['adams_moulton_o4'] = [0.2, 0.7, 0.6]    # Lighter Green

# Backward Differential methods (cluster 3: shades of orange)
colors['backward_differential_o2'] = [0.9, 0.5, 0.2]  # Orange
colors['backward_differential_o3'] = [0.95, 0.55, 0.3]  # Lighter Orange
colors['backward_differential_o4'] = [1.0, 0.6, 0.4]  # Lighter Orange

# Euler methods (cluster 4: shades of red)
colors['forward_euler'] = [0.8, 0.1, 0.1]  # Red
colors['backward_euler'] = [0.9, 0.3, 0.3] # Lighter Red

# Other methods (cluster 5: shades of purple and yellow)
colors['trapezoidal_rule'] = [0.5, 0.2, 0.6]  # Purple
colors['runge_kutta4'] = [0.95, 0.9, 0.25]    # Yellow

fontsettings = {
    'label_fontsize': 11,
    'title_fontsize': 15,
    'title_fontweight':'bold',
    'legend_fontsize': 10,
    }


def get_list_of_protocols(**kwargs):
    list_of_protocols = []


    solvers = kwargs.get('solvers', [
        "adams_bashforth_o2",
        "adams_bashforth_o3",
        # "adams_bashforth_o4",
        "adams_moulton_o2",
        "adams_moulton_o3",
        "adams_moulton_o4",
        "backward_differential_o2",
        "backward_differential_o3",
        "backward_differential_o4",
        "forward_euler",
        "backward_euler",
        ])
    dts = kwargs.get('dt', [
                0.000001,
                0.00001,
                0.0001,
                0.0005,
                0.001,
                0.002,
                0.005,
           ])

    for solver in solvers:
        for dt in dts:
            protocol = {
                'name': f'{solver} {dt*1e3}ms',
                'legend': f'{solver} {dt*1e3}ms',
                'solver': f'{solver}',
                'set': [
                    ('Solver.dt', dt),
                    ('Solver.dt_export', np.max([dt, 0.001])),
                    ],
                }

            list_of_protocols.append(protocol)


    if 'TriSeg_thresh_F' in kwargs:
        list_of_protocols1 = []
        for protocol in list_of_protocols:
            for thresh in kwargs['TriSeg_thresh_F']:
                protocol1 = copy.deepcopy(protocol)
                protocol1['name'] = protocol1['name'] + ' t ' + str(thresh)
                protocol1['set'].append(('Model.Peri.TriSeg.thresh_F', thresh))
                list_of_protocols1.append(protocol1)
        list_of_protocols = list_of_protocols1

    if 'fac_pfc' in kwargs:
        list_of_protocols1 = []
        for protocol in list_of_protocols:
            for thresh in kwargs['fac_pfc']:
                protocol1 = copy.deepcopy(protocol)
                protocol1['name'] = protocol1['name'] + ' facpfc ' + str(thresh)
                protocol1['set'].append(('Model.PFC.fac', thresh))
                list_of_protocols1.append(protocol1)
        list_of_protocols = list_of_protocols1

    return list_of_protocols



def boxplot_group_by_solver(ax, data_in, n_groups, idx_prot_groups, prot_groups,
                            y_log=False, bar=False):

    for i_prot in range(n_groups):
        idx = idx_prot_groups == i_prot
        data = data_in[idx].reshape((np.sum(idx), -1))
        data_percentile = np.nanpercentile(data, [2.5, 25, 50, 75, 97.5],
                                           axis=1)
        n = len(range(len(data_percentile[2, :])))
        w = 1 / (n_groups + 2)
        bar_X = np.linspace(i_prot, i_prot+0.9, n) - 0.45


        for i in range(data.shape[0]):
            c = np.array([0.9, 0.9, 0.9]) * (i / data.shape[0]) ** 0.5

            if bar:
                ax.bar(bar_X[i], data[i, np.invert(np.isnan(data[i, :]))],
                       width=w, fc=c)
            else:
                ax.boxplot(
                    [data[i, np.invert(np.isnan(data[i, :]))]],
                    positions = [bar_X[i]],
                    notch=False, patch_artist=True,
                    showfliers=False,
                    boxprops=dict(facecolor=(*c, 0.3), color=c),
                    capprops=dict(color=c),
                    whiskerprops=dict(color=c),
                    # flierprops=dict(color=c, markeredgecolor=c),
                    medianprops=dict(color=c),
                    widths=w,
                    )
    labels = [l.replace('_', ' ') for l in prot_groups]
    ax.set_xticks(np.linspace(0, n_groups-1, n_groups),
                                labels, rotation=25, ha='right', va = 'top',
                                fontsize=9)
    if y_log:
        ax.set_yscale('log')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)




