# -*- coding: utf-8 -*-
from matplotlib.colors import LogNorm, Normalize
import seaborn as sns
from _functions import *

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import addcopyfighandler

from tqdm import tqdm

import circadapt
model = circadapt.VanOsta2024()

from _stability_functions import *

# set seed for reproducibility
np.random.seed(1)

# parameters
folder_name = 'data_single-beat'
n_sims = 1000

# create samples
X = np.random.random((n_sims, n_par))

# define protocols
list_of_protocols = []

# %% forward_euler

use_plot_comparison_01ms = False
use_plot_solverFE = False
use_plot_solverCA = False
use_plot_solverBDF1 = False
use_plot_solverVar = False


# list_of_protocols = get_list_of_protocols(TriSeg_thresh_F=[1e-1])
list_of_protocols = get_list_of_protocols(TriSeg_thresh_F=[1e-6])


# list_of_protocols = (
#     get_list_of_protocols(solvers=["backward_differential_o2",], dt=[0.00001], TriSeg_thresh_F=[1e-6])
#     +
#     get_list_of_protocols(solvers=["backward_differential_o2",], dt=[0.001], TriSeg_thresh_F=[1e-3])
#     )


# %% Get reference state
model = circadapt.VanOsta2024()
model.run(1)
model_state = model.model_export()

# %% Run Protocol
calculation_time = np.ndarray((len(list_of_protocols), n_sims)) + np.nan
simulation_succes = np.zeros((len(list_of_protocols), n_sims), dtype=bool)
simulation_output = np.ndarray(
    (len(list_of_protocols), n_sims, n_out)) + np.nan
simulation_signals = np.ndarray((len(list_of_protocols), n_sims), dtype=object)

for i_protocol, protocol in enumerate(list_of_protocols):
    print('Run Protocol ', i_protocol, '/', len(list_of_protocols))

    # load prerunned data
    run_protocol = False
    filename = folder_name+'/'+protocol['name']+'.npy'
    try:
        data = np.load(filename, allow_pickle=True).item()
        calculation_time[i_protocol, :] = data['calculation_time']
        simulation_succes[i_protocol, :] = data['simulation_succes']
        simulation_output[i_protocol, :] = data['simulation_output']
        simulation_signals[i_protocol, :] = data['simulation_signals']

        if calculation_time.shape[1] != n_sims:
            run_protocol = True
    except:
        run_protocol = True

    if run_protocol:
        # for i_sim in range(n_sims):
        for i_sim in tqdm(range(n_sims), desc="Simulation Progress"):
            # print(f'\r {i_sim} / {n_sims}', end='\r')

        # load model
            solver = protocol['solver']
            if solver[-3:-1] == '_o':
                model = circadapt.VanOsta2024(solver[:-3])
                model.model_import(model_state)
                model.set('Solver.order', int(solver[-1]))
            else:
                model = circadapt.VanOsta2024(solver)
                model.model_import(model_state)
                
            

            # set paramteers
            for s in protocol['set']:
                model.set(*s)

            # set paremeters
            setX(model, X[i_sim, :])

            # run
            try:

                t0 = time.time()
                model.run(1)
                t1 = time.time()
                dt = t1-t0

                calculation_time[i_protocol, i_sim] = dt*1e3
                simulation_succes[i_protocol, i_sim] = True
                output, signals = getY(model)
                simulation_output[i_protocol, i_sim, :] = output
                simulation_signals[i_protocol, i_sim] = signals
            except circadapt.error.ModelCrashed:
                print(i_sim, 'crashed')
            model = []

        data = {
            'calculation_time': calculation_time[i_protocol, :],
            'simulation_succes': simulation_succes[i_protocol, :],
            'simulation_output': simulation_output[i_protocol, :],
            'simulation_signals': simulation_signals[i_protocol, :],
        }
        np.save(filename, data)

print('finished running simulations')


# %% Calculate error in signals

# get simulations with lowest dt
lowest_dt = 1
idx_prot = []
all_solvers = []
all_dt = []
for i_protocol, protocol in enumerate(list_of_protocols):
    if lowest_dt == protocol['set'][0][1]:
        idx_prot.append(i_protocol)
    elif protocol['set'][0][1] < lowest_dt:
        idx_prot = [i_protocol]
        lowest_dt = protocol['set'][0][1]

    # solvers
    s = protocol['solver']
    all_solvers.append(s)
    all_dt.append(protocol['set'][0][1])

idx_prot = np.array(idx_prot)
all_dt = np.array(all_dt)
prot_groups, idx_prot_groups = np.unique(all_solvers, False, True)
n_groups = np.max(idx_prot_groups)+1
prot_has_min_dt = all_dt == np.min(all_dt)


# first signal is time
# calculate error
n_signals = simulation_signals[0][0].shape[0]
signals_mean_abs_error = np.empty(
    (simulation_signals.shape[0], *simulation_signals.shape, n_signals-1))
signals_mean_abs_error[:] = np.nan

# %%
mean_signal = [[] for i_sim in range(signals_mean_abs_error.shape[2])]
for i_sim in range(len(mean_signal)):
    print(f'\r Calculate mean signal {
          i_sim} / {signals_mean_abs_error.shape[2]}')
    for i_prot0 in range(signals_mean_abs_error.shape[0]):
        if simulation_signals[i_prot0, i_sim] is None:
            continue
        data0 = simulation_signals[i_prot0, i_sim][1:]
        time0 = simulation_signals[i_prot0, i_sim][0]
        for i_prot1 in range(signals_mean_abs_error.shape[1]):
            if simulation_signals[i_prot1, i_sim] is None:
                continue
            data1 = simulation_signals[i_prot1, i_sim][1:]
            time1 = simulation_signals[i_prot1, i_sim][0]

            # check wich data to include in comparison
            time, idx0, idx1 = np.intersect1d(
                time0, time1, return_indices=True)
            e = np.abs(data0[:, idx0] - data1[:, idx1])
            signals_mean_abs_error[i_prot0, i_prot1,
                                   i_sim, :] = np.mean(e, axis=1)

    # calculate mean signal
    # correction adam bashforth
    aux = simulation_signals[prot_has_min_dt, i_sim]
    idx_include = [a is not None for a in aux]
    len_aux = [a.shape[1] for a in aux[idx_include]]
    if len(len_aux) == 0 or np.ptp(len_aux) > 1:
        aux = simulation_signals[np.argwhere(
            prot_has_min_dt).reshape(-1)+1, i_sim]
        idx_include = [a is not None for a in aux]
        len_aux = [a.shape[1] for a in aux[idx_include]]
        if np.ptp(len_aux) > 1:
            raise Exception('')

    for i_aux, a in enumerate(aux):
        if idx_include[i_aux]:
            aux[i_aux] = a[:, :np.min(len_aux)]
    mean_signal[i_sim] = np.nanmean(np.stack(aux[idx_include]), axis=0)


# %% Calc abs error relative to 'true' value
n_signals = simulation_signals[0][0].shape[0]
signals_mean_abs_error_true = np.empty(
    (*simulation_signals.shape, n_signals-1))
signals_mean_abs_error_true[:] = np.nan
for i_sim in range(len(mean_signal)):
    print(f'\r Calculate absolute error {i_sim} / {n_sims}')
    time_true = mean_signal[i_sim][0]
    data_true = mean_signal[i_sim][1:]
    for i_prot0 in range(signals_mean_abs_error.shape[0]):
        if simulation_signals[i_prot0, i_sim] is None:
            continue
        data0 = simulation_signals[i_prot0, i_sim][1:, :-1]
        time0 = simulation_signals[i_prot0, i_sim][0, :-1]

        interp_data = np.empty_like(data0)

        for i_data in range(interp_data.shape[0]):
            interp_data[i_data] = np.interp(
                time0, time_true, data_true[i_data])

        signals_mean_abs_error_true[i_prot0, i_sim, :] = np.nanmean(
            np.abs(interp_data - data0), axis=1)


# %%
plt.figure(999, clear=True)
i_signal = 0
i_sim = 7
plt.plot(mean_signal[i_sim][0].T, mean_signal[i_sim][1:].T)
for i_prot0 in range(signals_mean_abs_error.shape[0]):
    if simulation_signals[i_prot0, i_sim] is None:
        continue
    
    if all_dt[i_prot0] > 1e-6:
        continue
    data0 = simulation_signals[i_prot0, i_sim][1:]
    time0 = simulation_signals[i_prot0, i_sim][0]
    plt.plot(time0, data0.T, ls='--')

# %%
fig = plt.figure(202, clear=True, figsize=(12, 4))
signals_mean_mean_abs_error = np.nanmean(signals_mean_abs_error, axis=2)


include_protocols = all_dt == np.min(all_dt)
include_protocols = np.argwhere(include_protocols)[:, 0]
protocol_names = [list_of_protocols[i_prot]['name'].replace(
    ' ', '\n') for i_prot in range(len(list_of_protocols))]


def custom_sci_format(x, pos):
    return f'{x:.0e}'.replace('e-0', 'e-').replace('e+0', 'e+').replace('0.0e+0', '')


# Example 6x6 matrix of errors (replace this with your actual matrix)
error_matrix = signals_mean_mean_abs_error[:, :, 0]
error_matrix = error_matrix[include_protocols][:, include_protocols]

# Methods to be displayed on the axes
# methods = [f'Method {i+1}' for i in range(6)]
methodsx = [protocol_names[i].replace('_o2', '').replace('_o3', '').split('\n')[
    0].replace('_', '\n') for i in include_protocols]
methodsy = [protocol_names[i].replace('_o2', '').replace('_o3', '').split('\n')[
    0].replace('_', ' ') for i in include_protocols]

methodsx = [protocol_names[i].split('\n')[0].replace(
    '_', '\n') for i in include_protocols]
methodsy = [protocol_names[i].split('\n')[0].replace(
    '_', ' ') for i in include_protocols]


# Create a mask for the upper triangular part of the matrix
mask = 1 - np.tril(np.ones(error_matrix.shape), k=0)

# Apply the mask to the matrix to keep only the upper triangular part
masked_matrix = np.ma.masked_where(mask == 0, error_matrix)

# Create the heatmap
ax = plt.subplot(1, 2, 1)
ax = sns.heatmap(
    masked_matrix,
    annot=True,
    fmt=".1e",
    annot_kws={"size": 7},
    cmap="viridis",
    xticklabels=methodsy,
    yticklabels=methodsy,
    mask=1 - np.tril(np.ones(error_matrix.shape), k=0),
    cbar_kws={'label': 'Error [mL]'}, norm=LogNorm(),

)

# Apply custom formatter to annotations
for text in ax.texts:
    text.set_text(custom_sci_format(float(text.get_text()), None))

plt.xticks(rotation=45)
plt.yticks(rotation=45)
# Rotate y-axis labels 45 degrees
for label in ax.get_xticklabels():
    label.set_rotation(45)
    label.set_ha('right')  # Horizontal alignment
    label.set_va('top')  # Horizontal alignment
for label in ax.get_yticklabels():
    label.set_rotation(45)
    label.set_ha('right')  # Horizontal alignment
    label.set_va('top')  # Horizontal alignment

# Set axis labels
plt.title('Mean absolute error in LV volume')


ax = plt.subplot(1, 2, 2)

error_matrix = signals_mean_mean_abs_error[:, :, 1]
error_matrix = error_matrix[include_protocols][:, include_protocols]
# Create a mask for the upper triangular part of the matrix
mask = np.tril(np.ones(error_matrix.shape), k=1)

# Apply the mask to the matrix to keep only the upper triangular part
masked_matrix = np.ma.masked_where(mask == 0, error_matrix)
ax = sns.heatmap(
    masked_matrix,
    annot=True,
    fmt=".0e",
    annot_kws={"size": 7},
    cmap="viridis",
    xticklabels=methodsy,
    yticklabels=methodsy,
    mask=1 - np.tril(np.ones(error_matrix.shape), k=0),
    cbar_kws={'label': 'Error [mmHg]'}, norm=LogNorm(),
)
plt.xticks(rotation=45)
plt.yticks(rotation=45)

# Rotate y-axis labels 45 degrees
for label in ax.get_xticklabels():
    label.set_rotation(45)
    label.set_ha('right')  # Horizontal alignment
    label.set_va('top')  # Horizontal alignment
for label in ax.get_yticklabels():
    label.set_rotation(45)
    label.set_ha('right')  # Horizontal alignment
    label.set_va('top')  # Horizontal alignment

# Apply custom formatter to annotations
for text in ax.texts:
    text.set_text(custom_sci_format(float(text.get_text()), None))

# Set axis labels
plt.title('Mean absolute error in LV pressure')

plt.suptitle(
    'Single beat analysis at $\\Delta t = 10^{-3} ms$ and $e_{TriSeg} = 10^{-6}$')
plt.tight_layout()

# %%
fig = plt.figure(203, clear=True, figsize=(12, 4))
signals_mean_mean_abs_error = np.nanmean(signals_mean_abs_error, axis=2)


include_protocols = all_dt >= np.min(all_dt)
include_protocols = np.argwhere(include_protocols)[:, 0]
protocol_names = [list_of_protocols[i_prot]['name'].replace(
    ' ', '\n') for i_prot in range(len(list_of_protocols))]


def custom_sci_format(x, pos):
    return f'{x:.0e}'.replace('e-0', 'e-').replace('e+0', 'e+').replace('0.0e+0', '')


# Example 6x6 matrix of errors (replace this with your actual matrix)
error_matrix = signals_mean_mean_abs_error[:, :, 0]
error_matrix = error_matrix[include_protocols][:, include_protocols]

# Methods to be displayed on the axes
# methods = [f'Method {i+1}' for i in range(6)]
methodsx = [protocol_names[i].replace('_o2', '').replace(
    '_o3', '').replace('_', '\n') for i in include_protocols]
methodsy = [protocol_names[i].replace('_o2', '').replace(
    '_o3', '').replace('_', ' ') for i in include_protocols]

methodsx = [protocol_names[i].replace('_', ' ').replace(
    '\n', ' ') for i in include_protocols]
methodsy = [protocol_names[i].replace('_', ' ').replace(
    '\n', ' ') for i in include_protocols]
# methodsy = [protocol_names[i].replace('_', '') for i in include_protocols]


methodsy = [s.replace('adams bashforth', 'AB') for s in methodsy]
methodsy = [s.replace('adams moulton', 'AM') for s in methodsy]
methodsy = [s.replace('backward differential', 'BDF') for s in methodsy]
methodsy = [s.replace('forward euler', 'FE') for s in methodsy]
methodsy = [s.replace('backward euler', 'BE') for s in methodsy]
methodsy = [s[:-7] for s in methodsy]


# Create a mask for the upper triangular part of the matrix
mask = 1 - np.tril(np.ones(error_matrix.shape), k=0)

# Apply the mask to the matrix to keep only the upper triangular part
masked_matrix = np.ma.masked_where(mask == 0, error_matrix)

# Create the heatmap
ax = plt.subplot(1, 1, 1)
ax = sns.heatmap(
    masked_matrix,
    annot=True,
    fmt=".1e",
    annot_kws={"size": 6},
    cmap="viridis",
    xticklabels=methodsy,
    yticklabels=methodsy,
    mask=1 - np.tril(np.ones(error_matrix.shape), k=0),
    cbar_kws={'label': 'Error [mL]'}, norm=LogNorm(),

)

# Apply custom formatter to annotations
for text in ax.texts:
    text.set_text(custom_sci_format(float(text.get_text()), None))

# Rotate y-axis labels 45 degrees
for label in ax.get_xticklabels():
    label.set_rotation(45)
    label.set_ha('right')  # Horizontal alignment
    label.set_va('top')  # Horizontal alignment
for label in ax.get_yticklabels():
    label.set_rotation(30)
    label.set_ha('right')  # Horizontal alignment
    label.set_va('top')  # Horizontal alignment

# Set axis labels
plt.title('Mean absolute error in LV volume')

fig = plt.figure(204, clear=True, figsize=(12, 4))
ax = plt.subplot(1, 1, 1)

error_matrix = signals_mean_mean_abs_error[:, :, 1]
error_matrix = error_matrix[include_protocols][:, include_protocols]
# Create a mask for the upper triangular part of the matrix
mask = np.tril(np.ones(error_matrix.shape), k=1)

# Apply the mask to the matrix to keep only the upper triangular part
masked_matrix = np.ma.masked_where(mask == 0, error_matrix)
ax = sns.heatmap(
    masked_matrix,
    annot=True,
    fmt=".0e",
    annot_kws={"size": 6},
    cmap="viridis",
    xticklabels=methodsy,
    yticklabels=methodsy,
    mask=1 - np.tril(np.ones(error_matrix.shape), k=0),
    cbar_kws={'label': 'Error [mmHg]'}, norm=LogNorm(),
)
plt.xticks(rotation=45)
plt.yticks(rotation=45)

# Rotate y-axis labels 45 degrees
for label in ax.get_xticklabels():
    label.set_rotation(45)
    label.set_ha('right')  # Horizontal alignment
    label.set_va('top')  # Horizontal alignment
for label in ax.get_yticklabels():
    label.set_rotation(30)
    label.set_ha('right')  # Horizontal alignment
    label.set_va('top')  # Horizontal alignment

# Apply custom formatter to annotations
for text in ax.texts:
    text.set_text(custom_sci_format(float(text.get_text()), None))

# Set axis labels
plt.title('Mean absolute error in LV pressure')

plt.suptitle(
    'Single beat analysis at $\\Delta t = 10^{-3}ms$ and $e_{TriSeg} = 10^{-6}$')
plt.tight_layout()

# %%
plt.figure(1, figsize=(12, 8))
plt.clf()
m = 2
n = 1

calculation_time_filtered = [1e-3*calculation_time[i_prot, ~np.isnan(
    calculation_time[i_prot, :])] for i_prot in range(calculation_time.shape[0])]
protocol_names = [list_of_protocols[i_prot]['name'].replace(
    ' ', '\n') for i_prot in range(len(list_of_protocols))]

plt.subplot(m, n, 1)
plt.boxplot(calculation_time_filtered, whis=[5, 95], showfliers=False)
plt.xticks(range(1, len(protocol_names)+1),
           ['' for _ in range(len(protocol_names))], rotation=45)
plt.xlim(0, len(protocol_names)+1)
plt.yscale('log')
# plt.ylim([1e-0, 1e3])
plt.ylabel('Computation time [s]', fontsize=14, fontweight='bold')

plt.subplot(m, n, 2)
plt.bar(range(len(list_of_protocols)), np.sum(
    simulation_succes == False, axis=1)/simulation_succes.shape[1]*100)
plt.ylabel('Crashed simulations [%]', fontsize=14, fontweight='bold')
plt.xticks(range(len(protocol_names)), [n.replace('\n', ' ') for n in protocol_names], rotation=45, fontsize=9, fontweight='bold',
           ha='right', )
plt.xlim(-1, len(protocol_names))

plt.tight_layout()

# %%


def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 10))  # outward by 10 points
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])


if False:
    i_fig = 0
    plt.figure(2 + i_fig)
    plt.clf()

    include_protocols = np.sum(
        simulation_succes == False, axis=1) < simulation_succes.shape[1]
    include_protocols = np.argwhere(include_protocols)[:, 0]

    m = len(include_protocols)
    n = len(include_protocols)

    axes = []
    max_diff = 0
    min_diff = np.Inf
    for i_protocol1a in range(len(include_protocols)):
        for i_protocol0a in range(i_protocol1a+1, len(include_protocols)):

            i_protocol1 = include_protocols[i_protocol1a]
            i_protocol0 = include_protocols[i_protocol0a]

            i_dat = 1

            # Left Bottom
            for i in range(2):
                if i == 0:
                    ax = plt.subplot(
                        m, n, 1 + i_protocol1a + (i_protocol0a) * n)
                    scatX = simulation_output[i_protocol0, :, i_dat]
                    scatY = simulation_output[i_protocol1, :, i_dat]
                else:
                    # Right Top
                    ax = plt.subplot(
                        m, n, 1 + i_protocol0a + (i_protocol1a) * n)
                    scatX = simulation_output[i_protocol0, :, i_dat+1]
                    scatY = simulation_output[i_protocol1, :, i_dat+1]

                axes.append(ax)
                adjust_spines(ax, ['left', 'bottom'])

                xl = np.array([np.nanmin(0.5*(scatX+scatY)),
                              np.nanmax(0.5*(scatX+scatY))])

                ax.set_xlim(xl)
                ax.scatter(0.5*(scatX+scatY), (scatX-scatY), s=1)
                ax.plot([0, xl[1]], [0, 0], c='k', linewidth=1)

                # plt.yscale('log')
                md = np.nanmax(np.abs(scatX-scatY))
                if md > max_diff:
                    max_diff = md
                md = np.nanmin(np.abs(scatX-scatY))
                if md < min_diff:
                    min_diff = md

                if i == 0:
                    if i_protocol0a == m-1:
                        plt.xlabel(protocol_names[i_protocol1])

                    if i_protocol1a == 0:
                        plt.ylabel(protocol_names[i_protocol0])

    for ax in axes:
        ax.set_ylim([-max_diff*1.2, max_diff*1.2])

    plt.tight_layout()


# %%
i_fig = 0
plt.figure(3 + i_fig)
plt.clf()

include_protocols = np.sum(simulation_succes == False,
                           axis=1) < simulation_succes.shape[1]
include_protocols = np.argwhere(include_protocols)[:, 0]

m = len(include_protocols)
n = len(include_protocols)

axes = []
max_diff = 0
min_diff = np.inf

corr = np.zeros((len(include_protocols), len(include_protocols)))
corr[:] = np.nan

for i_protocol1a in range(len(include_protocols)):
    for i_protocol0a in range(i_protocol1a+1, len(include_protocols)):

        i_protocol1 = include_protocols[i_protocol1a]
        i_protocol0 = include_protocols[i_protocol0a]

        # Left Bottom
        for i in range(2):
            i_dat = [8, 9][i]
            if i == 0:
                scatX = simulation_output[i_protocol0, :, i_dat]
                scatY = simulation_output[i_protocol1, :, i_dat]

                idx = (np.isnan(scatX) == False) & (np.isnan(scatY) == False)

                scatX = scatX[idx]
                scatY = scatY[idx]

                corr[i_protocol1a, i_protocol0a] = np.corrcoef(scatX, scatY)[
                    0, 1]
            else:
                # Right Top
                scatX = simulation_output[i_protocol0, :, i_dat]
                scatY = simulation_output[i_protocol1, :, i_dat]

                idx = (np.isnan(scatX) == False) & (np.isnan(scatY) == False)

                scatX = scatX[idx]
                scatY = scatY[idx]

                corr[i_protocol0a, i_protocol1a] = np.corrcoef(scatX, scatY)[
                    1, 0]


# %%


# True values: mean of lowest dt's
semi_true_values = np.mean(simulation_output[idx_prot, :], axis=0)

plt.figure(10, figsize=(12, 8), clear=True)
m = 5
n = 6

for i_data in range(semi_true_values.shape[1]):
    ax = plt.subplot(m, n, i_data+1)
    for i_prot in range(n_groups):
        data = np.abs(
            semi_true_values[:, i_data] - simulation_output[idx_prot_groups == i_prot, :, i_data])
        data /= semi_true_values[:, i_data]
        data_percentile = np.nanpercentile(data, [5, 25, 50, 75, 95], axis=1)
        # plt.fill_between(all_dt[idx_prot_groups==i_prot],
        #                  data_percentile[0,:],
        #                  data_percentile[-1,:],
        #                  alpha=0.5)
        # plt.fill_between(all_dt[idx_prot_groups==i_prot],
        #                  data_percentile[1,:],
        #                  data_percentile[3,:],
        #                  alpha=0.5)
        plt.plot(all_dt[idx_prot_groups == i_prot],
                 data_percentile[2, :], label=prot_groups[i_prot])
    plt.xscale('log')
    # plt.yscale('log')
    plt.title(output_names[i_data])
    plt.legend()
# plt.tight_layout()

# %% Nice plot


# %% FINAL PLOT
fig = plt.figure(99, clear=True, figsize=(12, 8))
gs = fig.add_gridspec(11, 19)

subplots_adjust = {'top': 0.93,
                   'bottom': 0.1,
                   'left': 0.066,
                   'right': 0.988,
                   'hspace': 0.3,
                   'wspace': 0.4,
                   }
fig.subplots_adjust(**subplots_adjust)

axLeft = fig.add_subplot(gs[5:, :7])

# Calculation time plot
for i_prot in range(n_groups):
    idx = idx_prot_groups == i_prot

    plot_ranges = np.nanpercentile(
        calculation_time[idx, :]*1e-3, [5, 25, 50, 75, 95], axis=1)

    include_sims = np.sum(
        np.isnan(calculation_time[idx, :]), axis=1) < (n_sims / 2)

    axLeft.plot(all_dt[idx][include_sims], plot_ranges[2, include_sims],
                color=colors[prot_groups[i_prot]], zorder=9)

    axLeft.fill_between(all_dt[idx][include_sims], plot_ranges[0, include_sims], plot_ranges[4, include_sims],
                        color=colors[prot_groups[i_prot]], zorder=7, alpha=0.1)
    axLeft.fill_between(all_dt[idx][include_sims], plot_ranges[1, include_sims], plot_ranges[3, include_sims],
                        color=colors[prot_groups[i_prot]], zorder=8, alpha=0.1)

axLeft.set_yscale('log')
axLeft.set_xscale('log')

axLeft.set_xlabel('Integration step size $\\Delta t$ [s]',
                  fontsize=fontsettings['label_fontsize'])
axLeft.set_ylabel('Calculation time [s]',
                  fontsize=fontsettings['label_fontsize'])
axLeft.spines['top'].set_visible(False)
axLeft.spines['right'].set_visible(False)

left_location = subplots_adjust['left'] + 0.2*(
    subplots_adjust['right']-subplots_adjust['left'])
axLeft.annotate('Single beat calculation time', (left_location, 0.62),
                xycoords='figure fraction',
                horizontalalignment='center', verticalalignment='center',
                fontsize=fontsettings['title_fontsize'],
                fontweight=fontsettings['title_fontweight'])

# Verification plots Energy
axEnergy = fig.add_subplot(gs[:3, :3])
for i_prot in range(n_groups):
    idx = idx_prot_groups == i_prot
    error_in_energy = 1 / 2 * (simulation_output[idx, :, 10] - simulation_output[idx, :, 11]) / (
        simulation_output[idx, :, 10] + simulation_output[idx, :, 11])
    data_percentile = np.nanpercentile(error_in_energy, [5, 25, 50, 75, 95],
                                       axis=1)

    axEnergy.plot(all_dt[idx], data_percentile[2, :],
                  color=colors[prot_groups[i_prot]])
    axEnergy.fill_between(all_dt[idx], data_percentile[0, :], data_percentile[4, :],
                          color=colors[prot_groups[i_prot]],
                          zorder=7, alpha=0.1)
    axEnergy.fill_between(all_dt[idx], data_percentile[1, :], data_percentile[3, :],
                          color=colors[prot_groups[i_prot]],
                          zorder=8, alpha=0.1)
axEnergy.set_xscale('log')
axEnergy.set_yscale('log')
axEnergy.set_ylim([1e-11, 1e0])
axEnergy.annotate('Verification Patch and TriSeg', (0.05, 0.975),
                  xycoords='figure fraction',
                  horizontalalignment='left', verticalalignment='center',
                  fontsize=fontsettings['title_fontsize'],
                  fontweight=fontsettings['title_fontweight'])
axEnergy.set_title('Error in local and global energy',
                   fontsize=fontsettings['label_fontsize'])
axEnergy.set_ylabel('error [%]',
                    fontsize=fontsettings['label_fontsize'])
axEnergy.set_xlabel('Integration step size $\\Delta t$ [s]',
                    fontsize=fontsettings['label_fontsize'])

# Verification plots TriSeg
axTriSegX = fig.add_subplot(gs[:3, 4:7])
for i_prot in range(n_groups):
    idx = idx_prot_groups == i_prot
    # error_in_energy = 1 / 2 * (simulation_output[idx, :, 10] - simulation_output[idx, :, 11]) / (simulation_output[idx, :, 10] + simulation_output[idx, :, 11])
    # error_in_energy = simulation_output[idx, :, 12] / simulation_output[idx, :, 14]
    error_in_energy = np.sqrt(simulation_output[idx, :, 12]**2 + simulation_output[idx, :, 13]**2) / np.sqrt(
        simulation_output[idx, :, 14]**2 + simulation_output[idx, :, 15]**2)
    data_percentile = np.nanpercentile(error_in_energy, [5, 25, 50, 75, 95],
                                       axis=1)

    axTriSegX.plot(all_dt[idx], data_percentile[2, :],
                   color=colors[prot_groups[i_prot]])
    axTriSegX.fill_between(all_dt[idx], data_percentile[0, :], data_percentile[4, :],
                           color=colors[prot_groups[i_prot]],
                           zorder=7, alpha=0.1)
    axTriSegX.fill_between(all_dt[idx], data_percentile[1, :], data_percentile[3, :],
                           color=colors[prot_groups[i_prot]],
                           zorder=8, alpha=0.1)
axTriSegX.set_xscale('log')
axTriSegX.set_yscale('log')
axTriSegX.set_ylim([1e-11, 1e0])
axTriSegX.set_title('Tension imbalance',
                    fontsize=fontsettings['label_fontsize'])
axTriSegX.set_ylabel('error [%]',
                     fontsize=fontsettings['label_fontsize'])
axTriSegX.set_xlabel('Integration step size $\\Delta t$ [s]',
                     fontsize=fontsettings['label_fontsize'])


# error plots
axErrors = [fig.add_subplot(gs[0:3, 14:19]),
            fig.add_subplot(gs[4:7, 8:13]),
            fig.add_subplot(gs[4:7, 14:19]),
            fig.add_subplot(gs[8:, 8:13]),
            fig.add_subplot(gs[8:, 14:19]),
            ]
output_include = [8, 0, 1, 2, 3]
error_ylabels = ['MAP [mmHg]',
                 'EDV [mL]',
                 'ESV [mL]',
                 r'max $p_{LV}$ [mmHg]',
                 r'min $p_{LV}$ [mmHg]',

                 '$L^2_{EDV}$ [mL]',
                 '$L^2_{ESV}$ [mL]',
                 '$L^2_{max pLv}$ [Pa]',
                 '$L^2_{min pLv}$ [Pa]',
                 ]

plot_against_calculation_time = False
plot_bar = True
bar_group_by = 1

for i_plot in range(len(axErrors)):
    i_data = output_include[i_plot]
    print('')
    for i_prot in range(n_groups):
        idx = idx_prot_groups == i_prot
        data = np.abs(semi_true_values[:, i_data] -
                      simulation_output[idx, :, i_data])

        if plot_against_calculation_time:
            # axErrors[i_plot].scatter(calculation_time[idx, :], data,
            #                          color=colors[prot_groups[i_prot]],
            #                          alpha=0.5,
            #                          s=1,
            #                          )

            axErrors[i_plot].errorbar(
                np.nanpercentile(calculation_time[idx, :], 50, axis=1),
                np.nanpercentile(data, 50, axis=1),
                np.abs(np.nanpercentile(
                    data, [50], axis=1) - np.nanpercentile(data, [5, 95], axis=1)),
                np.abs(np.nanpercentile(calculation_time[idx, :], [
                       50], axis=1) - np.nanpercentile(calculation_time[idx, :], [5, 95], axis=1)),
                color=colors[prot_groups[i_prot]],
                lw=2,
                ls='-',
            )

        elif plot_bar:
            data_percentile = np.nanpercentile(data, [5, 25, 50, 75, 95],
                                               axis=1)

            n = len(range(len(data_percentile[2, :])))
            w = 1 / (n_groups + 2)

            if bar_group_by == 0:
                bar_X = np.linspace(0, n-1, n) + i_prot*w
                c = colors[prot_groups[i_prot]]
            elif bar_group_by == 1:
                bar_X = np.linspace(i_prot, i_prot+0.9, n) - 0.45

            # axErrors[i_plot].bar(
            #     bar_X,
            #     data_percentile[2, :],
            #     width=w,
            #     color=colors[prot_groups[i_prot]],
            #     )
            # axErrors[i_plot].errorbar(
            #     bar_X,
            #     data_percentile[2, :],
            #     yerr = np.abs(data_percentile[2, :] - data_percentile[[0, 4], :]),
            #     color=colors[prot_groups[i_prot]],
            #     lw=2,
            #     ls='',
            #     )
            medians = []
            for i in range(data.shape[0]):
                if bar_group_by == 1:
                    c = np.array([0.9, 0.9, 0.9]) * (i / data.shape[0]) ** 0.5
                axErrors[i_plot].boxplot(
                    [data[i, np.invert(np.isnan(data[i, :]))]],
                    positions=[bar_X[i]],
                    notch=False, patch_artist=True,
                    showfliers=False,
                    boxprops=dict(facecolor=(*c, 0.3), color=c),
                    capprops=dict(color=c),
                    whiskerprops=dict(color=c),
                    # flierprops=dict(color=c, markeredgecolor=c),
                    medianprops=dict(color=c),
                    widths=w,
                )
                medians.append(
                    np.median([data[i, np.invert(np.isnan(data[i, :]))]]))
            # axErrors[i_plot].plot(bar_X, medians, c=[0.5,0.5,0.5])

                if all_dt[i] == 1e-3:
                    print(i_plot, i_prot, i,
                          error_ylabels[i_plot], prot_groups[i_prot], all_dt[i])
                    print(np.nanpercentile(
                        [data[i, np.invert(np.isnan(data[i, :]))]], 97.5))

        else:

            data_percentile = np.nanpercentile(data, [5, 25, 50, 75, 95],
                                               axis=1)

            axErrors[i_plot].plot(all_dt[idx], data_percentile[2, :],
                                  color=colors[prot_groups[i_prot]])
            axErrors[i_plot].fill_between(all_dt[idx], data_percentile[0, :], data_percentile[4, :],
                                          color=colors[prot_groups[i_prot]],
                                          zorder=7, alpha=0.1)
            axErrors[i_plot].fill_between(all_dt[idx], data_percentile[1, :], data_percentile[3, :],
                                          color=colors[prot_groups[i_prot]],
                                          zorder=8, alpha=0.1)

            # axErrors[i_plot].errorbar(
            #     all_dt[idx],
            #     data_percentile[2, :],
            #     yerr = np.abs(data_percentile[2, :] - data_percentile[[0, 4], :]),
            #     color=colors[prot_groups[i_prot]],
            #     lw=2,
            #     ls='-',
            #     )

        # print(error_ylabels[i_plot], prot_groups[i_prot], ': {:.2E}'.format(data_percentile[2, 0]))
        # print(error_ylabels[i_plot], prot_groups[i_prot], ': {:.2E}'.format(data_percentile[2, 3]))
        # print(error_ylabels[i_plot], prot_groups[i_prot], ': {:.2E}'.format(data_percentile[2, -1]))

    if not plot_bar:
        axErrors[i_plot].set_xscale('log')
    if plot_bar and bar_group_by == 0:
        axErrors[i_plot].set_xticks(np.linspace(0, n-1, n), all_dt[idx]*1e3)
    if plot_bar and bar_group_by == 1:
        labels = [l.replace('_', ' ') for l in prot_groups]
        axErrors[i_plot].set_xticks(np.linspace(0, n_groups-1, n_groups),
                                    labels, rotation=20, ha='right', va='top')
    axErrors[i_plot].set_yscale('log')
    axErrors[i_plot].set_ylim([2e-5, 2e0])
    axErrors[i_plot].set_ylabel(error_ylabels[i_plot],
                                fontsize=fontsettings['label_fontsize'])
    axErrors[i_plot].spines['top'].set_visible(False)
    axErrors[i_plot].spines['right'].set_visible(False)
    axErrors[i_plot].axhline(0.1, c='k', ls='--')

# axErrors[3].set_xlabel('Integration step size $\\Delta t$ [s]',
#                   fontsize=fontsettings['label_fontsize'])
# axErrors[4].set_xlabel('Integration step size $\\Delta t$ [s]',
#                   fontsize=fontsettings['label_fontsize'])


# Legend
axLegend1 = fig.add_subplot(gs[0:3, 10:13])

for i in range(data.shape[0]):
    c = np.array([0.9, 0.9, 0.9]) * (i / data.shape[0]) ** 0.5
    print(i, c)
    axLegend1.boxplot([[0, 1, 2, 3, 4]], 0, 'rs', 0, positions=[i*0.2],
                      patch_artist=True,
                      showfliers=False,
                      boxprops=dict(facecolor=(*c, 0.3), color=c),
                      capprops=dict(color=c),
                      whiskerprops=dict(color=c),
                      # flierprops=dict(color=c, markeredgecolor=c),
                      medianprops=dict(color=c),
                      widths=w,
                      )

    axLegend1.annotate(f'{all_dt[i]*1e3} ms', (4.5, i*0.2),
                       horizontalalignment='left', verticalalignment='center',
                       fontsize=fontsettings['legend_fontsize'])


axLegend1.set_xticks([])
axLegend1.set_yticks([])
axLegend1.set_xlim([-1, 9])
axLegend1.set_ylim([-1, 2])
# axLegend1.spines['top'].set_visible(False)
# axLegend1.spines['right'].set_visible(False)
# axLegend1.spines['bottom'].set_visible(False)
# axLegend1.spines['left'].set_visible(False)

# left_location = subplots_adjust['left'] + 0.75*(
#     subplots_adjust['right']-subplots_adjust['left'])


# Legend
axLegend = fig.add_subplot(gs[4:7, 3:5])
axLegend.set_facecolor((0, 0, 0, 0))

for i_prot in range(n_groups):
    axLegend.plot([0, 1], [i_prot, i_prot],
                  color=colors[prot_groups[i_prot]])
    axLegend.fill_between([0, 1], [i_prot-0.15, i_prot-0.15], [i_prot+0.15, i_prot+0.15],
                          color=colors[prot_groups[i_prot]], alpha=0.1)
    axLegend.fill_between([0, 1], [i_prot-0.3, i_prot-0.3], [i_prot+0.3, i_prot+0.3],
                          color=colors[prot_groups[i_prot]], alpha=0.1)
    axLegend.annotate(prot_groups[i_prot], (1.1, i_prot),
                      horizontalalignment='left', verticalalignment='center',
                      fontsize=fontsettings['legend_fontsize'])

axLegend.set_xticks([])
axLegend.set_yticks([])
axLegend.set_xlim([0, 5])
axLegend.set_ylim([-2, n_groups+1])
axLegend.spines['top'].set_visible(False)
axLegend.spines['right'].set_visible(False)
axLegend.spines['bottom'].set_visible(False)
axLegend.spines['left'].set_visible(False)

left_location = subplots_adjust['left'] + 0.75*(
    subplots_adjust['right']-subplots_adjust['left'])

axLegend.annotate('Integration error (relative)', (left_location, 0.975),
                  xycoords='figure fraction',
                  horizontalalignment='center', verticalalignment='center',
                  fontsize=fontsettings['title_fontsize'],
                  fontweight=fontsettings['title_fontweight'])


# %% FINAL PLOT
fig = plt.figure(100, clear=True, figsize=(12, 8))
gs = fig.add_gridspec(3, 4)

subplots_adjust = {'top': 0.93,
                   'bottom': 0.11,
                   'left': 0.09,
                   'right': 0.988,
                   'hspace': 0.5,
                   'wspace': 0.4,
                   }
fig.subplots_adjust(**subplots_adjust)

# axLeft = fig.add_subplot(gs[0, 3])

# ## Calculation time plot
# for i_prot in range(n_groups):
#     idx = idx_prot_groups == i_prot

#     plot_ranges = np.nanpercentile(calculation_time[idx, :]*1e-3, [5, 25, 50, 75, 95], axis=1)

#     include_sims = np.sum(np.isnan(calculation_time[idx, :]), axis=1) < (n_sims / 2)

#     axLeft.plot(all_dt[idx][include_sims], plot_ranges[2, include_sims],
#                 color=colors[prot_groups[i_prot]], zorder=9)

#     axLeft.fill_between(all_dt[idx][include_sims], plot_ranges[0, include_sims], plot_ranges[4, include_sims],
#                         color=colors[prot_groups[i_prot]], zorder=7, alpha=0.1)
#     axLeft.fill_between(all_dt[idx][include_sims], plot_ranges[1, include_sims], plot_ranges[3, include_sims],
#                         color=colors[prot_groups[i_prot]], zorder=8, alpha=0.1)

# axLeft.set_yscale('log')
# axLeft.set_xscale('log')

# axLeft.set_xlabel('Integration step size $\\Delta t$ [s]',
#                   fontsize=fontsettings['label_fontsize'])
# axLeft.set_ylabel('Calculation time [s]',
#                   fontsize=fontsettings['label_fontsize'])
# axLeft.spines['top'].set_visible(False)
# axLeft.spines['right'].set_visible(False)

# left_location = subplots_adjust['left'] + 0.2*(
#     subplots_adjust['right']-subplots_adjust['left'])

####
axLeft1 = fig.add_subplot(gs[0, 3])
boxplot_group_by_solver(axLeft1, calculation_time*1e-3, n_groups, idx_prot_groups, prot_groups,
                        y_log=True)
axLeft1.set_ylabel('Calculation time [s]',
                   fontsize=fontsettings['label_fontsize'])
axLeft1.annotate('Single beat calculation time', (0.99, 0.975),
                 xycoords='figure fraction',
                 horizontalalignment='right', verticalalignment='center',
                 fontsize=fontsettings['title_fontsize'],
                 fontweight=fontsettings['title_fontweight'])


axLeft2 = fig.add_subplot(gs[1, 3])
percentage_crashed = np.sum(
    simulation_succes == False, axis=1)/simulation_succes.shape[1]*100
# plt.bar(range(len(list_of_protocols)), )
plt.ylabel('Crashed [%]', fontsize=fontsettings['label_fontsize'])
offset = 1
boxplot_group_by_solver(axLeft2, percentage_crashed+offset, n_groups, idx_prot_groups, prot_groups,
                        y_log=False, bar=True)
axLeft2.set_yticks(np.array([0, 50, 100])+offset, labels=[0, 50, 100])


# Verification plots Energy
axEnergy = fig.add_subplot(gs[0, 1])
for i_prot in range(n_groups):
    idx = idx_prot_groups == i_prot
    error_in_energy = 100 * 1 / 2 * (simulation_output[idx, :, 10] - simulation_output[idx, :, 11]) / (
        simulation_output[idx, :, 10] + simulation_output[idx, :, 11])
    data_percentile = np.nanpercentile(error_in_energy, [5, 25, 50, 75, 95],
                                       axis=1)

    # axEnergy.plot(all_dt[idx], data_percentile[2, :],
    #                       color=colors[prot_groups[i_prot]])
    # axEnergy.fill_between(all_dt[idx], data_percentile[0, :], data_percentile[4, :],
    #                               color=colors[prot_groups[i_prot]],
    #                               zorder=7, alpha=0.1)
    # axEnergy.fill_between(all_dt[idx], data_percentile[1, :], data_percentile[3, :],
    #                               color=colors[prot_groups[i_prot]],
    #                               zorder=8, alpha=0.1)
    if bar_group_by == 0:
        bar_X = np.linspace(
            0, error_in_energy.shape[0]-1, error_in_energy.shape[0]) + i_prot*w
        c = colors[prot_groups[i_prot]]
    elif bar_group_by == 1:
        bar_X = np.linspace(i_prot, i_prot+0.9,
                            error_in_energy.shape[0]) - 0.45
    for i in range(error_in_energy.shape[0]):
        if bar_group_by == 1:
            c = np.array([0.9, 0.9, 0.9]) * \
                (i / error_in_energy.shape[0]) ** 0.5
        axEnergy.boxplot(
            [error_in_energy[i, np.invert(np.isnan(error_in_energy[i, :]))]],
            positions=[bar_X[i]],
            notch=False, patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor=(*c, 0.3), color=c),
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            # flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color=c),
            widths=w,
        )
    axEnergy.plot(bar_X, data_percentile[4, :], c=colors[prot_groups[i_prot]])
if plot_bar and bar_group_by == 1:
    labels = [l.replace('_', ' ') for l in prot_groups]
    axEnergy.set_xticks(np.linspace(0, n_groups-1, n_groups),
                        labels, rotation=25, ha='right', va='top',
                        fontsize=9)

# axEnergy.set_xscale('log')
axEnergy.set_yscale('log')
axEnergy.set_ylim([1e-11, 1e1])
axEnergy.annotate('Verification Patch and TriSeg', (0.05, 0.975),
                  xycoords='figure fraction',
                  horizontalalignment='left', verticalalignment='center',
                  fontsize=fontsettings['title_fontsize'],
                  fontweight=fontsettings['title_fontweight'])
axEnergy.set_title('Error in local and global energy',
                   fontsize=fontsettings['label_fontsize'])
axEnergy.set_ylabel('error [%]',
                    fontsize=fontsettings['label_fontsize'])
# axEnergy.set_xlabel('Integration step size $\\Delta t$ [s]',
#                   fontsize=fontsettings['label_fontsize'])

# Verification plots TriSeg
axTriSegX = fig.add_subplot(gs[0, 2])
for i_prot in range(n_groups):
    idx = idx_prot_groups == i_prot
    # error_in_energy = 1 / 2 * (simulation_output[idx, :, 10] - simulation_output[idx, :, 11]) / (simulation_output[idx, :, 10] + simulation_output[idx, :, 11])
    # error_in_energy = simulation_output[idx, :, 12] / simulation_output[idx, :, 14]
    error_in_energy = np.sqrt(simulation_output[idx, :, 12]**2 + simulation_output[idx, :, 13]**2) / np.sqrt(
        simulation_output[idx, :, 14]**2 + simulation_output[idx, :, 15]**2)
    data_percentile = np.nanpercentile(error_in_energy, [5, 25, 50, 75, 95],
                                       axis=1)

#     axTriSegX.plot(all_dt[idx], data_percentile[2, :],
#                           color=colors[prot_groups[i_prot]])
#     axTriSegX.fill_between(all_dt[idx], data_percentile[0, :], data_percentile[4, :],
#                                   color=colors[prot_groups[i_prot]],
#                                   zorder=7, alpha=0.1)
#     axTriSegX.fill_between(all_dt[idx], data_percentile[1, :], data_percentile[3, :],
#                                   color=colors[prot_groups[i_prot]],
#                                   zorder=8, alpha=0.1)

    if bar_group_by == 0:
        bar_X = np.linspace(
            0, error_in_energy.shape[0]-1, error_in_energy.shape[0]) + i_prot*w
        c = colors[prot_groups[i_prot]]
    elif bar_group_by == 1:
        bar_X = np.linspace(i_prot, i_prot+0.9,
                            error_in_energy.shape[0]) - 0.45
    for i in range(error_in_energy.shape[0]):
        if bar_group_by == 1:
            c = np.array([0.9, 0.9, 0.9]) * \
                (i / error_in_energy.shape[0]) ** 0.5
        axTriSegX.boxplot(
            [error_in_energy[i, np.invert(np.isnan(error_in_energy[i, :]))]],
            positions=[bar_X[i]],
            notch=False, patch_artist=True,
            showfliers=False,
            boxprops=dict(facecolor=(*c, 0.3), color=c),
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            # flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color=c),
            widths=w,
        )
    axTriSegX.plot(bar_X, data_percentile[4, :], c=colors[prot_groups[i_prot]])
if plot_bar and bar_group_by == 1:
    labels = [l.replace('_', ' ') for l in prot_groups]
    axTriSegX.set_xticks(np.linspace(0, n_groups-1, n_groups),
                         labels, rotation=25, ha='right', va='top',
                         fontsize=9)
# axTriSegX.set_xscale('log')
axTriSegX.set_yscale('log')
axTriSegX.set_ylim([1e-11, 1e0])
axTriSegX.set_title('Tension imbalance',
                    fontsize=fontsettings['label_fontsize'])
axTriSegX.set_ylabel('error [%]',
                     fontsize=fontsettings['label_fontsize'])
axTriSegX.set_xlabel('Integration step size $\\Delta t$ [s]',
                     fontsize=fontsettings['label_fontsize'])


# error plots
axErrors = [fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[1, 2]),
            # fig.add_subplot(gs[2, 0]),
            # fig.add_subplot(gs[2, 1]),
            # fig.add_subplot(gs[2, 2]),
            ]
output_include = [8, 0, 1, 2, 3]
output_include = [1, 2]
error_ylabels = [  # 'MAP [mmHg]',
    # 'EDV [mL]',
    'ESV [mL]',
    r'max $p_{LV}$ [mmHg]',
    # r'min $p_{LV}$ [mmHg]',

    '$L^2_{EDV}$ [mL]',
    '$L^2_{ESV}$ [mL]',
    '$L^2_{max pLv}$ [Pa]',
    '$L^2_{min pLv}$ [Pa]',
]

plot_against_calculation_time = False
plot_bar = True
bar_group_by = 1

for i_plot in range(len(axErrors)):
    i_data = output_include[i_plot]
    print('')
    for i_prot in range(n_groups):
        idx = idx_prot_groups == i_prot
        data = np.abs(semi_true_values[:, i_data] -
                      simulation_output[idx, :, i_data])

        if plot_against_calculation_time:
            # axErrors[i_plot].scatter(calculation_time[idx, :], data,
            #                          color=colors[prot_groups[i_prot]],
            #                          alpha=0.5,
            #                          s=1,
            #                          )

            axErrors[i_plot].errorbar(
                np.nanpercentile(calculation_time[idx, :], 50, axis=1),
                np.nanpercentile(data, 50, axis=1),
                np.abs(np.nanpercentile(
                    data, [50], axis=1) - np.nanpercentile(data, [5, 95], axis=1)),
                np.abs(np.nanpercentile(calculation_time[idx, :], [
                       50], axis=1) - np.nanpercentile(calculation_time[idx, :], [5, 95], axis=1)),
                color=colors[prot_groups[i_prot]],
                lw=2,
                ls='-',
            )

        elif plot_bar:
            data_percentile = np.nanpercentile(data, [5, 25, 50, 75, 95],
                                               axis=1)

            n = len(range(len(data_percentile[2, :])))
            w = 1 / (n_groups + 2)

            if bar_group_by == 0:
                bar_X = np.linspace(0, n-1, n) + i_prot*w
                c = colors[prot_groups[i_prot]]
            elif bar_group_by == 1:
                bar_X = np.linspace(i_prot, i_prot+0.9, n) - 0.45

            axErrors[i_plot].axhline(0.1, c='k', ls='-', lw=0.5)

            # axErrors[i_plot].bar(
            #     bar_X,
            #     data_percentile[2, :],
            #     width=w,
            #     color=colors[prot_groups[i_prot]],
            #     )
            # axErrors[i_plot].errorbar(
            #     bar_X,
            #     data_percentile[2, :],
            #     yerr = np.abs(data_percentile[2, :] - data_percentile[[0, 4], :]),
            #     color=colors[prot_groups[i_prot]],
            #     lw=2,
            #     ls='',
            #     )
            medians = []
            for i in range(data.shape[0]):
                if bar_group_by == 1:
                    c = np.array([0.9, 0.9, 0.9]) * (i / data.shape[0]) ** 0.5
                axErrors[i_plot].boxplot(
                    [data[i, np.invert(np.isnan(data[i, :]))]],
                    positions=[bar_X[i]],
                    notch=False, patch_artist=True,
                    showfliers=False,
                    boxprops=dict(facecolor=(*c, 0.3), color=c),
                    capprops=dict(color=c),
                    whiskerprops=dict(color=c),
                    # flierprops=dict(color=c, markeredgecolor=c),
                    medianprops=dict(color=c),
                    widths=w,
                )
                medians.append(
                    np.median([data[i, np.invert(np.isnan(data[i, :]))]]))
            axErrors[i_plot].plot(
                bar_X, data_percentile[4, :], c=colors[prot_groups[i_prot]])

        else:

            data_percentile = np.nanpercentile(data, [5, 25, 50, 75, 95],
                                               axis=1)

            axErrors[i_plot].plot(all_dt[idx], data_percentile[2, :],
                                  color=colors[prot_groups[i_prot]])
            axErrors[i_plot].fill_between(all_dt[idx], data_percentile[0, :], data_percentile[4, :],
                                          color=colors[prot_groups[i_prot]],
                                          zorder=7, alpha=0.1)
            axErrors[i_plot].fill_between(all_dt[idx], data_percentile[1, :], data_percentile[3, :],
                                          color=colors[prot_groups[i_prot]],
                                          zorder=8, alpha=0.1)

            # axErrors[i_plot].errorbar(
            #     all_dt[idx],
            #     data_percentile[2, :],
            #     yerr = np.abs(data_percentile[2, :] - data_percentile[[0, 4], :]),
            #     color=colors[prot_groups[i_prot]],
            #     lw=2,
            #     ls='-',
            #     )

        # print(error_ylabels[i_plot], prot_groups[i_prot], ': {:.2E}'.format(data_percentile[2, 0]))
        # print(error_ylabels[i_plot], prot_groups[i_prot], ': {:.2E}'.format(data_percentile[2, 3]))
        # print(error_ylabels[i_plot], prot_groups[i_prot], ': {:.2E}'.format(data_percentile[2, -1]))

    if not plot_bar:
        axErrors[i_plot].set_xscale('log')
    if plot_bar and bar_group_by == 0:
        axErrors[i_plot].set_xticks(np.linspace(0, n-1, n), all_dt[idx]*1e3)
    if plot_bar and bar_group_by == 1:
        labels = [l.replace('_', ' ') for l in prot_groups]
        axErrors[i_plot].set_xticks(np.linspace(0, n_groups-1, n_groups),
                                    labels, rotation=25, ha='right', va='top',
                                    fontsize=9)
    axErrors[i_plot].set_yscale('log')
    axErrors[i_plot].set_ylim([2e-5, 2e0])
    axErrors[i_plot].set_ylabel(error_ylabels[i_plot],
                                fontsize=fontsettings['label_fontsize'])
    axErrors[i_plot].spines['top'].set_visible(False)
    axErrors[i_plot].spines['right'].set_visible(False)


# axErrors[3].set_xlabel('Integration step size $\\Delta t$ [s]',
#                   fontsize=fontsettings['label_fontsize'])
# axErrors[4].set_xlabel('Integration step size $\\Delta t$ [s]',
#                   fontsize=fontsettings['label_fontsize'])


# Legend
axLegend1 = fig.add_subplot(gs[1, 0])

for i in range(data.shape[0]):
    c = np.array([0.9, 0.9, 0.9]) * (i / data.shape[0]) ** 0.5
    print(i, c)
    axLegend1.boxplot([[0, 1, 2, 3, 4]], 0, 'rs', 0, positions=[i*0.2],
                      patch_artist=True,
                      showfliers=False,
                      boxprops=dict(facecolor=(*c, 0.3), color=c),
                      capprops=dict(color=c),
                      whiskerprops=dict(color=c),
                      # flierprops=dict(color=c, markeredgecolor=c),
                      medianprops=dict(color=c),
                      widths=w,
                      )

    axLegend1.annotate(f'{all_dt[i]*1e3} ms', (4.5, i*0.2),
                       horizontalalignment='left', verticalalignment='center',
                       fontsize=fontsettings['legend_fontsize'])


axLegend1.set_xticks([])
axLegend1.set_yticks([])
axLegend1.set_xlim([-1, 9])
axLegend1.set_ylim([-1, 2])
# axLegend1.spines['top'].set_visible(False)
# axLegend1.spines['right'].set_visible(False)
# axLegend1.spines['bottom'].set_visible(False)
# axLegend1.spines['left'].set_visible(False)

# left_location = subplots_adjust['left'] + 0.75*(
#     subplots_adjust['right']-subplots_adjust['left'])


# Legend
axLegend = fig.add_subplot(gs[0, 0])
axLegend.set_facecolor((0, 0, 0, 0))

for i_prot in range(n_groups):
    axLegend.plot([0, 1], [i_prot, i_prot],
                  color=colors[prot_groups[i_prot]])
    axLegend.fill_between([0, 1], [i_prot-0.15, i_prot-0.15], [i_prot+0.15, i_prot+0.15],
                          color=colors[prot_groups[i_prot]], alpha=0.1)
    axLegend.fill_between([0, 1], [i_prot-0.3, i_prot-0.3], [i_prot+0.3, i_prot+0.3],
                          color=colors[prot_groups[i_prot]], alpha=0.1)
    axLegend.annotate(prot_groups[i_prot], (1.1, i_prot),
                      horizontalalignment='left', verticalalignment='center',
                      fontsize=fontsettings['legend_fontsize'])

axLegend.set_xticks([])
axLegend.set_yticks([])
axLegend.set_xlim([0, 5])
axLegend.set_ylim([-2, n_groups+1])
axLegend.spines['top'].set_visible(False)
axLegend.spines['right'].set_visible(False)
axLegend.spines['bottom'].set_visible(False)
axLegend.spines['left'].set_visible(False)

left_location = subplots_adjust['left'] + 0.75*(
    subplots_adjust['right']-subplots_adjust['left'])

axLegend.annotate('Integration error (relative)', (0.05, 0.64),
                  xycoords='figure fraction',
                  horizontalalignment='left', verticalalignment='center',
                  fontsize=fontsettings['title_fontsize'],
                  fontweight=fontsettings['title_fontweight'])


# %% Blant Altman
i_fig = 0
fig = plt.figure(102, figsize=(9.5, 9.5), clear=True)

subplots_adjust = {'top': 0.85,
                   'bottom': 0.085,
                   'left': 0.18,
                   'right': 0.82,
                   'hspace': 0.3,
                   'wspace': 0.4,
                   }
fig.subplots_adjust(**subplots_adjust)

include_protocols = np.sum(
    simulation_succes[:, :] == False, axis=1) < simulation_succes.shape[1]
include_protocols = np.argwhere(include_protocols)[:, 0]

include_protocols = all_dt == np.min(all_dt)
include_protocols = np.argwhere(include_protocols)[:, 0]

m = len(include_protocols)
n = len(include_protocols)

axes = [[], []]
max_diff = [0, 0]

i_dats = [2, 1]


# CORNERS
ax = plt.subplot(m, n, 1)
ax.spines['left'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.set_xticks([])
ax.set_yticks([])
plt.annotate('B: ' + protocol_names[include_protocols[0]].split('\n')[0].replace('_', '\n'),
             xycoords='axes fraction',
             xy=(-1.7, 0.5),
             fontsize=9,
             rotation=0,
             ha='center', va='center',
             )
plt.annotate('A: ' + protocol_names[include_protocols[0]].split('\n')[0].replace('_', '\n'),
             xycoords='axes fraction',
             xy=(0.5, 1.7),
             fontsize=9,
             rotation=0,
             ha='center', va='top',
             )

ax = plt.subplot(m, n, m*n)
ax.spines['left'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.set_xticks([])
ax.set_yticks([])
plt.annotate('B: ' + protocol_names[include_protocols[-1]].split('\n')[0].replace('_', '\n'),
             xycoords='axes fraction',
             xy=(2.7, 0.5),
             fontsize=9,
             rotation=0,
             ha='center', va='center',
             )

# Plots


for i_protocol1a in range(len(include_protocols)):
    for i_protocol0a in range(i_protocol1a+1, len(include_protocols)):

        i_protocol1 = include_protocols[i_protocol1a]
        i_protocol0 = include_protocols[i_protocol0a]

        for i, i_dat in enumerate(i_dats):

            if i == 1:
                ax = plt.subplot(m, n, 1 + i_protocol1a + (i_protocol0a) * n)
            else:
                # Right Top
                ax = plt.subplot(m, n, 1 + i_protocol0a + (i_protocol1a) * n)
            scatX = simulation_output[i_protocol0, :, i_dat]
            scatY = simulation_output[i_protocol1, :, i_dat]

            axes[i].append(ax)
            # adjust_spines(ax, ['left', 'bottom'])

            xl = np.array([np.nanmin(0.5*(scatX+scatY)),
                          np.nanmax(0.5*(scatX+scatY))])

            ax.set_xlim(xl)
            ax.scatter(0.5*(scatX+scatY), (scatX-scatY), s=1)
            # ax.set_title('STD: {:.2f}'.format(np.nanstd((scatX-scatY))))
            ax.plot([0, xl[1]], [0, 0], c='k', linewidth=1)

            # plt.yscale('log')
            md = np.nanmax(np.abs(scatX-scatY))
            if md > max_diff[i]:
                max_diff[i] = md

            if (i_protocol0 == 40 or i_protocol1 == 40) and i_dat == 2:
                print(i_protocol0, i_protocol1, i_dat,
                      np.nanmean((scatX-scatY)))

            xticks = ax.get_xticks()
            yticks = ax.get_yticks()
            if i == 1:
                ax.spines['right'].set_color('none')
                ax.spines['top'].set_color('none')
                if i_protocol0a == m-1:
                    plt.xlabel('1/2(A+B)')
                    ax.set_xticks(xticks, None)
                else:
                    ax.set_xticks(xticks, ['' for _ in xticks])

                if i_protocol1a == 0:
                    plt.ylabel('A-B')
                    plt.annotate('B: ' + protocol_names[i_protocol0].split('\n')[0].replace('_', '\n'),
                                 xycoords='axes fraction',
                                 xy=(-1.7, 0.5),
                                 fontsize=9,
                                 rotation=0,
                                 ha='center', va='center',
                                 )
                    ax.set_yticks(yticks)
                else:
                    ax.set_yticks(yticks, ['' for _ in yticks])

            else:
                ax.spines['left'].set_color('none')
                ax.spines['top'].set_color('none')
                if i_protocol1a == 0:
                    # plt.title('A: '+ protocol_names[i_protocol0].split('\n')[0].replace('_', '\n'),
                    #           fontsize=9)
                    plt.annotate('A: ' + protocol_names[i_protocol0].split('\n')[0].replace('_', '\n'),
                                 xycoords='axes fraction',
                                 xy=(0.5, 1.7),
                                 fontsize=9,
                                 rotation=0,
                                 ha='center', va='top',
                                 )

                if i_protocol1a == i_protocol0a-1:
                    # plt.xlabel('1/2(A+B)')
                    ax.set_xticks(xticks, None)
                else:
                    xticks = ax.get_xticks()
                    ax.set_xticks(xticks, ['' for _ in xticks])

                if i_protocol0a == m-1:
                    plt.annotate('B: ' + protocol_names[i_protocol1].split('\n')[0].replace('_', '\n'),
                                 xycoords='axes fraction',
                                 xy=(2.7, 0.5),
                                 fontsize=9,
                                 rotation=0,
                                 ha='center', va='center',
                                 )
                    plt.ylabel('A-B')
                    ax.yaxis.set_label_position("right")
                    ax.set_yticks(yticks)
                else:
                    ax.set_yticks(yticks, ['' for _ in yticks])
                ax.yaxis.tick_right()
            ax.set_xlim(xl)


# max_diff = 1
for i in range(2):
    for ax in axes[i]:
        ax.set_ylim([-max_diff[i]*1.05, max_diff[i]*1.05])
        md = 10**np.floor(np.log10(max_diff[i]))

        options = np.array([1, 2, 5, 10])
        md *= options[np.argmax(md*options > max_diff[i])]

        ax.set_yticks([-md, 0, md])

plt.annotate(output_names[i_dats[0]],
             xycoords='figure fraction',
             xy=(0.5, 0.95),
             fontsize=14,
             weight='bold',
             ha='center', va='center',
             )

plt.annotate(output_names[i_dats[1]],
             xycoords='figure fraction',
             xy=(0.02, 0.5),
             rotation=90,
             fontsize=14,
             weight='bold',
             ha='center', va='center',
             )

plt.tight_layout()


# %%
fig = plt.figure(101, clear=True, figsize=(12, 6))
gs = fig.add_gridspec(2, 3)

subplots_adjust = {'top': 0.93,
                   'bottom': 0.11,
                   'left': 0.09,
                   'right': 0.988,
                   'hspace': 0.5,
                   'wspace': 0.4,
                   }
fig.subplots_adjust(**subplots_adjust)

####
axLeft1 = fig.add_subplot(gs[1, 0])
axLeft1.spines['top'].set_visible(False)
axLeft1.spines['right'].set_visible(False)

# boxplot_group_by_solver(axLeft1, calculation_time*1e-3, n_groups, idx_prot_groups, prot_groups,
#                             y_log=True)

for i_prot in range(n_groups):
    idx = idx_prot_groups == i_prot
    data_percentile = np.nanpercentile(calculation_time[idx], [5, 25, 50, 75, 95],
                                       axis=1)

    # axLeft1.plot(all_dt[idx], data_percentile[2, :], c=colors[prot_groups[i_prot]])
    axLeft1.plot(all_dt[idx]*1e3, np.nanmean(calculation_time[idx]
                 * 1e-3, axis=1), c=colors[prot_groups[i_prot]])

axLeft1.set_xscale('log')
axLeft1.set_yscale('log')

axLeft1.set_ylabel('Mean calculation time [s]',
                   fontsize=fontsettings['label_fontsize'])
axLeft1.annotate('Single beat calculation time', (0.01, 0.47),
                 xycoords='figure fraction',
                 horizontalalignment='left', verticalalignment='center',
                 fontsize=fontsettings['title_fontsize'],
                 fontweight=fontsettings['title_fontweight'])
axLeft1.spines['top'].set_visible(False)
axLeft1.spines['right'].set_visible(False)

axLeft1.set_xlabel('Integration step size $\\Delta t$ [ms]',
                  fontsize=fontsettings['label_fontsize'])


# axLeft2 = fig.add_subplot(gs[1, 3])
# percentage_crashed = np.sum(simulation_succes==False, axis=1)/simulation_succes.shape[1]*100
# # plt.bar(range(len(list_of_protocols)), )
# plt.ylabel('Crashed [%]', fontsize=fontsettings['label_fontsize'])
# offset = 1
# boxplot_group_by_solver(axLeft2, percentage_crashed+offset, n_groups, idx_prot_groups, prot_groups,
#                             y_log=False, bar=True)
# axLeft2.set_yticks(np.array([0, 50, 100])+offset, labels=[0, 50, 100])

# axLeft2.spines['top'].set_visible(False)
# axLeft2.spines['right'].set_visible(False)

# def dt_to_str(t):
#     if t < 1e-3:
#         return f'{t*1e6:.0f} $\\mu$s'
#     if t < 1e-0:
#         return f'{t*1e3:.0f} ms'
#     return f'{t:.0f} s'

# legend_dt = ['t='+dt_to_str(t) for t in np.unique(all_dt)]
# axLeft2.legend(legend_dt, loc='upper center', bbox_to_anchor=(0.5, 1.5),
#           ncol=2, fancybox=False, shadow=False)

# Verification plots Energy
axEnergy = fig.add_subplot(gs[0, 0])
for i_prot in range(n_groups):
    idx = idx_prot_groups == i_prot
    error_in_energy = np.abs(
        simulation_output[idx, :, 10] - simulation_output[idx, :, 11])
    data_percentile = np.nanpercentile(error_in_energy, [5, 25, 50, 75, 95],
                                       axis=1)

    axEnergy.plot(all_dt[idx]*1e3, data_percentile[4, :],
                  c=colors[prot_groups[i_prot]])

axEnergy.set_xscale('log')
axEnergy.set_yscale('log')
axEnergy.set_ylim([5e-9, 5e-2])
# axEnergy.annotate('Verification Patch and TriSeg', (0.05, 0.975),
#                   xycoords='figure fraction',
#                   horizontalalignment='left', verticalalignment='center',
#                   fontsize=fontsettings['title_fontsize'],
#                   fontweight=fontsettings['title_fontweight'])
axEnergy.set_title('Error in local and global energy',
                   fontsize=fontsettings['label_fontsize'])
axEnergy.set_ylabel('error [kPa]',
                    fontsize=fontsettings['label_fontsize'])
axEnergy.spines['top'].set_visible(False)
axEnergy.spines['right'].set_visible(False)

axEnergy.set_xlabel('Integration step size $\\Delta t$ [ms]',
                  fontsize=fontsettings['label_fontsize'])

# Verification plots TriSeg
axTriSegX = fig.add_subplot(gs[0, 1])
for i_prot in range(n_groups):
    idx = idx_prot_groups == i_prot
    # error_in_energy = 1 / 2 * (simulation_output[idx, :, 10] - simulation_output[idx, :, 11]) / (simulation_output[idx, :, 10] + simulation_output[idx, :, 11])
    # error_in_energy = simulation_output[idx, :, 12] / simulation_output[idx, :, 14]
    # / np.sqrt(simulation_output[idx, :, 14]**2 + simulation_output[idx, :, 15]**2)
    error_in_energy = np.sqrt(
        simulation_output[idx, :, 12]**2 + simulation_output[idx, :, 13]**2)
    data_percentile = np.nanpercentile(error_in_energy, [5, 25, 50, 75, 95],
                                       axis=1)

    axTriSegX.plot(all_dt[idx]*1e3, data_percentile[4, :],
                   c=colors[prot_groups[i_prot]])

axTriSegX.set_xscale('log')
axTriSegX.set_yscale('log')
axTriSegX.set_ylim([5e-8, 5e-1])
axTriSegX.set_title('Tension imbalance',
                    fontsize=fontsettings['label_fontsize'])
axTriSegX.set_ylabel('Mean absolute error [N/m]',
                     fontsize=fontsettings['label_fontsize'])
axTriSegX.set_xlabel('Integration step size $\\Delta t$ [ms]',
                     fontsize=fontsettings['label_fontsize'])
axTriSegX.spines['top'].set_visible(False)
axTriSegX.spines['right'].set_visible(False)


# error plots
axErrors = [fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[1, 2]),
            # fig.add_subplot(gs[2, 0]),
            # fig.add_subplot(gs[2, 1]),
            # fig.add_subplot(gs[2, 2]),
            ]
output_include = [8, 0, 1, 2, 3]
output_include = [1, 2]
# error_ylabels = [# 'MAP [mmHg]',
#                  # 'EDV [mL]',
#                   'min $V_{LV}$ [mL]',
#                  r'max $p_{LV}$ [mmHg]',
#                  # r'min $p_{LV}$ [mmHg]',

#                  '$L^2_{EDV}$ [mL]',
#                  '$L^2_{ESV}$ [mL]',
#                  '$L^2_{max pLv}$ [Pa]',
#                  '$L^2_{min pLv}$ [Pa]',
#                  ]
error_ylabels = [
    'mean absolute error\nLV volume [mL]',
    'mean absolute error\nLV pressure [mmHg]',
]


plot_against_calculation_time = False
plot_bar = True
bar_group_by = 1

for i_plot in range(len(axErrors)):
    # i_data = output_include[i_plot]
    i_data = i_plot
    print('')
    for i_prot in range(n_groups):
        idx = idx_prot_groups == i_prot
        # data = np.abs(semi_true_values[:, i_data] - simulation_output[idx, :, i_data])

        # data_percentile = np.nanpercentile(data, [5, 25, 50, 75, 95],
        #                                    axis=1)

        # n = len(range(len(data_percentile[2, :])))
        w = 1 / (n_groups + 2)

        # if bar_group_by == 0:
        #     bar_X = np.linspace(0, n-1, n) + i_prot*w
        #     c = colors[prot_groups[i_prot]]
        # elif bar_group_by == 1:
        #     bar_X = np.linspace(i_prot, i_prot+0.9, n) - 0.45

        axErrors[i_plot].axhline(0.1, c='k', ls='-', lw=0.5)

        # axErrors[i_plot].plot(all_dt[idx], data_percentile[4, :], c=colors[prot_groups[i_prot]])
        axErrors[i_plot].plot(
            all_dt[idx]*1e3,
            np.nanmean(signals_mean_abs_error_true[idx, :, i_plot], axis=1),
            c=colors[prot_groups[i_prot]])


        errors_for_dt = np.nanmean(signals_mean_abs_error_true[idx, :, i_plot], axis=1)
        i_dt = np.nanargmin(errors_for_dt<0.1) - 1
        print(i_plot, prot_groups[i_prot], all_dt[idx][i_dt], errors_for_dt[i_dt], errors_for_dt[-3:-1])

    axErrors[i_plot].set_xscale('log')
    axErrors[i_plot].set_yscale('log')
    axErrors[i_plot].set_ylim([2e-5, 2e0])
    axErrors[i_plot].set_ylabel(error_ylabels[i_plot],
                                fontsize=fontsettings['label_fontsize'])
    axErrors[i_plot].spines['top'].set_visible(False)
    axErrors[i_plot].spines['right'].set_visible(False)


axErrors[0].set_xlabel('Integration step size $\\Delta t$ [ms]',
                  fontsize=fontsettings['label_fontsize'])
axErrors[1].set_xlabel('Integration step size $\\Delta t$ [ms]',
                  fontsize=fontsettings['label_fontsize'])


# Legend

# Legend
axLegend = fig.add_subplot(gs[0, 2])
axLegend.set_facecolor((0, 0, 0, 0))


for i_prot in range(n_groups):
    axLegend.plot([0, 1], [i_prot, i_prot],
                  color=colors[prot_groups[i_prot]])
    # axLegend.fill_between([0, 1], [i_prot-0.15, i_prot-0.15], [i_prot+0.15, i_prot+0.15],
    #                       color=colors[prot_groups[i_prot]], alpha=0.1)
    # axLegend.fill_between([0, 1], [i_prot-0.3, i_prot-0.3], [i_prot+0.3, i_prot+0.3],
    #                       color=colors[prot_groups[i_prot]], alpha=0.1)
    axLegend.annotate(prot_groups[i_prot], (1.1, i_prot),
                      horizontalalignment='left', verticalalignment='center',
                      fontsize=fontsettings['legend_fontsize'])

axLegend.set_xticks([])
axLegend.set_yticks([])
axLegend.set_xlim([0, 5])
axLegend.set_ylim([-2, n_groups+1])
axLegend.spines['top'].set_visible(False)
axLegend.spines['right'].set_visible(False)
axLegend.spines['bottom'].set_visible(False)
axLegend.spines['left'].set_visible(False)

left_location = subplots_adjust['left'] + 0.75*(
    subplots_adjust['right']-subplots_adjust['left'])

axLegend.annotate('Integration error', (0.45, 0.47),
                  xycoords='figure fraction',
                  horizontalalignment='left', verticalalignment='center',
                  fontsize=fontsettings['title_fontsize'],
                  fontweight=fontsettings['title_fontweight'])
