# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import addcopyfighandler

import circadapt

from tqdm import tqdm

from _functions import *
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

#%% forward_euler

use_plot_comparison_01ms = False
use_plot_solverFE = False
use_plot_solverCA = False
use_plot_solverBDF1 = False
use_plot_solverVar = False


all_thresh = [#1,
                1e-1,
                1e-2,
                1e-3,
                1e-4,
                1e-5,
                1e-6,
              ]
list_of_protocols = get_list_of_protocols(
    dt=[
        0.0001,
        0.001,
        0.002,
        0.005
        ],
    TriSeg_thresh_F=all_thresh,
    # solvers=['backward_differential_o2'],
    solvers=['adams_moulton_o2'],
    )

# %% Get reference state
model = circadapt.VanOsta2024()
model.run(1)
model_state = model.model_export()

#%% Run Protocol
calculation_time = np.ndarray((len(list_of_protocols), n_sims)) + np.nan
simulation_succes = np.zeros((len(list_of_protocols), n_sims), dtype=bool)
simulation_output = np.ndarray((len(list_of_protocols), n_sims, n_out)) + np.nan
simulation_signals = np.ndarray((len(list_of_protocols), n_sims), dtype=object)

for i_protocol, protocol in enumerate(list_of_protocols):
    print('Run Protocol ', i_protocol, '/', len(list_of_protocols))

    # load prerunned data
    run_protocol = False
    filename = folder_name+'/'+protocol['name']+'.npy'
    try:
        data = np.load(filename, allow_pickle=True).item()
        calculation_time[i_protocol,:] = data['calculation_time']
        simulation_succes[i_protocol,:] = data['simulation_succes']
        simulation_output[i_protocol,:] = data['simulation_output']
        simulation_signals[i_protocol, :] = data['simulation_signals']

        if calculation_time.shape[1]!=n_sims:
            run_protocol=True
    except:
        run_protocol = True

    if run_protocol:
        for i_sim in tqdm(range(n_sims), desc="Simulation Progress"):
            # print('\r', i_sim, end='\r')
            # load model
            solver = protocol['solver']
            if solver[-3:-1] == '_o':
                model = circadapt.VanOsta2024(solver[:-3])
                model.model_import(model_state)
                model.set('Solver.order', int(solver[-1]))
            else:
                model = circadapt.VanOsta2024(solver)
                model.model_import(model_state)
                

            #set paramteers
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
            'calculation_time': calculation_time[i_protocol,:],
            'simulation_succes': simulation_succes[i_protocol,:],
            'simulation_output': simulation_output[i_protocol,:],
            'simulation_signals': simulation_signals[i_protocol, :],
            }
        np.save(filename, data)

print('finished running simulations')


# %%

# %% Calculate error in signals

# get simulations with lowest dt
lowest_dt = 1
idx_prot = []
all_solvers = []
all_dt = []
all_thresh = []
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
    all_thresh.append(protocol['set'][2][1])

idx_prot = np.array(idx_prot)
all_dt = np.array(all_dt)
all_thresh = np.array(all_thresh)
prot_groups, idx_prot_groups = np.unique(all_solvers, False, True)
n_groups = np.max(idx_prot_groups)+1
prot_has_min_dt = (all_dt == np.min(all_dt)) & (all_thresh == np.min(all_thresh))


# first signal is time
# calculate error
n_signals = simulation_signals[0][0].shape[0]
signals_mean_abs_error = np.empty((simulation_signals.shape[0], *simulation_signals.shape, n_signals-1))
signals_mean_abs_error[:] = np.nan

# %%

mean_signal = [[] for i_sim in range(signals_mean_abs_error.shape[1])]
for i_sim in range(signals_mean_abs_error.shape[1]):
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
            time, idx0, idx1 = np.intersect1d(time0, time1, return_indices=True)
            e = np.abs(data0[:, idx0] - data1[:, idx1])
            signals_mean_abs_error[i_prot0, i_prot1, i_sim, :] = np.mean(e, axis=1)

    # calculate mean signal
    # correction adam bashforth
    aux = simulation_signals[prot_has_min_dt, i_sim]
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
signals_mean_abs_error_true = np.empty((*simulation_signals.shape, n_signals-1))
signals_mean_abs_error_true[:] = np.nan
for i_sim in range(signals_mean_abs_error.shape[1]):
    time_true = mean_signal[i_sim][0]
    data_true = mean_signal[i_sim][1:]
    for i_prot0 in range(signals_mean_abs_error.shape[0]):
        if simulation_signals[i_prot0, i_sim] is None:
            continue
        data0 = simulation_signals[i_prot0, i_sim][1:, :-1]
        time0 = simulation_signals[i_prot0, i_sim][0, :-1]

        interp_data = np.empty_like(data0)

        for i_data in range(interp_data.shape[0]):
            interp_data[i_data] = np.interp(time0, time_true, data_true[i_data])



        signals_mean_abs_error_true[i_prot0, i_sim, :] = np.nanmean(np.abs(interp_data - data0), axis=1)




#%%
plt.figure(1, figsize=(12, 8))
plt.clf()
m=2
n=1

calculation_time_filtered = [1e-3*calculation_time[i_prot, ~np.isnan(calculation_time[i_prot,:])] for i_prot in range(calculation_time.shape[0])]
protocol_names = [list_of_protocols[i_prot]['name'].replace(' ', '\n') for i_prot in range(len(list_of_protocols))]

plt.subplot(m,n,1)
plt.boxplot(calculation_time_filtered, whis=[2.5, 97.5], showfliers=False)
plt.xticks(range(1, len(protocol_names)+1), ['' for _ in range(len(protocol_names))], rotation=45)
plt.xlim(0, len(protocol_names)+1)
plt.yscale('log')
# plt.ylim([1e-0, 1e3])
plt.ylabel('Computation time [s]', fontsize=14, fontweight='bold')

plt.subplot(m,n,2)
plt.bar(range(len(list_of_protocols)), np.sum(simulation_succes==False, axis=1)/simulation_succes.shape[1]*100)
plt.ylabel('Crashed simulations [%]', fontsize=14, fontweight='bold')
plt.xticks(range(len(protocol_names)), [n.replace('\n', ' ') for n in protocol_names], rotation=45, fontsize=9, fontweight='bold',
           ha='right', )
plt.xlim(-1, len(protocol_names))

plt.tight_layout()

#%%
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

    include_protocols = np.sum(simulation_succes==False, axis=1)<simulation_succes.shape[1]
    include_protocols = np.argwhere(include_protocols)[:,0]

    m=len(include_protocols)
    n=len(include_protocols)

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
                if i==0:
                    ax = plt.subplot(m,n,1 + i_protocol1a + (i_protocol0a) * n)
                    scatX = simulation_output[i_protocol0, :, i_dat]
                    scatY = simulation_output[i_protocol1, :, i_dat]
                else:
                    # Right Top
                    ax = plt.subplot(m,n,1 + i_protocol0a+ (i_protocol1a ) * n)
                    scatX = simulation_output[i_protocol0, :, i_dat+1]
                    scatY = simulation_output[i_protocol1, :, i_dat+1]

                axes.append(ax)
                adjust_spines(ax, ['left', 'bottom'])

                xl = np.array([np.nanmin(0.5*(scatX+scatY)), np.nanmax(0.5*(scatX+scatY))])

                ax.set_xlim(xl)
                ax.scatter(0.5*(scatX+scatY), (scatX-scatY), s=1)
                ax.plot([0, xl[1]], [0,0], c='k', linewidth=1)

                # plt.yscale('log')
                md = np.nanmax(np.abs( scatX-scatY ))
                if md>max_diff:
                    max_diff = md
                md = np.nanmin(np.abs(scatX-scatY))
                if md<min_diff:
                    min_diff = md

                if i==0:
                    if i_protocol0a==m-1:
                        plt.xlabel(protocol_names[i_protocol1] )

                    if i_protocol1a==0:
                        plt.ylabel(protocol_names[i_protocol0] )


    for ax in axes:
        ax.set_ylim([-max_diff*1.2, max_diff*1.2])


    plt.tight_layout()


# %%
i_fig = 0
plt.figure(3 + i_fig)
plt.clf()

include_protocols = np.sum(simulation_succes==False, axis=1)<simulation_succes.shape[1]
include_protocols = np.argwhere(include_protocols)[:,0]

m=len(include_protocols)
n=len(include_protocols)

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
            if i==0:
                scatX = simulation_output[i_protocol0, :, i_dat]
                scatY = simulation_output[i_protocol1, :, i_dat]

                idx = (np.isnan(scatX)==False) & (np.isnan(scatY)==False)

                scatX = scatX[idx]
                scatY = scatY[idx]

                corr[i_protocol1a, i_protocol0a] = np.corrcoef(scatX, scatY)[0, 1]
            else:
                # Right Top
                scatX = simulation_output[i_protocol0, :, i_dat]
                scatY = simulation_output[i_protocol1, :, i_dat]

                idx = (np.isnan(scatX)==False) & (np.isnan(scatY)==False)

                scatX = scatX[idx]
                scatY = scatY[idx]

                corr[i_protocol0a, i_protocol1a] = np.corrcoef(scatX, scatY)[1, 0]






#%%

# get simulations with lowest dt
lowest_thresh = np.inf
idx_prot = []
all_solvers = []
all_dt = []
all_thresh = []
for i_protocol, protocol in enumerate(list_of_protocols):
    if lowest_thresh == protocol['set'][2][1]:
        idx_prot.append(i_protocol)
    elif protocol['set'][2][1] < lowest_thresh:
        idx_prot = [i_protocol]
        lowest_thresh = protocol['set'][2][1]

    # solvers
    s = protocol['solver']
    all_solvers.append(s)
    all_dt.append(protocol['set'][0][1])
    all_thresh.append(protocol['set'][2][1])

idx_prot = np.array(idx_prot)
all_dt = np.array(all_dt)
all_thresh = np.array(all_thresh)
prot_groups, idx_prot_groups = np.unique(all_solvers, False, True)
prot_groups, idx_prot_groups = np.unique(all_dt, False, True)
n_groups = np.max(idx_prot_groups)+1

prot_groups = [f"{x:.0e}" for x in prot_groups]
# True values: mean of lowest dt's
semi_true_values = np.mean(simulation_output[[idx_prot[0]],:], axis=0)

plt.figure(10, figsize=(12,8), clear=True)
m=5
n=6

for i_data in range(semi_true_values.shape[1]):
    ax = plt.subplot(m,n,i_data+1)
    for i_prot in range(n_groups):
        data = np.abs( semi_true_values[:,i_data] - simulation_output[idx_prot_groups==i_prot,:,i_data])
        data /= semi_true_values[:,i_data]
        data_percentile = np.nanpercentile(data, [2.5, 25, 50, 75, 97.5], axis=1)
        # plt.fill_between(all_dt[idx_prot_groups==i_prot],
        #                  data_percentile[0,:],
        #                  data_percentile[-1,:],
        #                  alpha=0.5)
        # plt.fill_between(all_dt[idx_prot_groups==i_prot],
        #                  data_percentile[1,:],
        #                  data_percentile[3,:],
        #                  alpha=0.5)
        plt.plot(all_dt[idx_prot_groups==i_prot], data_percentile[2,:], label=prot_groups[i_prot])
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

## Calculation time plot
for i_prot in range(n_groups):
    idx = idx_prot_groups == i_prot

    plot_ranges = np.nanpercentile(calculation_time[idx, :]*1e-3, [2.5, 25, 50, 75, 97.5], axis=1)

    include_sims = np.sum(np.isnan(calculation_time[idx, :]), axis=1) < (n_sims / 2)

    axLeft.plot(all_thresh[idx][include_sims], plot_ranges[2, include_sims],
                color=colors[prot_groups[i_prot]], zorder=9)

    axLeft.fill_between(all_thresh[idx][include_sims], plot_ranges[0, include_sims], plot_ranges[4, include_sims],
                        color=colors[prot_groups[i_prot]], zorder=7, alpha=0.1)
    axLeft.fill_between(all_thresh[idx][include_sims], plot_ranges[1, include_sims], plot_ranges[3, include_sims],
                        color=colors[prot_groups[i_prot]], zorder=8, alpha=0.1)

axLeft.set_yscale('log')
axLeft.set_xscale('log')

axLeft.set_xlabel('TriSeg Threshold force balance [-]',
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

#### Verification plots Energy
axEnergy = fig.add_subplot(gs[:3, :3])
for i_prot in range(n_groups):
    idx = idx_prot_groups == i_prot
    error_in_energy = 1 / 2 * (simulation_output[idx, :, 10] - simulation_output[idx, :, 11]) / (simulation_output[idx, :, 10] + simulation_output[idx, :, 11])
    data_percentile = np.nanpercentile(error_in_energy, [2.5, 25, 50, 75, 97.5],
                                       axis=1)

    axEnergy.plot(all_thresh[idx], data_percentile[2, :],
                          color=colors[prot_groups[i_prot]])
    axEnergy.fill_between(all_thresh[idx], data_percentile[0, :], data_percentile[4, :],
                                  color=colors[prot_groups[i_prot]],
                                  zorder=7, alpha=0.1)
    axEnergy.fill_between(all_thresh[idx], data_percentile[1, :], data_percentile[3, :],
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

#### Verification plots TriSeg
axTriSegX = fig.add_subplot(gs[:3, 4:7])
for i_prot in range(n_groups):
    idx = idx_prot_groups == i_prot
    # error_in_energy = 1 / 2 * (simulation_output[idx, :, 10] - simulation_output[idx, :, 11]) / (simulation_output[idx, :, 10] + simulation_output[idx, :, 11])
    # error_in_energy = simulation_output[idx, :, 12] / simulation_output[idx, :, 14]
    error_in_energy = np.sqrt(simulation_output[idx, :, 12]**2 + simulation_output[idx, :, 13]**2)
    data_percentile = np.nanpercentile(error_in_energy, [2.5, 25, 50, 75, 97.5],
                                       axis=1)

    axTriSegX.plot(all_thresh[idx], data_percentile[2, :],
                          color=colors[prot_groups[i_prot]])
    axTriSegX.fill_between(all_thresh[idx], data_percentile[0, :], data_percentile[4, :],
                                  color=colors[prot_groups[i_prot]],
                                  zorder=7, alpha=0.1)
    axTriSegX.fill_between(all_thresh[idx], data_percentile[1, :], data_percentile[3, :],
                                  color=colors[prot_groups[i_prot]],
                                  zorder=8, alpha=0.1)
axTriSegX.set_xscale('log')
axTriSegX.set_yscale('log')
axTriSegX.set_ylim([1e-11, 1e0])
axTriSegX.set_title('Tension imbalance',
                  fontsize=fontsettings['label_fontsize'])
axTriSegX.set_ylabel('error [N/m]',
                fontsize=fontsettings['label_fontsize'])
axTriSegX.set_xlabel('Integration step size $\\Delta t$ [s]',
                  fontsize=fontsettings['label_fontsize'])



#### error plots
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
        data = np.abs(semi_true_values[:, i_data] - simulation_output[idx, :, i_data])

        if plot_against_calculation_time:
            # axErrors[i_plot].scatter(calculation_time[idx, :], data,
            #                          color=colors[prot_groups[i_prot]],
            #                          alpha=0.5,
            #                          s=1,
            #                          )

            axErrors[i_plot].errorbar(
                np.nanpercentile(calculation_time[idx, :], 50, axis=1),
                np.nanpercentile(data, 50, axis=1),
                np.abs(np.nanpercentile(data, [50], axis=1) - np.nanpercentile(data, [2.5, 97.5], axis=1)),
                np.abs(np.nanpercentile(calculation_time[idx, :], [50], axis=1) - np.nanpercentile(calculation_time[idx, :], [2.5, 97.5], axis=1)),
                color=colors[prot_groups[i_prot]],
                lw=2,
                ls='-',
                )

        elif plot_bar:
            data_percentile = np.nanpercentile(data, [2.5, 25, 50, 75, 97.5],
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
                medians.append(np.median([data[i, np.invert(np.isnan(data[i, :]))]]))
            # axErrors[i_plot].plot(bar_X, medians, c=[0.5,0.5,0.5])




        else:

            data_percentile = np.nanpercentile(data, [2.5, 25, 50, 75, 97.5],
                                               axis=1)

            axErrors[i_plot].plot(all_thresh[idx], data_percentile[2, :],
                                  color=colors[prot_groups[i_prot]])
            axErrors[i_plot].fill_between(all_thresh[idx], data_percentile[0, :], data_percentile[4, :],
                                          color=colors[prot_groups[i_prot]],
                                          zorder=7, alpha=0.1)
            axErrors[i_plot].fill_between(all_thresh[idx], data_percentile[1, :], data_percentile[3, :],
                                          color=colors[prot_groups[i_prot]],
                                          zorder=8, alpha=0.1)

            # axErrors[i_plot].errorbar(
            #     all_thresh[idx],
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
    if plot_bar and bar_group_by==0:
        axErrors[i_plot].set_xticks(np.linspace(0, n-1, n), all_thresh[idx]*1e3)
    if plot_bar and bar_group_by==1:
        labels = [l.replace('_', ' ') for l in prot_groups]
        axErrors[i_plot].set_xticks(np.linspace(0, n_groups-1, n_groups),
                                    labels, rotation=20, ha='right', va = 'top')
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





#### Legend
axLegend1 = fig.add_subplot(gs[0:3, 10:13])

for i in range(data.shape[0]):
    c = np.array([0.9, 0.9, 0.9]) * (i / data.shape[0]) ** 0.5
    print(i, c)
    axLegend1.boxplot([[0,1,2,3,4]], 0, 'rs', 0, positions=[i*0.2],
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor=(*c, 0.3), color=c),
        capprops=dict(color=c),
        whiskerprops=dict(color=c),
        # flierprops=dict(color=c, markeredgecolor=c),
        medianprops=dict(color=c),
        widths=w,
        )

    axLegend1.annotate(f'{all_thresh[i]*1e3} ms', (4.5, i*0.2),
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



#### Legend
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

#     plot_ranges = np.nanpercentile(calculation_time[idx, :]*1e-3, [2.5, 25, 50, 75, 97.5], axis=1)

#     include_sims = np.sum(np.isnan(calculation_time[idx, :]), axis=1) < (n_sims / 2)

#     axLeft.plot(all_thresh[idx][include_sims], plot_ranges[2, include_sims],
#                 color=colors[prot_groups[i_prot]], zorder=9)

#     axLeft.fill_between(all_thresh[idx][include_sims], plot_ranges[0, include_sims], plot_ranges[4, include_sims],
#                         color=colors[prot_groups[i_prot]], zorder=7, alpha=0.1)
#     axLeft.fill_between(all_thresh[idx][include_sims], plot_ranges[1, include_sims], plot_ranges[3, include_sims],
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
percentage_crashed = np.sum(simulation_succes==False, axis=1)/simulation_succes.shape[1]*100
# plt.bar(range(len(list_of_protocols)), )
plt.ylabel('Crashed [%]', fontsize=fontsettings['label_fontsize'])
offset = 1
boxplot_group_by_solver(axLeft2, percentage_crashed+offset, n_groups, idx_prot_groups, prot_groups,
                            y_log=False, bar=True)
axLeft2.set_yticks(np.array([0, 50, 100])+offset, labels=[0, 50, 100])


#### Verification plots Energy
axEnergy = fig.add_subplot(gs[0, 1])
for i_prot in range(n_groups):
    idx = idx_prot_groups == i_prot
    error_in_energy = 1 / 2 * (simulation_output[idx, :, 10] - simulation_output[idx, :, 11]) / (simulation_output[idx, :, 10] + simulation_output[idx, :, 11])
    data_percentile = np.nanpercentile(error_in_energy, [2.5, 25, 50, 75, 97.5],
                                       axis=1)

    axEnergy.plot(all_thresh[idx], data_percentile[2, :],
                          color=colors[prot_groups[i_prot]])
    axEnergy.fill_between(all_thresh[idx], data_percentile[0, :], data_percentile[4, :],
                                  color=colors[prot_groups[i_prot]],
                                  zorder=7, alpha=0.1)
    axEnergy.fill_between(all_thresh[idx], data_percentile[1, :], data_percentile[3, :],
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

#### Verification plots TriSeg
axTriSegX = fig.add_subplot(gs[0, 2])
for i_prot in range(n_groups):
    idx = idx_prot_groups == i_prot
    # error_in_energy = 1 / 2 * (simulation_output[idx, :, 10] - simulation_output[idx, :, 11]) / (simulation_output[idx, :, 10] + simulation_output[idx, :, 11])
    # error_in_energy = simulation_output[idx, :, 12] / simulation_output[idx, :, 14]
    error_in_energy = np.sqrt(simulation_output[idx, :, 12]**2 + simulation_output[idx, :, 13]**2)  / np.sqrt(simulation_output[idx, :, 14]**2 + simulation_output[idx, :, 15]**2)
    data_percentile = np.nanpercentile(error_in_energy, [2.5, 25, 50, 75, 97.5],
                                       axis=1)

    axTriSegX.plot(all_thresh[idx], data_percentile[2, :],
                          color=colors[prot_groups[i_prot]])
    axTriSegX.fill_between(all_thresh[idx], data_percentile[0, :], data_percentile[4, :],
                                  color=colors[prot_groups[i_prot]],
                                  zorder=7, alpha=0.1)
    axTriSegX.fill_between(all_thresh[idx], data_percentile[1, :], data_percentile[3, :],
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



#### error plots
axErrors = [fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[1, 2]),
            fig.add_subplot(gs[2, 0]),
            fig.add_subplot(gs[2, 1]),
            fig.add_subplot(gs[2, 2]),
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
        data = np.abs(semi_true_values[:, i_data] - simulation_output[idx, :, i_data])

        if plot_against_calculation_time:
            # axErrors[i_plot].scatter(calculation_time[idx, :], data,
            #                          color=colors[prot_groups[i_prot]],
            #                          alpha=0.5,
            #                          s=1,
            #                          )

            axErrors[i_plot].errorbar(
                np.nanpercentile(calculation_time[idx, :], 50, axis=1),
                np.nanpercentile(data, 50, axis=1),
                np.abs(np.nanpercentile(data, [50], axis=1) - np.nanpercentile(data, [2.5, 97.5], axis=1)),
                np.abs(np.nanpercentile(calculation_time[idx, :], [50], axis=1) - np.nanpercentile(calculation_time[idx, :], [2.5, 97.5], axis=1)),
                color=colors[prot_groups[i_prot]],
                lw=2,
                ls='-',
                )

        elif plot_bar:
            data_percentile = np.nanpercentile(data, [2.5, 25, 50, 75, 97.5],
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
                medians.append(np.median([data[i, np.invert(np.isnan(data[i, :]))]]))
            # axErrors[i_plot].plot(bar_X, medians, c=[0.5,0.5,0.5])




        else:

            data_percentile = np.nanpercentile(data, [2.5, 25, 50, 75, 97.5],
                                               axis=1)

            axErrors[i_plot].plot(all_thresh[idx], data_percentile[2, :],
                                  color=colors[prot_groups[i_prot]])
            axErrors[i_plot].fill_between(all_thresh[idx], data_percentile[0, :], data_percentile[4, :],
                                          color=colors[prot_groups[i_prot]],
                                          zorder=7, alpha=0.1)
            axErrors[i_plot].fill_between(all_thresh[idx], data_percentile[1, :], data_percentile[3, :],
                                          color=colors[prot_groups[i_prot]],
                                          zorder=8, alpha=0.1)

            # axErrors[i_plot].errorbar(
            #     all_thresh[idx],
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
    if plot_bar and bar_group_by==0:
        axErrors[i_plot].set_xticks(np.linspace(0, n-1, n), all_thresh[idx]*1e3)
    if plot_bar and bar_group_by==1:
        labels = [l.replace('_', ' ') for l in prot_groups]
        axErrors[i_plot].set_xticks(np.linspace(0, n_groups-1, n_groups),
                                    labels, rotation=25, ha='right', va = 'top',
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





#### Legend
axLegend1 = fig.add_subplot(gs[1, 0])

for i in range(data.shape[0]):
    c = np.array([0.9, 0.9, 0.9]) * (i / data.shape[0]) ** 0.5
    print(i, c)
    axLegend1.boxplot([[0,1,2,3,4]], 0, 'rs', 0, positions=[i*0.2],
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor=(*c, 0.3), color=c),
        capprops=dict(color=c),
        whiskerprops=dict(color=c),
        # flierprops=dict(color=c, markeredgecolor=c),
        medianprops=dict(color=c),
        widths=w,
        )

    axLegend1.annotate(f'threshold = {all_thresh[i]}', (4.5, i*0.2),
                      horizontalalignment='left', verticalalignment='center',
                      fontsize=fontsettings['legend_fontsize'])


axLegend1.set_xticks([])
axLegend1.set_yticks([])
axLegend1.set_xlim([-1, 15])
axLegend1.set_ylim([-0.2, 1.2])
# axLegend1.spines['top'].set_visible(False)
# axLegend1.spines['right'].set_visible(False)
# axLegend1.spines['bottom'].set_visible(False)
# axLegend1.spines['left'].set_visible(False)

# left_location = subplots_adjust['left'] + 0.75*(
#     subplots_adjust['right']-subplots_adjust['left'])



#### Legend
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


# %%
fig = plt.figure(101, clear=True, figsize=(12, 8))
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
    data_percentile = np.nanpercentile(calculation_time[idx]*1e-3, [2.5, 25, 50, 75, 97.5],
                                        axis=1)

    # axLeft1.plot(all_thresh[idx], data_percentile[2, :], c=colors[prot_groups[i_prot]])

    axLeft1.fill_between(all_thresh[idx],
                      data_percentile[0, :],
                      data_percentile[-1, :],
                      fc = colors[prot_groups[i_prot]],
                      alpha = 0.1,
                      zorder=97)
    axLeft1.fill_between(all_thresh[idx],
                                     data_percentile[1, :],
                                     data_percentile[3, :],
                                     fc=colors[prot_groups[i_prot]],
                                     alpha=0.6,
                                     zorder=98)
    axLeft1.plot(all_thresh[idx], data_percentile[2,:],
             c=colors[prot_groups[i_prot]],
             zorder=99)



axLeft1.set_xscale('log')
axLeft1.set_yscale('log')

axLeft1.set_ylabel('Calculation time [s]',
                  fontsize=fontsettings['label_fontsize'])
axLeft1.annotate('Single beat calculation time', (0.05, 0.47),
                xycoords='figure fraction',
                horizontalalignment='left', verticalalignment='center',
                fontsize=fontsettings['title_fontsize'],
                fontweight=fontsettings['title_fontweight'])
axLeft1.spines['top'].set_visible(False)
axLeft1.spines['right'].set_visible(False)


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

#### Verification plots Energy
axEnergy = fig.add_subplot(gs[0, 0])
for i_prot in range(n_groups):
    idx = idx_prot_groups == i_prot
    error_in_energy = np.abs(simulation_output[idx, :, 10] - simulation_output[idx, :, 11])
    data_percentile = np.nanpercentile(error_in_energy, [2.5, 25, 50, 75, 97.5],
                                       axis=1)

    # axEnergy.plot(all_thresh[idx], data_percentile[2, :], c=colors[prot_groups[i_prot]])
    # axEnergy.plot(all_thresh[idx], 1e3 * np.nanmean(error_in_energy, axis=1), c=colors[prot_groups[i_prot]])

    axEnergy.fill_between(all_thresh[idx],
                      data_percentile[0, :],
                      data_percentile[-1, :],
                      fc = colors[prot_groups[i_prot]],
                      alpha = 0.1,
                      zorder=97)
    axEnergy.fill_between(all_thresh[idx],
                                     data_percentile[1, :],
                                     data_percentile[3, :],
                                     fc=colors[prot_groups[i_prot]],
                                     alpha=0.6,
                                     zorder=98)
    axEnergy.plot(all_thresh[idx], data_percentile[2,:],
             c=colors[prot_groups[i_prot]],
             zorder=99)


axEnergy.set_xscale('log')
axEnergy.set_yscale('log')
axEnergy.set_ylim([5e-9, 5e-2])
axEnergy.annotate('Verification Patch and TriSeg', (0.05, 0.975),
                xycoords='figure fraction',
                horizontalalignment='left', verticalalignment='center',
                fontsize=fontsettings['title_fontsize'],
                fontweight=fontsettings['title_fontweight'])
axEnergy.set_title('Error in local and global energy',
                  fontsize=fontsettings['label_fontsize'])
axEnergy.set_ylabel('error [kPa]',
                fontsize=fontsettings['label_fontsize'])
axEnergy.spines['top'].set_visible(False)
axEnergy.spines['right'].set_visible(False)

# axEnergy.set_xlabel('Integration step size $\\Delta t$ [s]',
#                   fontsize=fontsettings['label_fontsize'])

#### Verification plots TriSeg
axTriSegX = fig.add_subplot(gs[0, 1])
for i_prot in range(n_groups):
    idx = idx_prot_groups == i_prot
    # error_in_energy = 1 / 2 * (simulation_output[idx, :, 10] - simulation_output[idx, :, 11]) / (simulation_output[idx, :, 10] + simulation_output[idx, :, 11])
    # error_in_energy = simulation_output[idx, :, 12] / simulation_output[idx, :, 14]
    error_in_energy = np.sqrt(simulation_output[idx, :, 12]**2 + simulation_output[idx, :, 13]**2)
    data_percentile = np.nanpercentile(error_in_energy, [2.5, 25, 50, 75, 97.5],
                                        axis=1)

    # axTriSegX.plot(all_thresh[idx], data_percentile[4, :], c=colors[prot_groups[i_prot]])
    # axTriSegX.plot(all_thresh[idx], np.nanmean(error_in_energy, axis=1), c=colors[prot_groups[i_prot]])

    axTriSegX.fill_between(all_thresh[idx],
                      data_percentile[0, :],
                      data_percentile[-1, :],
                      fc = colors[prot_groups[i_prot]],
                      alpha = 0.1)
    axTriSegX.fill_between(all_thresh[idx],
                                     data_percentile[1, :],
                                     data_percentile[3, :],
                                     fc=colors[prot_groups[i_prot]],
                                     alpha=0.6,
                                     zorder=98)
    axTriSegX.plot(all_thresh[idx], data_percentile[2,:],
             c=colors[prot_groups[i_prot]],
             zorder=99)


axTriSegX.set_xscale('log')
axTriSegX.set_yscale('log')
axTriSegX.set_ylim([5e-8, 5e-1])
axTriSegX.set_title('Tension imbalance',
                  fontsize=fontsettings['label_fontsize'])
axTriSegX.set_ylabel('mean error [N/m]',
                fontsize=fontsettings['label_fontsize'])
axTriSegX.set_xlabel('TriSeg Threshold force balance [-]',
                  fontsize=fontsettings['label_fontsize'])
axTriSegX.spines['top'].set_visible(False)
axTriSegX.spines['right'].set_visible(False)



#### error plots
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
    i_data = output_include[i_plot]
    print('')
    for i_prot in range(n_groups):
        idx = idx_prot_groups == i_prot
        data = np.abs(semi_true_values[:, i_data] - simulation_output[idx, :, i_data])

        data_percentile = np.nanpercentile(data, [2.5, 25, 50, 75, 97.5],
                                           axis=1)

        n = len(range(len(data_percentile[2, :])))
        w = 1 / (n_groups + 2)

        if bar_group_by == 0:
            bar_X = np.linspace(0, n-1, n) + i_prot*w
            c = colors[prot_groups[i_prot]]
        elif bar_group_by == 1:
            bar_X = np.linspace(i_prot, i_prot+0.9, n) - 0.45


        axErrors[i_plot].axhline(0.1, c='k', ls='-', lw=0.5)

        # axErrors[i_plot].plot(all_thresh[idx], data_percentile[4, :], c=colors[prot_groups[i_prot]])
        # axErrors[i_plot].plot(all_thresh[idx], np.nanmean(signals_mean_abs_error_true[idx, :, i_plot], axis=1), c=colors[prot_groups[i_prot]])

        axErrors[i_plot].fill_between(all_thresh[idx],
                          data_percentile[0, :],
                          data_percentile[-1, :],
                          fc = colors[prot_groups[i_prot]],
                          alpha = 0.1,
                          zorder=97)
        axErrors[i_plot].fill_between(all_thresh[idx],
                                         data_percentile[1, :],
                                         data_percentile[3, :],
                                         fc=colors[prot_groups[i_prot]],
                                         alpha=0.6,
                                         zorder=98)
        axErrors[i_plot].plot(all_thresh[idx], data_percentile[2,:],
                 c=colors[prot_groups[i_prot]], #label=list_of_protocols[include_prot[i_prot]]['legend'],
                 zorder=99)
        for i in [0,1,3,4]:
            axErrors[i_plot].plot(all_thresh[idx], data_percentile[i,:],
                     c=colors[prot_groups[i_prot]], #label=list_of_protocols[include_prot[i_prot]]['legend'],
                     zorder=99,
                     lw=0.5)



    axErrors[i_plot].set_xscale('log')
    axErrors[i_plot].set_yscale('log')
    axErrors[i_plot].set_ylim([2e-5, 2e1])
    axErrors[i_plot].set_ylabel(error_ylabels[i_plot],
                                fontsize=fontsettings['label_fontsize'])
    axErrors[i_plot].spines['top'].set_visible(False)
    axErrors[i_plot].spines['right'].set_visible(False)


axErrors[0].set_xlabel('TriSeg Threshold force balance [-]',
                  fontsize=fontsettings['label_fontsize'])
# axErrors[4].set_xlabel('Integration step size $\\Delta t$ [s]',
#                   fontsize=fontsettings['label_fontsize'])





#### Legend

#### Legend
axLegend = fig.add_subplot(gs[0, 2])
axLegend.set_facecolor((0, 0, 0, 0))

def dt_to_str(t):
    # if t < 1e-3:
    #     return f'{t*1e6:.0f} $\\mu$s'
    if t < 1e-3:
        return f'{t*1e3:.1f} ms'
    if t < 1e-1:
        return f'{t*1e3:.0f} ms'
    if t < 1e-0:
        return f'{t*1e3:.0f} ms'
    return f'{t:.0f} s'

legend_dt = ['t='+dt_to_str(float(t)) for t in prot_groups]

for i_prot in range(n_groups):
    axLegend.plot([0, 1], [i_prot, i_prot],
                  color=colors[prot_groups[i_prot]], zorder=99)
    axLegend.fill_between([0, 1], [i_prot-0.15, i_prot-0.15], [i_prot+0.15, i_prot+0.15],
                          color=colors[prot_groups[i_prot]], alpha=0.1, zorder=97)
    axLegend.fill_between([0, 1], [i_prot-0.3, i_prot-0.3], [i_prot+0.3, i_prot+0.3],
                          color=colors[prot_groups[i_prot]], alpha=0.1, zorder=98)
    axLegend.annotate(legend_dt[i_prot], (1.1, i_prot),
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

