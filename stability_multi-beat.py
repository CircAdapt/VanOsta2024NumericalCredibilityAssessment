# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import os

import circadapt
import time
import matplotlib.pyplot as plt

from tqdm import tqdm

from _functions import *
from _stability_functions import *
from _benchmark_functions import *

# set seed for reproducibility
np.random.seed(1)

# parameters
PFC_on = True

folder_name = 'data_multi-beat_PFC-' + ('on' if PFC_on else 'off')
# n_sims = 200 if PFC_on else 100
n_beats = 250 if PFC_on else 100

n_sims = 1000

# create samples
X = np.random.random((n_sims, n_par))

# define protocols
list_of_protocols = []

plot_colors = [[0.2, 0.5, 0.8], [0.5, 0.4, 0.2]]

#$umber: rgba(89, 79, 59, 1);
# $sage: rgba(201, 193, 159, 1);
# $star-command-blue: rgba(34, 116, 165, 1);
# $dark-purple: rgba(39, 7, 34, 1);
# $international-orange-aerospace: rgba(255, 78, 0, 1);
plot_colors = [[89/255, 79/255, 59/255],
               [201/255, 193/255, 159/255],
               [34/255, 116/255, 165/255],
               None,
               None,
               None]
plot_colors = [[65/255, 54/255, 32/255],
               [208/255, 0/255, 0/255],
               [159/255, 120/255, 51/255],
               [245/255, 205/255, 135/255],
               [192/255, 196/255, 209/255],
               None,
               None,
               None]



#%% SolverFE

list_of_protocols = (
    get_list_of_protocols(
        # solvers=['backward_differential_o2'],
        solvers=['adams_moulton_o2'],
        dt=[1e-4], TriSeg_thresh_F=[1e-3])
    +
    get_list_of_protocols(
        # solvers=['backward_differential_o2'],
        solvers=['adams_moulton_o2'],
        dt=[
            1e-3,
            2e-3,
            5e-3,
            ], TriSeg_thresh_F=[1e-1], fac_pfc=[1])
)


# list_of_protocols = list_of_protocols[12:13]
#
#%% Run Protocol
calculation_time = np.ndarray((len(list_of_protocols), n_sims, n_beats))
calculation_time[:] = np.nan
simulation_succes = np.zeros((len(list_of_protocols), n_sims, n_beats), dtype=bool)
simulation_stable = np.zeros((len(list_of_protocols), n_sims, n_beats), dtype=bool)
simulation_high_pressure = np.zeros((len(list_of_protocols), n_sims, n_beats), dtype=bool)
simulation_output = np.ndarray((len(list_of_protocols), n_sims, n_beats, n_out))
simulation_output[:] = np.nan
simulation_signals = np.ndarray((len(list_of_protocols), n_sims, n_beats), dtype=object)

for i_protocol, protocol in enumerate(list_of_protocols):
    print('Run Protocol ', i_protocol, '/', len(list_of_protocols))

    # load prerunned data
    run_protocol = False
    filename = folder_name+'/'+protocol['name']+'.npy'
    try:
        data = np.load(filename, allow_pickle=True).item()
        calculation_time[i_protocol,:] = data['calculation_time']
        simulation_succes[i_protocol,:] = data['simulation_succes']
        simulation_stable[i_protocol,:] = data['simulation_stable']
        simulation_high_pressure[i_protocol, :] = data['simulation_high_pressure']
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
            # load model
            solver = protocol['solver']
            if solver[-3:-1] == '_o':
                model = circadapt.VanOsta2024(solver[:-3])
                model.set('Solver.order', int(solver[-1]))
            else:
                model = circadapt.VanOsta2024(solver)

            model.set('Model.PFC.is_active', PFC_on)

            #set paramteers
            for s in protocol['set']:
                model.set(*s)

            # set paremeters
            setX(model, X[i_sim, :])
            high_pressure = False
            for i_beat in range(n_beats):
                # run
                t0 = time.time()
                try:
                    model.run(1)
                except circadapt.ModelCrashed:
                    pass
                t1 = time.time()
                dt = t1-t0

                is_crashed = model.get('Model.is_crashed')



                # is_crashed = is_crashed | high_pressure

                # store data
                if is_crashed:
                    print('Crashed ', i_sim, ' High pressure: ', high_pressure)
                    simulation_output[i_protocol, i_sim, i_beat:, :] = np.nan
                    simulation_stable[i_protocol, i_sim, i_beat:] = False
                    simulation_high_pressure[i_protocol, i_sim, i_beat:] = high_pressure
                    calculation_time[i_protocol, i_sim, i_beat:] = np.nan
                    break
                else:
                    calculation_time[i_protocol, i_sim, i_beat] = dt*1e3
                    simulation_succes[i_protocol, i_sim, i_beat] = True
                    simulation_stable[i_protocol, i_sim, i_beat] = model.is_stable()
                    simulation_high_pressure[i_protocol, i_sim, i_beat] = False
                    output, signals = getY(model)
                    simulation_output[i_protocol, i_sim, i_beat, :] = output
                    simulation_signals[i_protocol, i_sim, i_beat] = signals

                high_pressure = (np.mean(model['Cavity']['p'][:, 'La']) > 50*133)

        data = {
            'calculation_time': calculation_time[i_protocol],
            'simulation_succes': simulation_succes[i_protocol],
            'simulation_stable': simulation_stable[i_protocol],
            'simulation_high_pressure': simulation_high_pressure[i_protocol],
            'simulation_output': simulation_output[i_protocol],
            'simulation_signals': simulation_signals[i_protocol, :],
            }
        np.save(filename, data)
print('finished running simulations')

# %% Calc abs error relative to 'true' value


n_signals = simulation_signals[0][0][0].shape[0]
signals_mean_abs_error_true = np.empty((*simulation_signals.shape, n_signals-1))
signals_mean_abs_error_true[:] = np.nan
for i_sim in range(n_sims):
    true_signal = simulation_signals[0, i_sim, -1]
    if true_signal is None:
        continue
    time_true = true_signal[0]
    data_true = true_signal[1:]
    for i_prot0 in range(signals_mean_abs_error_true.shape[0]):
        if simulation_signals[i_prot0, i_sim] is None:
            continue

        for i_beat in range(simulation_signals.shape[2]):
            signal = simulation_signals[i_prot0, i_sim, i_beat]
            if signal is None:
                continue
            data0 = signal[1:, :-1]
            time0 = signal[0, :-1]

            interp_data = np.empty_like(data0)

            for i_data in range(interp_data.shape[0]):
                interp_data[i_data] = np.interp(time0, time_true, data_true[i_data])
            signals_mean_abs_error_true[i_prot0, i_sim, i_beat, :] = np.nanmean(np.abs(interp_data - data0), axis=1)


# %% Remove high pressures
# for i_protocol in range(len(simulation_output)):
#     idx_remove = (simulation_output[i_protocol, :, -1, 3] < 35) & (simulation_output[i_protocol, :, -1, 3] > 0)
#     simulation_output[i_protocol, idx_remove] = np.nan

# %% post processing
# get simulations with lowest dt
lowest_dt = 1
idx_prot = []
all_solvers = []
all_dt = []
all_solver_names = []
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

#%%
fig = plt.figure(1, figsize=(10,6), clear=True)
gs = fig.add_gridspec(11, 4)

subplots_adjust = {'top': 0.93,
                   'bottom': 0.085,
                   'left': 0.066,
                   'right': 0.988,
                   'hspace': 0.3,
                   'wspace': 0.4,
                   }
fig.subplots_adjust(**subplots_adjust)

m=4
n=2

calculation_time_filtered = [calculation_time[i_prot, ~np.isnan(calculation_time[i_prot,:,-1]), :].reshape(-1) for i_prot in range(calculation_time.shape[0])]
protocol_names = [list_of_protocols[i_prot]['legend'].replace(' ', '\n') for i_prot in range(len(list_of_protocols))]

ax = plt.subplot(m,n,2)
calculation_time_filtered = []
for i_prot in range(calculation_time.shape[0]):
    time_to_stable = []
    for i_sim in range(calculation_time.shape[1]):
        if np.sum(simulation_stable[i_prot, i_sim,:]==False)==n_beats:
            pass
            # time_to_stable.append(np.nan)
        else:
            time_to_stable.append(np.sum(calculation_time[i_prot, 0, simulation_stable[i_prot, i_sim,:]==False]))
    calculation_time_filtered.append(time_to_stable)

plt.boxplot(calculation_time_filtered, whis=[2.5, 97.5], showfliers=False)
plt.xticks(range(1, len(protocol_names)+1), ['' for _ in range(len(protocol_names))], rotation=45)
plt.xlim(0, len(protocol_names)+1)
plt.ylabel('Computation time \n hemodynamic stable [ms]', fontsize=10, fontweight='bold')
plt.yscale('log')
plt.grid(True, which='major', axis='y', c=[0.9,0.9,0.9])
adjust_spines(ax, ['left','bottom'], outward=0)


calculation_time_filtered = []
for i_prot in range(calculation_time.shape[0]):
    time_to_stable = []
    for i_sim in range(calculation_time.shape[1]):
        if np.sum(simulation_stable[i_prot, i_sim,:]==False)==n_beats:
            # pass
            time_to_stable.append(np.nan)
        else:
            time_to_stable.append(np.sum(simulation_stable[i_prot, i_sim,:]==False))

    calculation_time_filtered.append(time_to_stable)
calculation_time_filtered = np.array(calculation_time_filtered)
# ax = plt.subplot(m,n,3)
# plt.boxplot(calculation_time_filtered, whis=[2.5, 97.5], showfliers=False)
# plt.xticks(range(1, len(protocol_names)+1), ['' for _ in range(len(protocol_names))], rotation=45)
# plt.xlim(0, len(protocol_names)+1)
# plt.ylabel('Number of beats \n till stabiltiy [-]', fontsize=10, fontweight='bold')
# # plt.yscale('log')
# plt.grid(True, which='major', axis='y', c=[0.9,0.9,0.9])
# yl = plt.ylim()
# plt.ylim([np.min([9, yl[0]]), None])
# adjust_spines(ax, ['left','bottom'], outward=0)


ax = plt.subplot(m,n,4)
plt.bar(range(len(list_of_protocols)), np.sum(simulation_stable[:,:,-1]==False, axis=1)/simulation_succes.shape[1]*100, label='Not converged')
plt.bar(range(len(list_of_protocols)), np.sum(simulation_succes[:,:,-1]==False, axis=1)/simulation_succes.shape[1]*100, label='Numerical instability')
plt.bar(range(len(list_of_protocols)), np.sum(simulation_high_pressure[:,:,-1]==True, axis=1)/simulation_high_pressure.shape[1]*100, label='Pressure too high')
plt.ylabel('Unstable \n simulations [%]', fontsize=10, fontweight='bold')
plt.xticks(range(len(protocol_names)), protocol_names, rotation=45, fontsize=10, fontweight='bold')
plt.xlim(-1, len(protocol_names))
plt.legend()
plt.grid(True, which='major', axis='y', c=[0.9,0.9,0.9])
adjust_spines(ax, ['left','bottom'], outward=0)
plt.ylim([0,5])
plt.yticks([0,25,50,75,100])


################## Computation time per beat
ax_comp_time_beat = fig.add_subplot(gs[0:2, :2])
boxplot_group_by_solver(ax_comp_time_beat, calculation_time, n_groups, idx_prot_groups, prot_groups, y_log=True)

ax_beats_to_stability = fig.add_subplot(gs[3:5, :2])
boxplot_group_by_solver(ax_beats_to_stability, calculation_time_filtered, n_groups, idx_prot_groups, prot_groups, y_log=False)




plt.tight_layout()


#%%
i_fig = 0
fig = plt.figure(2 + i_fig, figsize=(10,6), clear=True)

subplots_adjust = {'top': 0.85,
                   'bottom': 0.085,
                   'left': 0.16,
                   'right': 0.85,
                   'hspace': 0.3,
                   'wspace': 0.4,
                   }
fig.subplots_adjust(**subplots_adjust)

include_protocols = np.sum(simulation_succes[:,:,-1]==False, axis=1)<simulation_succes.shape[1]
include_protocols = np.argwhere(include_protocols)[:,0]

include_protocols = all_dt == np.min(all_dt)
include_protocols = np.argwhere(include_protocols)[:,0]

m=len(include_protocols)
n=len(include_protocols)

axes = [[], []]
max_diff = [0, 0]

i_dats = [2, 0]


# CORNERS
ax = plt.subplot(m,n,1)
ax.spines['left'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.set_xticks([])
ax.set_yticks([])
plt.annotate('B: ' + protocol_names[include_protocols[0]].split('\n')[0].replace('_', '\n'),
             xycoords = 'axes fraction',
             xy = (-1.2, 0.5),
             fontsize=9,
             rotation=0,
             ha='center', va='center',
             )
plt.annotate('A: ' + protocol_names[include_protocols[0]].split('\n')[0].replace('_', '\n'),
             xycoords = 'axes fraction',
             xy = (0.5, 1.7),
             fontsize=9,
             rotation=0,
             ha='center', va='top',
             )

ax = plt.subplot(m,n,m*n)
ax.spines['left'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.set_xticks([])
ax.set_yticks([])
plt.annotate('B: ' + protocol_names[include_protocols[-1]].split('\n')[0].replace('_', '\n'),
             xycoords = 'axes fraction',
             xy = (2.3, 0.5),
             fontsize=9,
             rotation=0,
             ha='center', va='center',
             )

##### Plots


for i_protocol1a in range(len(include_protocols)):
    for i_protocol0a in range(i_protocol1a+1, len(include_protocols)):

        i_protocol1 = include_protocols[i_protocol1a]
        i_protocol0 = include_protocols[i_protocol0a]

        for i, i_dat in enumerate(i_dats):


            if i==1:
                ax = plt.subplot(m,n,1 + i_protocol1a + (i_protocol0a) * n)
            else:
                # Right Top
                ax = plt.subplot(m,n,1 + i_protocol0a+ (i_protocol1a ) * n)
            scatX = simulation_output[i_protocol0, :, -1, i_dat]
            scatY = simulation_output[i_protocol1, :, -1, i_dat]



            axes[i].append(ax)
            # adjust_spines(ax, ['left', 'bottom'])

            xl = np.array([np.nanmin(0.5*(scatX+scatY)), np.nanmax(0.5*(scatX+scatY))])

            ax.set_xlim(xl)
            ax.scatter(0.5*(scatX+scatY), (scatX-scatY), s=1)
            # ax.set_title('STD: {:.2f}'.format(np.nanstd((scatX-scatY))))
            ax.plot([0, xl[1]], [0,0], c='k', linewidth=1)

            # plt.yscale('log')
            md = np.nanmax(np.abs( scatX-scatY ))
            if md > max_diff[i]:
                max_diff[i] = md

            xticks = ax.get_xticks()
            yticks = ax.get_yticks()
            if i==1:
                ax.spines['right'].set_color('none')
                ax.spines['top'].set_color('none')
                if i_protocol0a==m-1:
                    plt.xlabel('1/2(A+B)')
                    ax.set_xticks(xticks, None)
                else:
                    ax.set_xticks(xticks, ['' for _ in xticks])


                if i_protocol1a==0:
                    plt.ylabel('A-B')
                    plt.annotate('B: ' + protocol_names[i_protocol0].split('\n')[0].replace('_', '\n'),
                                 xycoords = 'axes fraction',
                                 xy = (-1.2, 0.5),
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
                if i_protocol1a==0:
                    # plt.title('A: '+ protocol_names[i_protocol0].split('\n')[0].replace('_', '\n'),
                    #           fontsize=9)
                    plt.annotate('A: ' + protocol_names[i_protocol0].split('\n')[0].replace('_', '\n'),
                                 xycoords = 'axes fraction',
                                 xy = (0.5, 1.7),
                                 fontsize=9,
                                 rotation=0,
                                 ha='center', va='top',
                                 )

                if i_protocol1a==i_protocol0a-1:
                    # plt.xlabel('1/2(A+B)')
                    ax.set_xticks(xticks, None)
                else:
                    xticks = ax.get_xticks()
                    ax.set_xticks(xticks, ['' for _ in xticks])

                if i_protocol0a==m-1:
                    plt.annotate('B: ' + protocol_names[i_protocol1].split('\n')[0].replace('_', '\n'),
                                 xycoords = 'axes fraction',
                                 xy = (2.3, 0.5),
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
        md*= options[np.argmax(md*options > max_diff[i])]

        ax.set_yticks([-md, 0, md])


plt.annotate(output_names[i_dats[0]],
              xycoords = 'figure fraction',
              xy = (0.5, 0.95),
              fontsize=14,
              weight='bold',
              ha='center', va='center',
              )

plt.annotate(output_names[i_dats[1]],
              xycoords = 'figure fraction',
              xy = (0.02, 0.5),
              rotation=90,
              fontsize=14,
              weight='bold',
              ha='center', va='center',
             )

plt.tight_layout()

# %%
# STOP

#%% plot output
plt.figure(10, clear=True)

plot_indices = [7, 8]
plot_indices = [3, 4, 5, 6]
# plot_indices = [0]

prot_include = [0, 1, 2, 3, 4]
prot_include = [0, 1, 2, 3]

m = len(plot_indices)+1
n = len(prot_include)
for i, i_prot in enumerate(prot_include):
    for i_dat in range(m-1):
        plt.subplot(m, n, i_dat*n+i+1)
        idx_plot_sims = simulation_stable[i_prot,:,-1]==False
        idx_plot_sims[:]=True
        plt.plot(simulation_output[i_prot, idx_plot_sims, :, plot_indices[i_dat]].T)

        if i_dat==0:
            plt.title(protocol_names[i_prot], fontsize=14, fontweight='bold')
        if i==0:
            plt.ylabel(output_names[plot_indices[i_dat]], fontsize=14, fontweight='bold')

        # plt.ylim(output_ylim[i_dat])
    # crashed simulations
    plt.subplot(m, n, (i_dat+1)*n+i_prot+1)
    plt.plot(100-np.sum(simulation_succes[i_prot, :, :], axis=0)/simulation_output.shape[1]*100)
    if i_prot==0:
        plt.ylabel('Crashed [%]', fontsize=14, fontweight='bold')

#%%



# True values: mean of lowest dt's
semi_true_values = np.mean(simulation_output[idx_prot,:,-1,:], axis=0)

plt.figure(11, figsize=(10,6), clear=True)
m=6
n=5

idx_plot_indices = range(0, semi_true_values.shape[1])

for i_plot in range(len(idx_plot_indices)):
    i_data = idx_plot_indices[i_plot]
    ax = plt.subplot(m,n,i_plot+1)
    for i_prot in range(n_groups):
        idx_stable = np.argmax(simulation_stable, axis=2)
        idx_stable[idx_stable==0]=-1


        output_of_interest = np.ndarray(simulation_output.shape[:2])
        for i_prot1 in range(simulation_output.shape[0]):
            for i_sim in range(simulation_output.shape[1]):
                output_of_interest[i_prot1, i_sim] = simulation_output[i_prot1,i_sim,-1,i_data]
                # remove not converged simluatoins
                if idx_stable[i_prot1, i_sim]==0:
                    output_of_interest[i_prot1, i_sim] = np.nan

        # Select protocol
        data = np.array([ np.abs( semi_true_values[:,i_data]/output_of_interest[i_prot1,:]-1)
                for i_prot1 in np.argwhere(idx_prot_groups==i_prot).reshape(-1) ])
        idx_stable = idx_stable[idx_prot_groups==i_prot]



        data_percentile = np.nanpercentile(data, [2.5, 25, 50, 75, 97.5], axis=1)
        plt.fill_between(all_dt[idx_prot_groups==i_prot],
                          data_percentile[0,:],
                          data_percentile[-1,:],
                          fc=plot_colors[i_prot],
                          alpha=0.1)
        plt.fill_between(all_dt[idx_prot_groups==i_prot],
                          data_percentile[1,:],
                          data_percentile[3,:],
                          fc=plot_colors[i_prot],
                          alpha=0.6)
        plt.plot(all_dt[idx_prot_groups==i_prot], data_percentile[2,:],
                 c=plot_colors[i_prot], label=prot_groups[i_prot])
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='major', axis='y')
    plt.title(output_names[i_data], fontweight='bold')
    plt.xlabel(r'Solver time stepsize $\Delta$t [ms]')
    plt.ylabel(r'$L^2$ ' + output_names[i_data])
    adjust_spines(ax, ['left','bottom'], outward=0)
    plt.legend(loc='upper left')
plt.tight_layout()




#%% Plot with varying treshold
idx_Errq = 5
idx_ErrSy = 6

include_prot = np.array([1, 2, 3, 5])
include_prot = np.array([0, 1])
# include_prot = np.array([6, 7, 8]) # only CA
flowerror_thresh = 10**np.linspace(-5, 1, 17)

if PFC_on:
    FlowError = np.sqrt(1e6*simulation_output[:,:,:,idx_Errq]**2 + (simulation_output[:,:,:,idx_ErrSy]-1)**2 )
else:
    FlowError = 1e3*np.abs(simulation_output[:,:,:,idx_Errq])

plt.figure(13, figsize=(10, 6), clear=True)
h=len(include_prot)
v=2
all_idx_thresh_reached = []
for i_prot in range(len(include_prot)):
    plt.subplot(v, h, i_prot+1)
    plt.plot(FlowError[include_prot[i_prot],:,: ].T)

    plt.subplot(v,h,i_prot+1+1*v)

    idx_thresh_reached = []
    for i_flowerror_thresh in range(len(flowerror_thresh)):
        aux = FlowError[include_prot[i_prot],:,: ]<flowerror_thresh[i_flowerror_thresh]
        aux[:,0] = False # The first beat is never stable
        aux = np.argmax(aux, axis=1).astype(float)
        aux[aux==0] = np.nan # FlowError.shape[2] + 1
        idx_thresh_reached.append(aux)
    idx_thresh_reached = np.array(idx_thresh_reached)
    plt.plot(flowerror_thresh, idx_thresh_reached)
    all_idx_thresh_reached.append(idx_thresh_reached)
    plt.xscale('log')
all_idx_thresh_reached=np.array(all_idx_thresh_reached)

# plot
h=4
v=4
plt.figure(14, figsize=(10, 6), clear=True)
for i_plot in range(len(idx_plot_indices)):
    if i_plot>=v*h-1:
        break
    ax = plt.subplot(v,h,i_plot+1)

    output_of_interest = np.ndarray((all_idx_thresh_reached.shape[1], simulation_output.shape[1]))
    i_data = idx_plot_indices[i_plot]
    for i_prot in range(len(include_prot)):
        idx_stable = np.argmax(simulation_stable, axis=2)
        idx_stable[idx_stable==0]=-1


        for i_thresh in range(all_idx_thresh_reached.shape[1]):
            for i_sim in range(simulation_output.shape[1]):
                aux = simulation_output[include_prot[i_prot],i_sim,:,i_data]
                aux[np.isnan(aux)] = -1
                if np.isnan(all_idx_thresh_reached[i_prot,i_thresh, i_sim]):
                    output_of_interest[i_thresh, i_sim] = np.nan
                else:
                    output_of_interest[i_thresh, i_sim] = aux[all_idx_thresh_reached[i_prot,i_thresh, i_sim].astype(int) ]
                # remove not converged simluatoins
                if idx_stable[i_prot1, i_sim]==0:
                    output_of_interest[i_prot, i_sim] = np.nan

        # Select protocol
        data = np.abs( semi_true_values[:,i_data] / output_of_interest[:,:] - 1)
        idx_stable = idx_stable[idx_prot_groups==i_prot]


        data_percentile = np.nanpercentile(data, [2.5, 25, 50, 75, 97.5], axis=1)
        plt.fill_between(flowerror_thresh,
                          data_percentile[0,:],
                          data_percentile[-1,:],
                          fc=plot_colors[i_prot],
                          alpha=0.1)
        plt.fill_between(flowerror_thresh,
                          data_percentile[1,:],
                          data_percentile[3,:],
                          fc=plot_colors[i_prot],
                          alpha=0.6)
        plt.plot(flowerror_thresh, data_percentile[2,:],
                 c=plot_colors[i_prot], label=list_of_protocols[include_prot[i_prot]]['legend'])
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='major', axis='y')
    plt.title(output_names[i_data], fontweight='bold')
    plt.xlabel(r'Steady-state threshold [-]')
    plt.ylabel(r'$L^2$ ' + output_names[i_data])
    adjust_spines(ax, ['left','bottom'], outward=0)
    plt.legend(loc='upper right')
    ax.invert_xaxis()




# plot computational time threshold
h=4
v=4
plt.figure(15, figsize=(10, 6), clear=True)
for i_plot in range(1):
    if i_plot>=v*h-1:
        break
    ax = plt.subplot(v,h,i_plot+1)

    output_of_interest = np.ndarray((all_idx_thresh_reached.shape[1], simulation_output.shape[1]))
    i_data = idx_plot_indices[i_plot]
    for i_prot in range(len(include_prot)):
        idx_stable = np.argmax(simulation_stable, axis=2)
        idx_stable[idx_stable==0]=-1


        data = np.ndarray(output_of_interest.shape)

        # for i_thresh in range(all_idx_thresh_reached.shape[1]):
        #     for i_sim in range(simulation_output.shape[1]):
        #         aux = simulation_output[include_prot[i_prot],i_sim,:,i_data]
        #         aux[np.isnan(aux)] = -1
        #         if np.isnan(all_idx_thresh_reached[i_prot,i_thresh, i_sim]):
        #             output_of_interest[i_thresh, i_sim] = np.nan
        #         else:
        #             output_of_interest[i_thresh, i_sim] = aux[all_idx_thresh_reached[i_prot,i_thresh, i_sim].astype(int) ]
        #         # remove not converged simluatoins
        #         if idx_stable[i_prot1, i_sim]==0:
        #             output_of_interest[i_prot, i_sim] = np.nan

        # # Select protocol
        # data = np.abs( semi_true_values[:,i_data] - output_of_interest[:,:])

        for i_thresh in range(all_idx_thresh_reached.shape[1]):
            for i_sim in range(simulation_output.shape[1]):
                if np.isnan(all_idx_thresh_reached[i_prot,i_thresh, i_sim]):
                    data[i_thresh, i_sim] = np.nan
                else:
                    data[i_thresh, i_sim] = np.sum(calculation_time[include_prot[i_prot], i_sim, :all_idx_thresh_reached[i_prot,i_thresh, i_sim].astype(int)])




        idx_stable = idx_stable[idx_prot_groups==i_prot]


        data_percentile = np.nanpercentile(data, [2.5, 25, 50, 75, 97.5], axis=1)
        plt.fill_between(flowerror_thresh,
                          data_percentile[0,:],
                          data_percentile[-1,:],
                          fc=plot_colors[i_prot],
                          alpha=0.1)
        plt.fill_between(flowerror_thresh,
                          data_percentile[1,:],
                          data_percentile[3,:],
                          fc=plot_colors[i_prot],
                          alpha=0.6)
        plt.plot(flowerror_thresh, data_percentile[2,:],
                 c=plot_colors[i_prot], label=list_of_protocols[include_prot[i_prot]]['legend'])
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='major', axis='y')
    plt.title('Calculation time', fontweight='bold')
    plt.xlabel(r'Steady-state threshold [-]')
    plt.ylabel(r'Calculation Time [ms] ')
    adjust_spines(ax, ['left','bottom'], outward=0)
    plt.legend(loc='lower right')
    ax.invert_xaxis()


#%% grouped indices

# get simulations with lowest dt
lowest_dt = 1
idx_prot = []
all_solvers = []
all_dt = []
all_solver_names = []
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

# True values: mean of lowest dt's
semi_true_values = np.mean(simulation_output[idx_prot,:,-1,:], axis=0)

plt.figure(18, figsize=(10,6), clear=True)
m=3
n=5

idx_plot_indices = [[0], [1], [2], [3], [], [4], [5], [6], [7], [8],]


for i_plot in range(len(idx_plot_indices)):
    i_data = idx_plot_indices[i_plot]

    if i_data == []:
        continue

    ax = plt.subplot(m,n,i_plot+1)
    for i_prot in range(n_groups):
        idx_stable = np.argmax(simulation_stable, axis=2)
        idx_stable[idx_stable==0]=-1



        output_of_interest_all = []
        for ii_data in i_data:
            output_of_interest = np.ndarray(simulation_output.shape[:2])
            for i_prot1 in range(simulation_output.shape[0]):
                for i_sim in range(simulation_output.shape[1]):
                    output_of_interest[i_prot1, i_sim] = simulation_output[i_prot1,i_sim,-1,ii_data]
                    # remove not converged simluatoins
                    if idx_stable[i_prot1, i_sim]==0:
                        output_of_interest[i_prot1, i_sim] = np.nan
            output_of_interest_all.append(output_of_interest)
        # Select protocol

        data = []
        for i, ii_data in enumerate(i_data):
            data1 = np.array([ np.abs( semi_true_values[:,ii_data]/output_of_interest_all[i][i_prot1,:]-1)
                for i_prot1 in np.argwhere(idx_prot_groups==i_prot).reshape(-1) ])
            data.append(data1)
        data = np.concatenate(data, axis=1)



        idx_stable = idx_stable[idx_prot_groups==i_prot]



        data_percentile = np.nanpercentile(data, [2.5, 25, 50, 75, 97.5], axis=1)
        plt.fill_between(all_dt[idx_prot_groups==i_prot],
                          data_percentile[0,:],
                          data_percentile[-1,:],
                          fc=plot_colors[i_prot],
                          alpha=0.1)
        plt.fill_between(all_dt[idx_prot_groups==i_prot],
                          data_percentile[1,:],
                          data_percentile[3,:],
                          fc=plot_colors[i_prot],
                          alpha=0.6)
        plt.plot(all_dt[idx_prot_groups==i_prot], data_percentile[2,:],
                 c=plot_colors[i_prot], label=prot_groups[i_prot])
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='major', axis='y')
    if len(i_data)==1:
        plt.title(output_names[i_data[0]], fontweight='bold')
    else:
        plt.title(output_names[i_data[0]][:-3], fontweight='bold')
    plt.xlabel(r'Solver time stepsize $\Delta$t [ms]')
    plt.ylabel(r'$L^2$ ' + output_names[i_data[0]])
    adjust_spines(ax, ['left','bottom'], outward=0)
    plt.legend(loc='upper left')
plt.tight_layout()






# %% Nice plot, assume last beat is stable
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

# True values: mean of lowest dt's from last iteration
semi_true_values = np.mean(simulation_output[idx_prot, :, -1, :], axis=0)





fig = plt.figure(98, clear=True, figsize=(12, 6))
gs = fig.add_gridspec(3, 5)

subplots_adjust = {'top': 0.95,
                   'bottom': 0.085,
                   'left': 0.066,
                   'right': 0.988,
                   'hspace': 0.05,
                   'wspace': 0.5,
                   }
fig.subplots_adjust(**subplots_adjust)

axLeft = fig.add_subplot(gs[:, :2])

## Calculation time plot
for i_prot in range(n_groups):
    idx = idx_prot_groups == i_prot

    plot_ranges = np.nanpercentile(np.sum(calculation_time[idx, :, :], axis=2)*1e-3, [2.5, 25, 50, 75, 97.5], axis=1)
    axLeft.plot(all_dt[idx], plot_ranges[2, :],
                color=colors[prot_groups[i_prot]], zorder=9)

    axLeft.fill_between(all_dt[idx], plot_ranges[0, :], plot_ranges[4, :],
                        color=colors[prot_groups[i_prot]], zorder=7, alpha=0.1)
    axLeft.fill_between(all_dt[idx], plot_ranges[1, :], plot_ranges[3, :],
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
axLeft.annotate('Calculation time of '+str(calculation_time.shape[2])+' beats',
                (left_location, 0.975),
                xycoords='figure fraction',
                horizontalalignment='center', verticalalignment='center',
                fontsize=fontsettings['title_fontsize'],
                fontweight=fontsettings['title_fontweight'])



axLeftCrash = axLeft.twinx()
for i_prot in range(n_groups):
    idx = idx_prot_groups == i_prot
    if PFC_on:
        axLeftCrash.plot(all_dt[idx],
                         np.sum(simulation_stable[idx, :, -1]==False, axis=1),
                         '--',
                         # color=colors[prot_groups[i_prot]],
                         )
    axLeftCrash.plot(all_dt[idx],
                     np.sum(simulation_succes[idx, :, -1]==False, axis=1)/simulation_succes.shape[1]*100,
                     '-.',
                     # color=colors[prot_groups[i_prot]],
                     )
axLeftCrash.spines['top'].set_visible(False)
axLeftCrash.spines['right'].set_visible(True)
axLeftCrash.set_ylim(-1, 101)
axLeftCrash.set_ylabel('Numerical unstable simulations [%]',
                       fontsize=fontsettings['label_fontsize'])



#### error plots
axErrors = [fig.add_subplot(gs[0, 3]),
            fig.add_subplot(gs[0, 4]),
            fig.add_subplot(gs[1, 3]),
            fig.add_subplot(gs[1, 4]),
            fig.add_subplot(gs[2, 3]),
            fig.add_subplot(gs[2, 4]),
            ]
output_include = [8, 9, 0, 1, 2, 3]
error_ylabels = ['MAP [-]',
                 'Venous Return [-]',
                 'EDV [-]',
                 'ESV [-]',
                 'max LV pressure [-]',
                 'min LV pressure [-]',
                 ]

for i_plot in range(len(axErrors)):
    i_data = output_include[i_plot]
    for i_prot in range(n_groups):
        idx = idx_prot_groups == i_prot
        data = np.abs(semi_true_values[:, i_data] /
                      simulation_output[idx, :, -1, i_data] - 1)

        # remove unstable datapoints
        idx_is_stable = simulation_stable[idx,:,-1]
        data[idx_is_stable==False] = np.nan

        data_percentile = np.nanpercentile(data, [2.5, 25, 50, 75, 97.5],
                                           axis=1)

        axErrors[i_plot].plot(all_dt[idx], data_percentile[2, :],
                              color=colors[prot_groups[i_prot]])
        axErrors[i_plot].fill_between(all_dt[idx], data_percentile[0, :], data_percentile[4, :],
                                      color=colors[prot_groups[i_prot]],
                                      zorder=7, alpha=0.1)
        axErrors[i_plot].fill_between(all_dt[idx], data_percentile[1, :], data_percentile[3, :],
                                      color=colors[prot_groups[i_prot]],
                                      zorder=8, alpha=0.1)

    axErrors[i_plot].set_ylim([2e-6, 9e-1])
    axErrors[i_plot].set_xscale('log')
    axErrors[i_plot].set_yscale('log')
    axErrors[i_plot].set_ylabel(error_ylabels[i_plot],
                                fontsize=fontsettings['label_fontsize'])
    axErrors[i_plot].spines['top'].set_visible(False)
    axErrors[i_plot].spines['right'].set_visible(False)


# axErrors[3].set_xlabel('Integration step size $\\Delta t$ [s]',
#                   fontsize=fontsettings['label_fontsize'])
axErrors[4].set_xlabel('Integration step size $\\Delta t$ [s]',
                  fontsize=fontsettings['label_fontsize'])



###### Plot stability after 50 beats
if False:
    if PFC_on:
        axErrors = [fig.add_subplot(gs[2, 3]),
                    fig.add_subplot(gs[1, 3]),
                    ]
        output_include = [5, 6]
        error_ylabels = ['errQ [-]',
                         'errSy [-]',
                         ]
    else:
        axErrors = [
                    fig.add_subplot(gs[2, 2]),
                    ]
        output_include = [5]
        error_ylabels = ['errQ [-]',
                         ]

    for i_plot in range(len(axErrors)):
        i_data = output_include[i_plot]
        for i_prot in range(n_groups):
            idx = idx_prot_groups == i_prot
            data = simulation_output[idx, :, -1, i_data]
            if i_data==6:
                data -= 1 # correction for errSy
            data_percentile = np.nanpercentile(data, [2.5, 25, 50, 75, 97.5],
                                               axis=1)

            axErrors[i_plot].plot(all_dt[idx], data_percentile[2, :],
                                  color=colors[prot_groups[i_prot]])
            axErrors[i_plot].fill_between(all_dt[idx], data_percentile[0, :], data_percentile[4, :],
                                          color=colors[prot_groups[i_prot]],
                                          zorder=7, alpha=0.1)
            axErrors[i_plot].fill_between(all_dt[idx], data_percentile[1, :], data_percentile[3, :],
                                          color=colors[prot_groups[i_prot]],
                                          zorder=8, alpha=0.1)

        # axErrors[i_plot].set_ylim([2e-6, 9e-1])
        axErrors[i_plot].set_xscale('log')
        axErrors[i_plot].set_yscale('log')
        axErrors[i_plot].set_ylabel(error_ylabels[i_plot],
                                    fontsize=fontsettings['label_fontsize'])
        axErrors[i_plot].spines['top'].set_visible(False)
        axErrors[i_plot].spines['right'].set_visible(False)


# axErrors[3].set_xlabel('Integration step size $\\Delta t$ [s]',
#                   fontsize=fontsettings['label_fontsize'])
# axErrors[4].set_xlabel('Integration step size $\\Delta t$ [s]',
#                   fontsize=fontsettings['label_fontsize'])




#### Legend
axLegend = fig.add_subplot(gs[0, 2])

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
axLegend.set_xlim([0, 4])
axLegend.set_ylim([-2, 2 + n_groups])
axLegend.spines['top'].set_visible(False)
axLegend.spines['right'].set_visible(False)
axLegend.spines['bottom'].set_visible(False)
axLegend.spines['left'].set_visible(False)

left_location = subplots_adjust['left'] + 0.8*(
    subplots_adjust['right']-subplots_adjust['left'])

axLegend.annotate('Integration error (relative)', (left_location, 0.975),
                xycoords='figure fraction',
                horizontalalignment='center', verticalalignment='center',
                fontsize=fontsettings['title_fontsize'],
                fontweight=fontsettings['title_fontweight'])



axLegend1 = fig.add_subplot(gs[1, 2])
axLegend1.plot([0, 1], [2, 2], 'k--')
axLegend1.plot([0, 1], [1, 1], 'k-.')
axLegend1.annotate('numerical\nunstable', (1.1, 1),
                  horizontalalignment='left', verticalalignment='center',
                  fontsize=fontsettings['legend_fontsize'])
if PFC_on:
    axLegend1.annotate('unsuccess', (1.1, 2),
                  horizontalalignment='left', verticalalignment='center',
                  fontsize=fontsettings['legend_fontsize'])

axLegend1.set_xticks([])
axLegend1.set_yticks([])
axLegend1.set_xlim([0, 4])
axLegend1.set_ylim([-1, 2.5])
axLegend1.spines['top'].set_visible(False)
axLegend1.spines['right'].set_visible(False)
axLegend1.spines['bottom'].set_visible(False)
axLegend1.spines['left'].set_visible(False)


# %% nice plot PFC thresh
fig = plt.figure(99, clear=True, figsize=(12, 6))
gs = fig.add_gridspec(3, 5)
subplots_adjust = {'top': 0.95,
                   'bottom': 0.085,
                   'left': 0.066,
                   'right': 0.988,
                   'hspace': 0.05,
                   'wspace': 0.4,
                   }
fig.subplots_adjust(**subplots_adjust)


color_fill = [[0.9, 0.1, 0.1], [0.4, 0.7, 0.4], [0.5, 1, 0.5], [0.2, 0.3, 1]]
idx_Errq = 5
idx_ErrSy = 6

# include_prot = np.array([1, 2, 3, 5, 6, 7])
# include_prot = np.array([6, 7, 8, 9, 10]) # only CA
# include_prot = np.array([4, 5, 6, 7]) # only CA
# include_prot = np.array([0, 5, 10, 15, 20]) # only CA
# include_prot = np.array([0, 5, 10, 20]) # only CA
# include_prot = np.argwhere(all_dt==2e-3)[:, 0]
include_prot = np.array([0, 1, 2])

flowerror_thresh = 10**np.linspace(-4, 3, 89)

if PFC_on:
    FlowError = np.sqrt(1e6*simulation_output[:,:,:,idx_ErrSy]**2 + (simulation_output[:,:,:,idx_Errq]-1)**2 )
    plot_opt_thresh = 1e-2
else:
    FlowError = 1e3*np.abs(simulation_output[:,:,:,idx_ErrSy])
    plot_opt_thresh = 1e-1

# calculate data
all_idx_thresh_reached = []
for i_prot in range(len(include_prot)):
    idx_thresh_reached = []
    for i_flowerror_thresh in range(len(flowerror_thresh)):
        aux = FlowError[include_prot[i_prot],:,: ]<flowerror_thresh[i_flowerror_thresh]
        aux[:,0] = False # The first beat is never stable
        aux = np.argmax(aux, axis=1).astype(float)
        aux[aux==0] = np.nan # FlowError.shape[2] + 1
        idx_thresh_reached.append(aux)
    idx_thresh_reached = np.array(idx_thresh_reached)
    all_idx_thresh_reached.append(idx_thresh_reached)
all_idx_thresh_reached=np.array(all_idx_thresh_reached)


# plot indices
axPlotIndex = [fig.add_subplot(gs[0, 3]),
               fig.add_subplot(gs[0, 4]),
               fig.add_subplot(gs[1, 3]),
               fig.add_subplot(gs[1, 4]),
               fig.add_subplot(gs[2, 3]),
               fig.add_subplot(gs[2, 4]),
               ]
idx_plot_indices = [8, 9, 0, 1, 2, 3]
error_ylabels = ['MAP [-]',
                 'Venous Return [-]',
                 'EDV [-]',
                 'ESV [-]',
                 'max LV pressure [-]',
                 'min LV pressure [-]',
                 ]
error_ylabels = np.array(output_names)[idx_plot_indices]



for i_plot in range(len(idx_plot_indices)):
    output_of_interest = np.ndarray((all_idx_thresh_reached.shape[1], simulation_output.shape[1]))
    i_data = idx_plot_indices[i_plot]
    for i_prot in range(len(include_prot)):
        idx_stable = np.argmax(simulation_stable, axis=2)
        idx_stable[idx_stable==0]=-1


        for i_thresh in range(all_idx_thresh_reached.shape[1]):
            for i_sim in range(simulation_output.shape[1]):
                aux = simulation_output[include_prot[i_prot],i_sim,:,i_data]
                aux[np.isnan(aux)] = -1
                if (np.isnan(all_idx_thresh_reached[i_prot,i_thresh, i_sim]) or
                    FlowError[i_prot, i_sim, -1] > flowerror_thresh[i_thresh]):
                    output_of_interest[i_thresh, i_sim] = np.nan
                else:
                    output_of_interest[i_thresh, i_sim] = aux[all_idx_thresh_reached[i_prot,i_thresh, i_sim].astype(int) ]
                # remove not converged simluatoins
                if idx_stable[i_prot1, i_sim]==0:
                    output_of_interest[i_prot, i_sim] = np.nan

        # Select protocol
        data = np.abs( semi_true_values[:,i_data] - output_of_interest[:,:])
        if np.all(np.isnan(data)):
            continue
        idx_stable = idx_stable[idx_prot_groups==i_prot]


        data_percentile = np.nanpercentile(data, [5, 25, 50, 75, 95], axis=1)
        axPlotIndex[i_plot].fill_between(flowerror_thresh,
                          data_percentile[0, :],
                          data_percentile[-1, :],
                          fc = color_fill[i_prot],
                          alpha = 0.1)
        axPlotIndex[i_plot].fill_between(flowerror_thresh,
                                         data_percentile[1, :],
                                         data_percentile[3, :],
                                         fc=color_fill[i_prot],
                                         alpha=0.6)
        axPlotIndex[i_plot].plot(flowerror_thresh, data_percentile[2,:],
                 c=color_fill[i_prot], label=list_of_protocols[include_prot[i_prot]]['legend'])
    axPlotIndex[i_plot].set_xscale('log')
    axPlotIndex[i_plot].set_yscale('log')
    axPlotIndex[i_plot].grid(True, which='major', axis='y')
    # axPlotIndex[i_plot].set_title(output_names[i_data], fontweight='bold')
    if i_plot > 3:
        axPlotIndex[i_plot].set_xlabel(r'Steady-state threshold [-]')
    axPlotIndex[i_plot].set_ylabel(error_ylabels[i_plot],
                                   fontsize=fontsettings['label_fontsize'])
    adjust_spines(axPlotIndex[i_plot], ['left','bottom'], outward=0)
    # plt.legend(loc='upper right')
    axPlotIndex[i_plot].invert_xaxis()

    axPlotIndex[i_plot].axvline(plot_opt_thresh, c='k', ls='--', lw=1)



# plot computational time threshold


for i_plot in range(1):
    if i_plot>=v*h-1:
        break
    ax = fig.add_subplot(gs[:3, :2])

    output_of_interest = np.ndarray((all_idx_thresh_reached.shape[1], simulation_output.shape[1]))

    number_of_succes = np.zeros((len(include_prot), all_idx_thresh_reached.shape[1]),
                                dtype=float)



    i_data = idx_plot_indices[i_plot]
    for i_prot in range(len(include_prot)):
        idx_stable = np.argmax(simulation_stable, axis=2)
        idx_stable[idx_stable==0]=-1


        data = np.ndarray(output_of_interest.shape)

        # for i_thresh in range(all_idx_thresh_reached.shape[1]):
        #     for i_sim in range(simulation_output.shape[1]):
        #         aux = simulation_output[include_prot[i_prot],i_sim,:,i_data]
        #         aux[np.isnan(aux)] = -1
        #         if np.isnan(all_idx_thresh_reached[i_prot,i_thresh, i_sim]):
        #             output_of_interest[i_thresh, i_sim] = np.nan
        #         else:
        #             output_of_interest[i_thresh, i_sim] = aux[all_idx_thresh_reached[i_prot,i_thresh, i_sim].astype(int) ]
        #         # remove not converged simluatoins
        #         if idx_stable[i_prot1, i_sim]==0:
        #             output_of_interest[i_prot, i_sim] = np.nan

        # # Select protocol
        # data = np.abs( semi_true_values[:,i_data] - output_of_interest[:,:])

        for i_thresh in range(all_idx_thresh_reached.shape[1]):
            for i_sim in range(simulation_output.shape[1]):
                # print(FlowError[i_prot, i_sim, -1], flowerror_thresh[i_thresh])
                # if (np.isnan(all_idx_thresh_reached[i_prot,i_thresh, i_sim]) or
                #     FlowError[i_prot, i_sim, -1] > flowerror_thresh[i_thresh]
                #     ):
                #     data[i_thresh, i_sim] = np.nan
                # else:
                #     data[i_thresh, i_sim] = np.sum(calculation_time[include_prot[i_prot], i_sim, :all_idx_thresh_reached[i_prot,i_thresh, i_sim].astype(int)])
                #     number_of_succes[i_prot, i_thresh] += 1

                # np.sum(FlowError[0, :, [-1]] > flowerror_thresh.reshape((-1, 1)), axis=1).shape
                if FlowError[include_prot[i_prot], i_sim, -1] < flowerror_thresh[i_thresh]:
                    # print('a', FlowError[i_prot, i_sim, -1], flowerror_thresh[i_thresh])
                    data[i_thresh, i_sim] = np.sum(calculation_time[include_prot[i_prot], i_sim, :all_idx_thresh_reached[i_prot,i_thresh, i_sim].astype(int)])
                    number_of_succes[i_prot, i_thresh] += 1 * 100/FlowError.shape[1]
                else:
                    print('r', FlowError[i_prot, i_sim, -1], flowerror_thresh[i_thresh])
                    data[i_thresh, i_sim] = np.nan




        # remove non-stable sims
        # data = data[:, idx_stable[i_prot, :]>0]


        # calculate
        idx_stable = idx_stable[idx_prot_groups==i_prot]

        data_percentile = np.nanpercentile(data, [5, 25, 50, 75, 95], axis=1)
        plt.fill_between(flowerror_thresh,
                          data_percentile[0,:],
                          data_percentile[-1,:],
                          fc=color_fill[i_prot],
                          alpha=0.1)
        plt.fill_between(flowerror_thresh,
                          data_percentile[1,:],
                          data_percentile[3,:],
                          fc=color_fill[i_prot],
                          alpha=0.6)
        plt.plot(flowerror_thresh, data_percentile[2,:],
                 c=color_fill[i_prot], label=list_of_protocols[include_prot[i_prot]]['legend'])

        plt.axvline(plot_opt_thresh, c='k', ls='--', lw=1)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which='major', axis='y')
    plt.title('Calculation time', fontweight='bold',
        fontsize=fontsettings['title_fontsize'],
        weight=fontsettings['title_fontweight'],
        )

    plt.xlabel(r'Steady-state threshold [-]')
    plt.ylabel(r'Calculation Time [ms] ')
    adjust_spines(ax, ['left','bottom'], outward=0)
    # plt.legend(loc='lower right')
    ax.invert_xaxis()


    left_location = subplots_adjust['left'] + 0.8*(
    subplots_adjust['right']-subplots_adjust['left'])

    ax.annotate('Pressure flow control error', (left_location, 0.975),
                    xycoords='figure fraction',
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=fontsettings['title_fontsize'],
                    fontweight=fontsettings['title_fontweight'])

    # axSucces = ax.twinx()
    # for i_prot in range(number_of_succes.shape[0]):
    #     axSucces.plot(flowerror_thresh,
    #                   number_of_succes[i_prot,:],
    #                   color=color_fill[i_prot])
    # axSucces.spines['top'].set_visible(False)
    # axSucces.set_ylim([-1, 101])

    # n_runned_simulations = FlowError.shape[2]
    # axSucces.set_ylabel(f'Converged simulations within {n_runned_simulations} simulations [%]',
    #                     fontsize=fontsettings['label_fontsize'])

# Legend
axLegend = fig.add_subplot(gs[:4, 2])
for i_prot in range(len(include_prot)):
    axLegend.plot([0, 1], [i_prot, i_prot],
                  color=color_fill[i_prot])
    axLegend.fill_between([0, 1], [i_prot-0.1, i_prot-0.1], [i_prot+0.1, i_prot+0.1],
                          color=color_fill[i_prot], alpha=0.1)
    axLegend.fill_between([0, 1], [i_prot-0.2, i_prot-0.2], [i_prot+0.2, i_prot+0.2],
                          color=color_fill[i_prot], alpha=0.1)
    axLegend.annotate(list_of_protocols[include_prot[i_prot]]['legend'].replace('_', '\n'), (1.1, i_prot),
                      horizontalalignment='left', verticalalignment='center',
                      fontsize=fontsettings['legend_fontsize'])



axLegend.set_xticks([])
axLegend.set_yticks([])
axLegend.set_xlim([0, 4])
axLegend.set_ylim([-2, len(include_prot)+1])
axLegend.spines['top'].set_visible(False)
axLegend.spines['right'].set_visible(False)
axLegend.spines['bottom'].set_visible(False)
axLegend.spines['left'].set_visible(False)




# %% nice plot PFC thresh
fig = plt.figure(100, clear=True, figsize=(12, 3))
gs = fig.add_gridspec(1, 4)
subplots_adjust = {'top': 0.9,
                   'bottom': 0.2,
                   'left': 0.1,
                   'right': 1,
                   'hspace': 0.05,
                   'wspace': 0.4,
                   }
fig.subplots_adjust(**subplots_adjust)


color_fill = [[0.9, 0.1, 0.1], [0.4, 0.7, 0.4], [0.5, 1, 0.5], [0.2, 0.3, 1]]

# from _functions import colors
# color_fill = [colors[prot] for prot in ]

idx_Errq = 5
idx_ErrSy = 6

# include_prot = np.array([1, 2, 3, 5, 6, 7])
# include_prot = np.array([6, 7, 8, 9, 10]) # only CA
# include_prot = np.array([4, 5, 6, 7]) # only CA
# include_prot = np.array([0, 5, 10, 15, 20]) # only CA
# include_prot = np.array([0, 5, 10, 20]) # only CA
# include_prot = np.argwhere(all_dt==2e-3)[:, 0]
include_prot = np.array([0, 1, 2])

flowerror_thresh = 10**np.linspace(-4, 3, 89)

if PFC_on:
    FlowError = np.sqrt(1e6*simulation_output[:,:,:,idx_ErrSy]**2 + (simulation_output[:,:,:,idx_Errq]-1)**2 )
    plot_opt_thresh = 1e-2
else:
    FlowError = 1e3*np.abs(simulation_output[:,:,:,idx_ErrSy])
    plot_opt_thresh = 1e-1

# calculate data
all_idx_thresh_reached = []
for i_prot in range(len(include_prot)):
    idx_thresh_reached = []
    for i_flowerror_thresh in range(len(flowerror_thresh)):
        aux = FlowError[include_prot[i_prot],:,: ]<flowerror_thresh[i_flowerror_thresh]
        aux[:,0] = False # The first beat is never stable
        aux = np.argmax(aux, axis=1).astype(float)
        aux[aux==0] = np.nan # FlowError.shape[2] + 1
        idx_thresh_reached.append(aux)
    idx_thresh_reached = np.array(idx_thresh_reached)
    all_idx_thresh_reached.append(idx_thresh_reached)
all_idx_thresh_reached=np.array(all_idx_thresh_reached)


# plot indices
axPlotIndex = [fig.add_subplot(gs[0, 1]),
               fig.add_subplot(gs[0, 2]),
               # fig.add_subplot(gs[1, 3]),
               # fig.add_subplot(gs[1, 4]),
               # fig.add_subplot(gs[2, 3]),
               # fig.add_subplot(gs[2, 4]),
               ]
idx_plot_indices = [0, 1]
error_ylabels = ['V [-]',
                 'p [-]',

                 ]
error_ylabels = np.array(output_names)[idx_plot_indices]



for i_plot in range(len(idx_plot_indices)):
    output_of_interest = np.ndarray((all_idx_thresh_reached.shape[1], simulation_output.shape[1]))
    i_data = idx_plot_indices[i_plot]
    for i_prot in range(len(include_prot)):
        idx_stable = np.argmax(simulation_stable, axis=2)
        idx_stable[idx_stable==0]=-1

        data = np.empty((all_idx_thresh_reached.shape[1], simulation_output.shape[1]))
        data[:] = np.nan
        for i_thresh in range(all_idx_thresh_reached.shape[1]):
            for i_sim in range(simulation_output.shape[1]):
                aux = simulation_output[include_prot[i_prot],i_sim,:,i_data]
                aux[np.isnan(aux)] = -1
                if (np.isnan(all_idx_thresh_reached[i_prot,i_thresh, i_sim]) or
                    FlowError[i_prot, i_sim, -1] > flowerror_thresh[i_thresh]):
                    output_of_interest[i_thresh, i_sim] = np.nan
                else:
                    output_of_interest[i_thresh, i_sim] = aux[all_idx_thresh_reached[i_prot,i_thresh, i_sim].astype(int) ]
                # remove not converged simluatoins
                if idx_stable[i_prot, i_sim]==0:
                    output_of_interest[i_prot, i_sim] = np.nan

                if np.isnan(all_idx_thresh_reached[i_prot,i_thresh, i_sim]):
                    continue
                data[i_thresh, i_sim] = signals_mean_abs_error_true[i_prot, i_sim, all_idx_thresh_reached[i_prot,i_thresh, i_sim].astype(int), i_plot]

        # Select protocol
        # data = np.abs( semi_true_values[:,i_data] - output_of_interest[:,:])

        # data = signals_mean_abs_error_true



        if np.all(np.isnan(data)):
            continue
        idx_stable = idx_stable[idx_prot_groups==i_prot]


        data_percentile = np.nanpercentile(data, [5, 25, 50, 75, 95], axis=1)
        axPlotIndex[i_plot].fill_between(flowerror_thresh,
                          data_percentile[0, :],
                          data_percentile[-1, :],
                          fc = color_fill[i_prot],
                          alpha = 0.1)
        axPlotIndex[i_plot].fill_between(flowerror_thresh,
                                         data_percentile[1, :],
                                         data_percentile[3, :],
                                         fc=color_fill[i_prot],
                                         alpha=0.6)
        axPlotIndex[i_plot].plot(flowerror_thresh, data_percentile[2,:],
                 c=color_fill[i_prot], label=list_of_protocols[include_prot[i_prot]]['legend'])
    axPlotIndex[i_plot].set_xscale('log')
    axPlotIndex[i_plot].set_yscale('log')
    axPlotIndex[i_plot].grid(True, which='major', axis='y')
    # axPlotIndex[i_plot].set_title(output_names[i_data], fontweight='bold')
    axPlotIndex[i_plot].set_xlabel(r'Steady-state threshold [-]')
    axPlotIndex[i_plot].set_ylabel(['LV volume', 'LV pressure'][i_plot],
                                   fontsize=fontsettings['label_fontsize'])
    adjust_spines(axPlotIndex[i_plot], ['left','bottom'], outward=0)
    # plt.legend(loc='upper right')
    axPlotIndex[i_plot].invert_xaxis()

    axPlotIndex[i_plot].axvline(plot_opt_thresh, c='k', ls='--', lw=1)

yl = [ax.get_ylim() for ax in axPlotIndex]
yl = [np.min(yl), np.max(yl)]
[ax.set_ylim(yl) for ax in axPlotIndex]

# plot computational time threshold


for i_plot in range(1):
    # if i_plot>=v*h-1:
    #     break
    ax = fig.add_subplot(gs[0, 0])

    output_of_interest = np.ndarray((all_idx_thresh_reached.shape[1], simulation_output.shape[1]))

    number_of_succes = np.zeros((len(include_prot), all_idx_thresh_reached.shape[1]),
                                dtype=float)



    i_data = idx_plot_indices[i_plot]
    for i_prot in range(len(include_prot)):
        idx_stable = np.argmax(simulation_stable, axis=2)
        idx_stable[idx_stable==0]=-1


        data = np.ndarray(output_of_interest.shape)

        # for i_thresh in range(all_idx_thresh_reached.shape[1]):
        #     for i_sim in range(simulation_output.shape[1]):
        #         aux = simulation_output[include_prot[i_prot],i_sim,:,i_data]
        #         aux[np.isnan(aux)] = -1
        #         if np.isnan(all_idx_thresh_reached[i_prot,i_thresh, i_sim]):
        #             output_of_interest[i_thresh, i_sim] = np.nan
        #         else:
        #             output_of_interest[i_thresh, i_sim] = aux[all_idx_thresh_reached[i_prot,i_thresh, i_sim].astype(int) ]
        #         # remove not converged simluatoins
        #         if idx_stable[i_prot1, i_sim]==0:
        #             output_of_interest[i_prot, i_sim] = np.nan

        # # Select protocol
        # data = np.abs( semi_true_values[:,i_data] - output_of_interest[:,:])

        for i_thresh in range(all_idx_thresh_reached.shape[1]):
            for i_sim in range(simulation_output.shape[1]):
                # print(FlowError[i_prot, i_sim, -1], flowerror_thresh[i_thresh])
                # if (np.isnan(all_idx_thresh_reached[i_prot,i_thresh, i_sim]) or
                #     FlowError[i_prot, i_sim, -1] > flowerror_thresh[i_thresh]
                #     ):
                #     data[i_thresh, i_sim] = np.nan
                # else:
                #     data[i_thresh, i_sim] = np.sum(calculation_time[include_prot[i_prot], i_sim, :all_idx_thresh_reached[i_prot,i_thresh, i_sim].astype(int)])
                #     number_of_succes[i_prot, i_thresh] += 1

                # np.sum(FlowError[0, :, [-1]] > flowerror_thresh.reshape((-1, 1)), axis=1).shape
                if FlowError[include_prot[i_prot], i_sim, -1] < flowerror_thresh[i_thresh]:
                    # print('a', FlowError[i_prot, i_sim, -1], flowerror_thresh[i_thresh])
                    data[i_thresh, i_sim] = all_idx_thresh_reached[i_prot,i_thresh, i_sim]
                    number_of_succes[i_prot, i_thresh] += 1 * 100/FlowError.shape[1]
                else:
                    print('r', FlowError[i_prot, i_sim, -1], flowerror_thresh[i_thresh])
                    data[i_thresh, i_sim] = np.nan




        # remove non-stable sims
        # data = data[:, idx_stable[i_prot, :]>0]


        # calculate
        idx_stable = idx_stable[idx_prot_groups==i_prot]

        data_percentile = np.nanpercentile(data, [2.5, 25, 50, 75, 97.5], axis=1)
        plt.fill_between(flowerror_thresh,
                          data_percentile[0,:],
                          data_percentile[-1,:],
                          fc=color_fill[i_prot],
                          alpha=0.1)
        plt.fill_between(flowerror_thresh,
                          data_percentile[1,:],
                          data_percentile[3,:],
                          fc=color_fill[i_prot],
                          alpha=0.6)
        plt.plot(flowerror_thresh, data_percentile[2,:],
                 c=color_fill[i_prot], label=list_of_protocols[include_prot[i_prot]]['legend'])

        plt.axvline(plot_opt_thresh, c='k', ls='--', lw=1)
    plt.xscale('log')
    # plt.yscale('log')
    plt.grid(True, which='major', axis='y')
    plt.title('Beats to steady-stable', fontweight='bold',
        fontsize=fontsettings['title_fontsize'],
        weight=fontsettings['title_fontweight'],
        )


    plt.xlabel(r'Steady-state threshold [-]')
    plt.ylabel(r'Number of beats [-] ')
    adjust_spines(ax, ['left','bottom'], outward=0)
    # plt.legend(loc='lower right')
    ax.invert_xaxis()


    left_location = subplots_adjust['left'] + 0.8*(
    subplots_adjust['right']-subplots_adjust['left'])

    ax.annotate('Pressure flow control error', (0.52, 0.95),
                    xycoords='figure fraction',
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=fontsettings['title_fontsize'],
                    fontweight=fontsettings['title_fontweight'])

    # axSucces = ax.twinx()
    # for i_prot in range(number_of_succes.shape[0]):
    #     axSucces.plot(flowerror_thresh,
    #                   number_of_succes[i_prot,:],
    #                   color=color_fill[i_prot])
    # axSucces.spines['top'].set_visible(False)
    # axSucces.set_ylim([-1, 101])

    # n_runned_simulations = FlowError.shape[2]
    # axSucces.set_ylabel(f'Converged simulations within {n_runned_simulations} simulations [%]',
    #                     fontsize=fontsettings['label_fontsize'])

# Legend
axLegend = fig.add_subplot(gs[0, 3])
for i_prot in range(len(include_prot)):
    axLegend.plot([0, 1], [i_prot, i_prot],
                  color=color_fill[i_prot])
    axLegend.fill_between([0, 1], [i_prot-0.1, i_prot-0.1], [i_prot+0.1, i_prot+0.1],
                          color=color_fill[i_prot], alpha=0.1)
    axLegend.fill_between([0, 1], [i_prot-0.2, i_prot-0.2], [i_prot+0.2, i_prot+0.2],
                          color=color_fill[i_prot], alpha=0.1)
    axLegend.annotate(list_of_protocols[include_prot[i_prot]]['legend'].replace('_o', '\no').replace('_', ' '), (1.1, i_prot),
                      horizontalalignment='left', verticalalignment='center',
                      fontsize=fontsettings['legend_fontsize'])



axLegend.set_xticks([])
axLegend.set_yticks([])
axLegend.set_xlim([0, 4])
axLegend.set_ylim([-1, len(include_prot)])
axLegend.spines['top'].set_visible(False)
axLegend.spines['right'].set_visible(False)
axLegend.spines['bottom'].set_visible(False)
axLegend.spines['left'].set_visible(False)
