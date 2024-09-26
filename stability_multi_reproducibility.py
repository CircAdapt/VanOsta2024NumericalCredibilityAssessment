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
folder_name = 'data_multi_reproducibility_PFC-' + ('on' if PFC_on else 'off')
n_sims = 100
n_beats = 100

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
    # get_list_of_protocols(
    #     solvers=['backward_differential_o2'],
    #     dt=[1e-4], TriSeg_thresh_F=[1e-3])
    # +
    get_list_of_protocols(
        solvers=['adams_moulton_o2'],
        dt=[1e-3], TriSeg_thresh_F=[1e-1], fac_pfc=[1])
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


calculation_time1 = np.ndarray((len(list_of_protocols), n_sims, n_sims, n_beats))
calculation_time1[:] = np.nan
simulation_succes1 = np.zeros((len(list_of_protocols), n_sims, n_sims, n_beats), dtype=bool)
simulation_stable1 = np.zeros((len(list_of_protocols), n_sims, n_sims, n_beats), dtype=bool)
simulation_high_pressure1 = np.zeros((len(list_of_protocols), n_sims, n_sims, n_beats), dtype=bool)
simulation_output1 = np.ndarray((len(list_of_protocols), n_sims, n_sims, n_beats, n_out))
simulation_output1[:] = np.nan
simulation_signals1 = np.ndarray((len(list_of_protocols), n_sims, n_sims, n_beats), dtype=object)

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


        calculation_time1[i_protocol,:] = data['calculation_time1']
        simulation_succes1[i_protocol,:] = data['simulation_succes1']
        simulation_stable1[i_protocol,:] = data['simulation_stable1']
        simulation_high_pressure1[i_protocol, :] = data['simulation_high_pressure1']
        simulation_output1[i_protocol,:] = data['simulation_output1']

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
                    # simulation_output[i_protocol, i_sim, i_beat, :], _ = getY(model)
                    output, signals = getY(model)
                    simulation_output[i_protocol, i_sim, i_beat, :] = output
                    simulation_signals[i_protocol, i_sim, i_beat] = signals

                high_pressure = (np.mean(model['Cavity']['p'][:, 'La']) > 50*133)


            if is_crashed:
                continue
            # Go to next beat
            model_state = model.model_export()

            for i_sim1 in range(n_sims):
                # print(f'\r {i_sim}.{i_sim1}', end='\r')
                if solver[-3:-1] == '_o':
                    model = circadapt.VanOsta2024(solver[:-3])
                    model.set('Solver.order', int(solver[-1]))
                else:
                    model = circadapt.VanOsta2024(solver)
                model.model_import(model_state)

                # set paremeters
                setX(model, X[i_sim1, :])
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
                        simulation_output1[i_protocol, i_sim, i_sim1, i_beat:, :] = np.nan
                        simulation_stable1[i_protocol, i_sim, i_sim1, i_beat:] = False
                        simulation_high_pressure1[i_protocol, i_sim, i_sim1, i_beat:] = high_pressure
                        calculation_time1[i_protocol, i_sim, i_sim1, i_beat:] = np.nan
                        break
                    else:
                        calculation_time1[i_protocol, i_sim, i_sim1, i_beat] = dt*1e3
                        simulation_succes1[i_protocol, i_sim, i_sim1, i_beat] = True
                        simulation_stable1[i_protocol, i_sim, i_sim1, i_beat] = model.is_stable()
                        simulation_high_pressure1[i_protocol, i_sim, i_sim1, i_beat] = False
                        # simulation_output1[i_protocol, i_sim, i_sim1, i_beat, :], _ = getY(model)
                        output, signals = getY(model)
                        simulation_output1[i_protocol, i_sim, i_sim1, i_beat, :] = output
                        simulation_signals1[i_protocol, i_sim, i_sim1, i_beat] = signals

                    high_pressure = (np.mean(model['Cavity']['p'][:, 'La']) > 50*133)







        data = {
            'calculation_time': calculation_time[i_protocol],
            'simulation_succes': simulation_succes[i_protocol],
            'simulation_stable': simulation_stable[i_protocol],
            'simulation_high_pressure': simulation_high_pressure[i_protocol],
            'simulation_output': simulation_output[i_protocol],

            'calculation_time1': calculation_time1[i_protocol],
            'simulation_succes1': simulation_succes1[i_protocol],
            'simulation_stable1': simulation_stable1[i_protocol],
            'simulation_high_pressure1': simulation_high_pressure1[i_protocol],
            'simulation_output1': simulation_output1[i_protocol],
            }
        np.save(filename, data)
print('finished running simulations')


# %%
plt.figure(1, clear=True)
plt.plot(simulation_output1[0, :, 1, :, 2].T)

# %%

idx = simulation_succes[0, :, -1]
for i_output in range(len(output_names)):
    print(output_names[i_output], np.std(simulation_output1[0, idx, 0, -1, i_output]))


# %% stable
simulation_succes
idx_stable = simulation_succes[0, :, -1]
print('Succes: ', np.sum(idx_stable), '/', n_sims)

print('Succes: ', np.sum(simulation_succes1[0, idx_stable, :, -1][:, idx_stable]), '/', np.sum(idx_stable)**2)

idx_crashed = np.invert(simulation_succes1[0, idx_stable, :, -1][:, idx_stable]).reshape(-1)

# nanmax = np.nanmax(simulation_output1[0, idx_stable, idx_stable, :, 3].reshape(-1, 250)[idx_crashed, :], axis=1)

#
# np.nanmax(simulation_output1[0, idx_stable, :, :, 3][:, idx_stable].reshape(-1, 250)[idx_crashed], axis=1)



print('')

# %%
simulation_signals
i_prot=0
errors = np.empty((n_sims, n_sims, 2))
errors[:] = np.nan
for i_sim in range(n_sims):
    
    # plt.figure(124124+i_sim, clear=True)
    # ax = [plt.subplot(2, 1, 1), plt.subplot(2, 1, 2)]
    for i_dat in range(2):
        if simulation_signals[i_prot, i_sim, -1] is None:
            continue
        signal_true = simulation_signals[i_prot, i_sim, -1][i_dat+1]
        # ax[i_dat].plot(signal_true)

        for i_sim1 in range(n_sims):
            if simulation_signals1[i_prot, i_sim1, i_sim, -1] is None:
                continue
            signal_compare = simulation_signals1[i_prot, i_sim1, i_sim, -1][i_dat+1]

            errors[i_sim, i_sim1, i_dat] = np.mean(np.abs(signal_true - signal_compare))
            # ax[i_dat].plot(signal_compare, '--')
            

print('Mean absolute volume error [5, 50, 95%]: ', np.nanpercentile(errors[:, :, 0].reshape(-1), [2.5, 50, 97.5]))
print('Mean absolute pressure error [5, 50, 95%]: ', np.nanpercentile(errors[:, :, 1].reshape(-1), [2.5, 50, 97.5]))


# %%
# true_data = np.load('data_multi-beat_PFC-on/')

# %%
output_include = [8, 9, 0, 1, 2, 3]
error_ylabels = ['MAP [-]',
                 'Venous Return [-]',
                 'EDV [-]',
                 'ESV [-]',
                 'max LV pressure [-]',
                 'min LV pressure [-]',
                 ]

simulation_output1[0, :, 0, -1, output_include]

np.nanmean([np.nanstd(simulation_output1[0, :, i, -1, output_include], axis=1) for i in range(n_sims)], axis=0)

np.nanmean(np.nanmean([np.abs(simulation_output1[0, :, i, -1, output_include] - np.nanmean(simulation_output1[0, :, i, -1, output_include], axis=1).reshape(-1, 1))
            for i in range(n_sims)], axis=0), axis=1)*1e3
