# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import matplotlib

import circadapt

model = circadapt.VanOsta2024()


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

# %%
model = circadapt.VanOsta2024()
model.run(1)

plt.figure(3, clear=True, figsize=(4,2))
ax = plt.subplot(1,2,1)
ax.plot(model['Patch']['Ef'][:, ['pLv0', 'pRv0', 'pLa0', 'pRa0', 'pSv0']],
         model['Patch']['Sf'][:, ['pLv0', 'pRv0', 'pLa0', 'pRa0', 'pSv0']]*1e-3,
         )
ax = plt.subplot(1,2,2)
ax.plot(model['Cavity']['V'][:, ['cLv', 'cRv', 'La', 'Ra']] * 1e6,
         model['Cavity']['p'][:, ['cLv', 'cRv', 'La', 'Ra']] / 133,
         )




# %%
model['Solver']['store_beats'] = 10
# model['PFC']['q0'] *= 1.5
model['Patch']['Sf_act'][2:] *= 0.4
model.run(10)

plt.figure(1, clear=True, figsize=(6, 2))
ax = plt.subplot(1,1,1)
ax.plot(model['Solver']['t']*1e3, 
         model['Cavity']['p'][:, ['cLv', 'SyArt', 'La']] / 133,
         )
ax.axhline(model['PFC']['p0']/133)


plt.figure(2, clear=True, figsize=(6, 2))
ax = plt.subplot(1,1,1)
ax.plot(model['Solver']['t']*1e3, 
         model['Cavity']['V'][:, 'cLv'] * 1e6,
         )


