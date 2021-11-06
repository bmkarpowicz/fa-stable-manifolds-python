# %% 
from factor_analysis import (get_factor_analysis_loading, get_stabilization_matrices, 
    update_factor_analysis_loading)
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

# %% Define data paths 
# Saved directly from within Degenhart code before and 
# after instabilities are applied
calibration_data_path = 'calibration_data.mat'
update_data_path = 'stabilizer_update_data.mat'

orig_eval_data_path = 'orig_eval_data.mat'
update_eval_data_path = 'instab_eval_data.mat'

#%% Define parameters
NLATENTS = 10

# %% Load data 
calibration_data = loadmat(calibration_data_path)['calibrationData'][0]
update_data = loadmat(update_data_path)['stabilizerUpdateData'][0]

# %% Fit baseline stabilizer 
cal_data_concat = np.hstack([x for x in calibration_data]) # neurons x time 
cal_data_concat = np.expand_dims(cal_data_concat.T, axis=0) # trial x time x neurons
loading, psi, d = \
    get_factor_analysis_loading(cal_data_concat, NLATENTS, n_restarts=5)

# %% Get beta, O from original model 
baseline_beta, baselineO = get_stabilization_matrices(loading, psi, d)

# %% Update the stabilizer 

update_data_concat = np.hstack([x for x in update_data])
update_data_concat = np.expand_dims(update_data_concat.T, axis=0) # trial x time x neurons 

updated_beta, updatedO, aligned_channels = update_factor_analysis_loading(loading, 
    update_data_concat, NLATENTS, n_restarts=5, n_stable_rows=60, threshold=0.01)

#%% Load evaluation data 
orig_eval_data = loadmat(orig_eval_data_path)
update_eval_data = loadmat(update_eval_data_path)

# %% Compute latents 
orig_latent = [np.matmul(baseline_beta, t) + \
    np.expand_dims(baselineO, axis=-1) for t in orig_eval_data['origInstabilityEvalData'][0]]
upd_aligned_latent = [np.matmul(updated_beta, t) + \
    np.expand_dims(updatedO, axis=-1) for t in update_eval_data['instabilityEvalData'][0]]
upd_unaligned_latent = [np.matmul(baseline_beta, t) + \
    np.expand_dims(baselineO, axis=-1) for t in update_eval_data['instabilityEvalData'][0]]

# %% Plotting 
exTrial = 9 #python indices
fig, ax = plt.subplots(int(np.ceil(NLATENTS/2)), 2, figsize=(8, 10))
ax = ax.flatten()
for lI in range(0, NLATENTS): 
    ax[lI].plot(orig_latent[exTrial][lI,:], 'ko-', label ='Baseline Stabilizer with Original Data')
    ax[lI].plot(upd_aligned_latent[exTrial][lI,:], 'b.-', label='Updated Stabilizer with Instabilities')
    ax[lI].plot(upd_unaligned_latent[exTrial][lI,:], 'r--', label='Baseline Stabilizer with Instabilities')
    ax[lI].set_ylabel('Latent ' + str(lI+1) + ' (a.u.)')
    
ax[-1].legend(loc='center', bbox_to_anchor=(0.5, -0.5))
plt.show(block=False)

import pdb; pdb.set_trace()
# %%
