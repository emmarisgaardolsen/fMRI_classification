"""
Author: Emma Risgaard Olsen

The current script makes the beta map contrast from the fitted model to use in later analysis
"""

import os
import pickle
import numpy as np 
import pickle
import nilearn 
from nilearn.glm.first_level import make_first_level_design_matrix
import pandas as pd
from datetime import datetime

# Get back all first_level_models (all subs) 
f = open('/work/807746/emma_folder/notebooks/fMRI/project_repo/notebooks/flms/InSpe_first_level_models.pkl', 'rb')
models, models_run_imgs, models_events, models_confounds = pickle.load(f)
f.close()

## Getting back the fitted models for each session for subject 0117:
f = open('/work/807746/emma_folder/notebooks/fMRI/project_repo/notebooks/flms/InSpe_first_level_models_all_trials_sub0117.pkl', 'rb')
model2, lsa_dm = pickle.load(f)
f.close()


# ----- Making beta map contrast from the fitted model to use in later analysis ---- #
now = datetime.now()
print('Computing contrasts:',now.strftime("%H:%M:%S"))
b_maps = []
conditions_label = []


# ----- Making beta map contrast from the fitted model to use in later analysis ---- #
for ii in range(len(models_events[1])):
    N=models_events[1][ii].shape[0]
    #Make an identity matrix with N= number of trials
    contrasts=np.eye(N)
    #print(contrasts.shape)
    #Find difference between columns in design matrix and number of trials
    dif=lsa_dm[ii].shape[1]-contrasts.shape[1]
    #print(dif)
    #Pad with zeros
    contrasts=np.pad(contrasts, ((0,0),(0,dif)),'constant')
    #print(contrasts.shape)
    print('Making contrasts for session : ', ii+1)
    print('Number of contrasts : ', N)
    for i in range(N):
        #Add a beta-contrast image from each trial
        b_maps.append(model2[ii].compute_contrast(contrasts[i,], output_type='effect_size'))
        # Make a variable with condition labels for use in later classification
        conditions_label.append(lsa_dm[ii].columns[i])
#        session_label.append(session)

now = datetime.now()
print('Done computing contrasts:',now.strftime("%H:%M:%S"))

import pickle

# ------- Save the first level models (beta maps) for classification -------# 
# Saving the objects:
f = open('/work/807746/emma_folder/notebooks/fMRI/project_repo/notebooks/flms/InSpe_first_level_models_all_trials_sub0117_for_classification.pkl', 'wb')
pickle.dump([model2, lsa_dm, conditions_label, b_maps], f)
f.close()

now = datetime.now()
print('Saved beta-maps:',now.strftime("%H:%M:%S"))


