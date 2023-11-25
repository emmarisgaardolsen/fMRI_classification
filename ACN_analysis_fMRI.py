"""
This script contains the code necessariy for conducting analysis of fMRI data
Author: Emma Risgaard Olsen

Before running this script, we need to run notebook 15 to get the first level models.
The script loads models and data from notebook 15 and change the design matrix to one that gives a beta estimate for each trial.
The script then saves the new design matrix and the fitted models for each session for subject 0117.
"""

import os
import pickle
import numpy as np 
import pickle
import nilearn 
from nilearn.glm.first_level import make_first_level_design_matrix
import pandas as pd
from datetime import datetime
from nilearn.plotting import plot_design_matrix
import matplotlib.pyplot as plt


# ------------- import data and first level models ----------------- #
f = open('/work/807746/emma_folder/notebooks/fMRI/project_repo/notebooks/flms/InSpe_first_level_models.pkl', 'rb')
models, models_run_imgs, models_events, models_confounds = pickle.load(f)
f.close()

now = datetime.now()
print('Loaded models:',now.strftime("%H:%M:%S"))


# ------------------------ Single trial models ---------------------- #
## Create new design matrices with a column per experimental trial 
lsa_dm=[]

# Changing the model for a single participant (the second):
for ii in range(len(models_events[1])):
    # Sort onsets by trial type to make later processing easier
    #models_events[1][ii].sort_values(by=['trial_type'], inplace=True)
    #N=number of events
    N=models_events[1][ii].shape[0]
    # time frame with 490 scans with 1s TR:
    t_fmri = np.linspace(0, 490,490,endpoint=False)
    # We have to create a dataframe with onsets/durations/trial_types
    # No need for modulation!
    trials = pd.DataFrame(models_events[1][ii], columns=['onset'])
    trials.loc[:, 'duration'] = 0.7
    trials.loc[:, 'trial_type'] = [models_events[1][ii]['trial_type'][i-1]+'_'+'t_'+str(i).zfill(3)  for i in range(1, N+1)]

    # lsa_dm = least squares all design matrix
    lsa_dm.append(make_first_level_design_matrix(
        frame_times=t_fmri,  # we defined this earlier 
        events=trials,
        add_regs=models_confounds[1][ii], # Add the confounds from fmriprep for the second participant
        hrf_model='glover',
        drift_model='cosine'  
    ))
    
now = datetime.now()
print('Finish making single trial models:',now.strftime("%H:%M:%S"))


# ------------------------- Save plots of heatmaps and design matrixces ------------------------ #
save_directory = '/work/807746/emma_folder/notebooks/fMRI/project_repo/notebooks/figs'

# Ensure the save_directory exists, create it if not
if not os.path.exists(save_directory):
    os.makedirs(save_directory)
    
# Define the desired width and height of the plots
plot_width = 15  # Adjust this value as needed
plot_height = 5  # Adjust this value as needed

for ii in range(len(models_events[1])):
    # Create a new figure with the specified width and height
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    
    # Plot the design matrix
    plot_design_matrix(lsa_dm[ii], ax=ax)
    
    # Save the plot with a unique filename
    plot_filename = os.path.join(save_directory, f'design_matrix_plot_{ii}.png')
    plt.savefig(plot_filename, bbox_inches='tight')  # Use bbox_inches='tight' to prevent cropping
    
    # Close the figure to release resources (optional)
    plt.close(fig)
    

now = datetime.now()
print('Saved design_matrix_plot:', now.strftime("%H:%M:%S"))

import seaborn as sns
dm_corr = lsa_dm[0].corr()
p1 = sns.heatmap(dm_corr)
heatmap_filename = os.path.join(save_directory, 'heatmap_plot.png')
plt.savefig(heatmap_filename)

now = datetime.now()
print('Saved heatmap_plot:', now.strftime("%H:%M:%S"))


# --------------- Fit models for all sessions for subject 0117 ------------------------ #
from nilearn.glm.first_level import FirstLevelModel

model2=[]
for ii in range(len(models_events[1])):
    
    # Get data and model info for the second participant, 1st session
    imgs2=models_run_imgs[1][ii]
    model2.append (FirstLevelModel())
    # Fit the model
    print('Fitting GLM: ', ii+1)
    model2[ii].fit(imgs2, design_matrices=lsa_dm[ii])

now = datetime.now()
print('Finishing model fit:',now.strftime("%H:%M:%S"))

# ---------------- Saving the fitted models and design matrices for sub0117 ---------------- # 
f = open('/work/807746/emma_folder/notebooks/fMRI/project_repo/notebooks/flms/InSpe_first_level_models_all_trials_sub0117.pkl', 'wb')
pickle.dump([model2, lsa_dm], f)
f.close()

