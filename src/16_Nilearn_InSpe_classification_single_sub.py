#!/usr/bin/env python
# coding: utf-8

"""
This script makes a trial by trial fit of InSpe data and uses that for searchlight classificiation. 

The script loads models and data from notebook 15 and modifies the design matrix
so that it gives one beta estimate per trial.

Afterwards, searchlight classification analysis is performed and the best performing voxels are found. Lastly, a permutation test is conducted on a test split of the data, eventually choosing the best voxels.

The script takes a long time to run and uses a lot of memory, so the most efficient way of using it is running it using tmux and using a machine with more RAM.
"""

import os
path='/work/807746/emma_folder/notebooks/' # Remember to change this to your own path
os.chdir(path)


# Additional imports
import sys
import nilearn
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd


# #### Importing data and models from tutorial 15
# The data in tutorial 15 were analysed to not include the self/other distinction. If you want to study that, you need to edit the event names.


import pickle

now = datetime.now()
print('Starting cell:',now.strftime("%H:%M:%S"))


f = open('/work/807746/emma_folder/notebooks/fMRI/project_repo/notebooks/new_flms/InSpe_first_level_models.pkl', 'rb')
models, models_run_imgs, models_events, models_confounds = pickle.load(f)
f.close()



now = datetime.now()
print('Loaded models:',now.strftime("%H:%M:%S"))


# ### Figuring out what is in the models_events variable


# Inspect number of scans and confounds included in subject 117, first run:
print(models_confounds[1][0].shape)
# Inspect number of trials, onsets and trial types for the first participant, first run:
print(models_events[1][0])


# ## Creating new design matrices with a column per experimental trial.

import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix
import numpy as np

lsa_dm=[]

# Changing the model for a single participant (the first):
for ii in range(len(models_events[1])):
    # Sort onsets by trial type to make later processing easier
    #models_events[0][ii].sort_values(by=['trial_type'], inplace=True)
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
        add_regs=models_confounds[1][ii], #Add the confounds from fmriprep
        hrf_model='glover',
        drift_model='cosine'  
    ))
    




now = datetime.now()
print('Finish making single trial models:',now.strftime("%H:%M:%S"))


# ### Check out the created design matrix
# Note that the index represents the frame times

# In[7]:


print(lsa_dm[0])


# ## Plot the new design matrices

from nilearn.plotting import plot_design_matrix
for ii in range(len(models_events[1])):
    plot_design_matrix(lsa_dm[ii]);

now = datetime.now()
print('Finishing cell:',now.strftime("%H:%M:%S"))


# ### Let's inspect the correlational structure of the design matrix
import seaborn as sns
dm_corr=lsa_dm[0].corr()
p1 = sns.heatmap(dm_corr)


# ## Fit the models for all sessions from one participant
from nilearn.glm.first_level import FirstLevelModel

model1=[]
for ii in range(len(models_events[1])):
    
    # Get data and model info for 1st participant, 1st session
    imgs1=models_run_imgs[1][ii]
    model1.append (FirstLevelModel())
    #Fit the model
    print('Fitting GLM: ', ii+1)
    model1[ii].fit(imgs1,design_matrices=lsa_dm[ii])

now = datetime.now()
print('Finishing model fit:',now.strftime("%H:%M:%S"))


# ## Saving/retrieving the fitted models and design matrices
import pickle

# Saving the objects:
f = open('/work/807746/emma_folder/notebooks/fMRI/project_repo/notebooks/flms/InSpe_first_level_models_all_trials_all.pkl', 'wb')
pickle.dump([model1, lsa_dm], f)
f.close()

## Getting back the objects:
#f = open('/work/807746/emma_folder/notebooks/fMRI/project_repo/notebooks/flms/InSpe_first_level_models_all_trials_all.pkl', 'rb')
#model1, lsa_dm = pickle.load(f)
#f.close()

#print(model1[0])
now = datetime.now()
print('Saved model and design matrices:',now.strftime("%H:%M:%S"))


# ## Making beta map contrasts from the fitted model to use in later analyses
now = datetime.now()
print('Computing contrasts:',now.strftime("%H:%M:%S"))
b_maps = []
conditions_label = []

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
        b_maps.append(model1[ii].compute_contrast(contrasts[i,], output_type='effect_size'))
        # Make a variable with condition labels for use in later classification
        conditions_label.append(lsa_dm[ii].columns[i])
#        session_label.append(session)

now = datetime.now()
print('Done computing contrasts:',now.strftime("%H:%M:%S"))


# ## Saving models and beta maps
import pickle

#Save the first level models

# Saving the objects:
f = open('/work/807746/emma_folder/notebooks/fMRI/project_repo/notebooks/flms/InSpe_first_level_models_all_trials_all.pkl', 'wb')
pickle.dump([model1, lsa_dm, conditions_label, b_maps], f)
f.close()

# Getting back the objects:
#f = open('/work/807746/emma_folder/notebooks/fMRI/project_repo/notebooks/flms/InSpe_first_level_models_all_trials_all.pkl', 'rb')

#model1, lsa_dm, conditions_label, b_maps = pickle.load(f)
#f.close()

now = datetime.now()
print('Saved beta-maps:',now.strftime("%H:%M:%S"))

del model1


# ## Reshape data for classification
# Checking that the design matrix and the condition labels look the same.

# In[19]:


print('Checking that column names for design matrix matches labels')
print(lsa_dm[0].columns[0:9])
print(conditions_label[0:9])


# Selecting Positive and negative trials
now = datetime.now()
print('Renaming labels to N, P, and B:',now.strftime("%H:%M:%S"))

f = open('/work/807746/emma_folder/notebooks/fMRI/project_repo/notebooks/flms/InSpe_first_level_models_all_trials_all.pkl', 'rb')
model1, lsa_dm, conditions_label, b_maps = pickle.load(f)
f.close()

import numpy as np
#from nilearn import datasets
from nilearn.image import new_img_like, load_img, index_img, clean_img, concat_imgs
from sklearn.model_selection import train_test_split, GroupKFold
n_trials=len(conditions_label)
#print(n_trials)

#Concatenate beta maps
b_maps_conc=concat_imgs(b_maps)
#print(b_maps_conc.shape)
del b_maps
# Reshaping data------------------------------
from nilearn.image import index_img, concat_imgs
#Find all negative and positive trials
idx_neg=[int(i) for i in range(len(conditions_label)) if 'N_' in conditions_label[i]]
idx_pos=[int(i) for i in range(len(conditions_label)) if 'P_' in conditions_label[i]]
idx_but=[int(i) for i in range(len(conditions_label)) if 'B_' in conditions_label[i]]

#print(idx_neg)
#print(conditions_label)
for i in range(len(conditions_label)):
    if i in idx_neg:
        conditions_label[i]='N'
    elif i in idx_pos:
        conditions_label[i]='P'
    elif i in idx_but:
        conditions_label[i]='B'
print(conditions_label)

now = datetime.now()
print('Selecting to N and P:',now.strftime("%H:%M:%S"))
# Make index of relevant trials
idx=np.concatenate((idx_neg, idx_pos))
#print(idx)

#Select trials
conditions=np.array(conditions_label)[idx]
print(conditions)

#Select images
b_maps_img = index_img(b_maps_conc, idx)
print(b_maps_img.shape)


# ## create training and testing vars on the basis of class labels

# In[21]:


now = datetime.now()
print('Making a trial and test set:',now.strftime("%H:%M:%S"))
#conditions_img=conditions[idx]
#print(conditions_img)
#Make an index for spliting fMRI data with same size as class labels
idx2=np.arange(conditions.shape[0])

# create training and testing vars on the basis of class labels
idx_train,idx_test, conditions_train,  conditions_test = train_test_split(idx2,conditions, test_size=0.2)
#print(idx_train, idx_test)

# Reshaping data------------------------------
from nilearn.image import index_img
fmri_img_train = index_img(b_maps_img, idx_train)
fmri_img_test = index_img(b_maps_img, idx_test)
#Check data sizes
print('Trial and test set shape:')
print(fmri_img_train.shape)
print(fmri_img_test.shape)

# Saving the objects:
f = open('/work/807746/emma_folder/notebooks/fMRI/project_repo/notebooks/flms/InSpe_first_level_models_testtrain.pkl', 'wb')
pickle.dump([fmri_img_train, fmri_img_test, idx_train,idx_test, conditions_train,  conditions_test], f)
f.close()

now = datetime.now()
print('Trial and test set saved:',now.strftime("%H:%M:%S"))


# ## Prepare a searchlight analysis on the first split

# In[22]:


now = datetime.now()
print('Making a mask for analysis:',now.strftime("%H:%M:%S"))
# -------------------
import pandas as pd
import numpy as np
from nilearn.image import new_img_like, load_img
from nilearn.plotting import plot_stat_map, plot_img, show
from nilearn import decoding
from nilearn.decoding import SearchLight
from sklearn import naive_bayes, model_selection #import GaussianNB

#########################################################################
#Make a mask with the whole brain

mask_wb_filename='/work/816119/InSpePosNegData/BIDS_2023E/derivatives/sub-0117/anat/sub-0117_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
anat_filename='/work/816119/InSpePosNegData/BIDS_2023E/derivatives/sub-0117/anat/sub-0117_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'
#Load the whole brain mask
mask_img = load_img(mask_wb_filename)


# .astype() makes a copy.
process_mask = mask_img.get_fdata().astype(int)
#Set slices below x in the z-dimension to zero (in voxel space)
process_mask[..., :10] = 0
#Set slices above x in the z-dimension to zero (in voxel space)
process_mask[..., 170:] = 0
process_mask_img = new_img_like(mask_img, process_mask)


#Plot the mask on an anatomical background
plot_img(process_mask_img, bg_img=anat_filename,#bg_img=mean_fmri,
         title="Mask", display_mode="z",cut_coords=[-60,-50,-30,-10,10,30,50,70,80],
         vmin=.40, cmap='jet', threshold=0.9, black_bg=True)


# ## Run the searchlight analysis
# 
# Note. This takes many hours for one participant. I strongly recommond running this in tmux (see notebook folder for info).

# In[24]:


now = datetime.now()
print('Starting searchlight analysis:',now.strftime("%H:%M:%S"))
#n_jobs=-1 means that all CPUs will be used

from nilearn.decoding import SearchLight
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB

# The radius is the one of the Searchlight sphere that will scan the volume
searchlight = SearchLight(
    mask_img,
    estimator=GaussianNB(),
    process_mask_img=process_mask_img,
    radius=5, n_jobs=-1,
    verbose=10, cv=10)
searchlight.fit(fmri_img_train, conditions_train)

now = datetime.now()
print('Finishing searchlight analysis:',now.strftime("%H:%M:%S"))


# ## Save/restore the variables

import pickle
import nilearn

#Save the searchlight model

# Saving the objects:
f = open('/work/807746/emma_folder/notebooks/fMRI/project_repo/notebooks/flms/InSpe_first_level_models_all_trials_searchlight_negpos.pkl', 'wb')
pickle.dump([searchlight, searchlight.scores_], f)
f.close()


# Getting back the objects:
f = open('/work/807746/emma_folder/notebooks/fMRI/project_repo/notebooks/flms/InSpe_first_level_models_all_trials_searchlight_negpos.pkl', 'rb')
searchlight,searchlight_scores_ = pickle.load(f)
f.close()


# Getting back the objects:
f = open('/work/807746/emma_folder/notebooks/fMRI/project_repo/notebooks/flms/InSpe_first_level_models_testtrain.pkl', 'rb')
fmri_img_train, fmri_img_test, idx_train,idx_test, conditions_train,  conditions_test= pickle.load(f)
f.close()

now = datetime.now()
print('Searchlight output saved:',now.strftime("%H:%M:%S"))


# ## Plot the outcome of the searchlight analysis
from nilearn import image, plotting
from nilearn.plotting import plot_glass_brain, plot_stat_map
from nilearn.image import new_img_like, load_img
mask_wb_filename='/work/816119/InSpePosNegData/BIDS_2023E/derivatives/sub-0117/anat/sub-0117_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
anat_filename='/work/816119/InSpePosNegData/BIDS_2023E/derivatives/sub-0117/anat/sub-0117_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'

now = datetime.now()
print('Plotting and saving searchlight output (threshold:0.6):',now.strftime("%H:%M:%S"))

#Create an image of the searchlight scores
searchlight_img = new_img_like(anat_filename, searchlight.scores_)


plot_glass_brain(searchlight_img, cmap='jet',colorbar=True, threshold=0.5,
                          title='Negative vs Positive (unthresholded)',
                          plot_abs=False)

fig=plotting.plot_glass_brain(searchlight_img,cmap='prism',colorbar=True,threshold=0.60,title='negative vs button (Acc>0.6')
fig.savefig("/work/807746/emma_folder/notebooks/fMRI/project_repo/notebooks/InSpe_neg_vs_pos_searchlightNB_glass.png", dpi=300)
#plt.show()


plot_stat_map(searchlight_img, cmap='jet',threshold=0.6, cut_coords=[-30,-20,-10,0,10,20,30],
              display_mode='z',  black_bg=False,
              title='pos vs neg (Acc>0.6)')
plt.show()


# ### Find the 500 most predictive voxels 
print('Number of voxels in searchlight: ',searchlight.scores_.size)
#Find the percentile that makes the cutoff for the 500 best voxels
perc=100*(1-500.0/searchlight.scores_.size)
#Print percentile
print('Percentile for 500 most predictive voxels: ',perc)
#Find the cutoff
cut=np.percentile(searchlight.scores_,perc)
#Print cutoff
print('Cutoff for 500 most predictive voxels: ', cut)
#cut=0
#Make a mask using cutoff

#Load the whole brain mask
mask_img2 = load_img(mask_wb_filename)

# .astype() makes a copy.
process_mask2 = mask_img2.get_fdata().astype(int)
process_mask2[searchlight.scores_<=cut] = 0
process_mask2_img = new_img_like(mask_img2, process_mask2)


# ### Visualization of the voxels
from nilearn import image
from nilearn.plotting import plot_stat_map, plot_img, show
from nilearn import plotting
#Create an image of the searchlight scores
searchlight_img = new_img_like(anat_filename, searchlight.scores_)
#Plot the searchlight scores on an anatomical background
plot_img(searchlight_img, bg_img=anat_filename,#bg_img=mean_fmri,
         title="Searchlight", display_mode="z",cut_coords=[-25,-20,-15,-10,-5,0,5],
         vmin=.40, cmap='jet', threshold=cut, black_bg=True)
#plotting.plot_glass_brain effects
fig=plotting.plot_glass_brain(searchlight_img,threshold=cut)
fig.savefig("/work/807746/emma_folder/notebooks/fMRI/project_repo/notebooks/InSpe_neg_vs_pos_searchlightNB_glass_500.png", dpi=300)

now = datetime.now()
print('Saving glass brain with 500 most predictive voxels:',now.strftime("%H:%M:%S"))


# ### Make a permutation classification test on the 2nd data split using the best voxels

# In[29]:


now = datetime.now()
print('Perform permutation test on test set using 500 predictive voxels:',now.strftime("%H:%M:%S"))
from sklearn.naive_bayes import GaussianNB
from nilearn.input_data import NiftiMasker
masker = NiftiMasker(mask_img=process_mask2_img, standardize=False)

# We use masker to retrieve a 2D array ready
# for machine learning with scikit-learn
fmri_masked = masker.fit_transform(fmri_img_test)
#Print size of matrix (images x voxels)
print(fmri_masked.shape)

from sklearn.model_selection import permutation_test_score
score_cv_test, scores_perm, pvalue= permutation_test_score(
    GaussianNB(), fmri_masked, conditions_test, cv=10, n_permutations=1000, 
    n_jobs=-1, random_state=0, verbose=0, scoring=None)
print("Classification Accuracy: %s (pvalue : %s)" % (score_cv_test, pvalue))



# ## Saving permutation outcomes
import pickle

now = datetime.now()
print('Saving permutation scores:',now.strftime("%H:%M:%S"))
#Save the permutation scores

# Saving the objects:
f = open('/work/807746/emma_folder/notebooks/fMRI/project_repo/notebooks/InSpe_first_level_models_all_trials_searchlight_perms.pkl', 'wb')
pickle.dump([score_cv_test, scores_perm, pvalue], f)
f.close()

# Getting back the objects:
#f = open('/work/MikkelWallentin#6287/InSpe_first_level_models_all_trials_searchlight_perms.pkl', 'rb')
#score_cv_test, scores_perm, pvalue = pickle.load(f)
#f.close()


# ### View a histogram of permutation scores
now = datetime.now()
print('Plotting and saving permutation scores:',now.strftime("%H:%M:%S"))

import numpy as np
import matplotlib.pyplot as plt
# How many classes
n_classes = np.unique(conditions_test).size

plt.hist(scores_perm, 20, label='Permutation scores',
         edgecolor='black')
ylim = plt.ylim()
plt.plot(2 * [score_cv_test], ylim, '--g', linewidth=3,label='Classification Score'
         ' (pvalue %s)' % pvalue)
plt.plot(2 * [1. / n_classes], ylim, '--k', linewidth=3, label='Chance level')

plt.ylim(ylim)
plt.legend()
plt.xlabel('Score')

plt.savefig("/work/807746/emma_folder/notebooks/fMRI/project_repo/notebooks/InSpe_neg_vs_pos_one_sub_perm.png", dpi=300)
plt.show()


# In[ ]:


#get_ipython().system('jupyter nbconvert --to python /work/857248/16_Nilearn_InSpe_classification_single_sub.ipynb')


# In[ ]:




