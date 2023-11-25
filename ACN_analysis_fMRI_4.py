"""
This script plots the searchlight results
Note: this script should be okay!
"""
import os
import pickle
import nilearn
import pandas as pd
import numpy as np
from nilearn.image import new_img_like, load_img
from nilearn.plotting import plot_stat_map, plot_img, show
from nilearn import decoding
from nilearn.decoding import SearchLight
from sklearn import naive_bayes, model_selection #import GaussianNB
from datetime import datetime


# Getting back the objects:
searchlight_path = '/work/807746/emma_folder/notebooks/fMRI/project_repo/notebooks/searchlight_result/InSpe_first_level_models_all_trials_searchlight.pkl'

#searchlight_path = '/work/807746/emma_folder/notebooks/fMRI/project_repo/notebooks/sub-0116.pkl'
f = open(searchlight_path, 'rb')
searchlight = pickle.load(f)
f.close()

    
f = open('/work/807746/emma_folder/notebooks/fMRI/project_repo/notebooks/flms/InSpe_first_level_models_testtrain.pkl', 'rb')
fmri_img_train, fmri_img_test, idx_train,idx_test, conditions_train,  conditions_test= pickle.load(f)
f.close()

# ------------- Plot searchlight analysis results ----------------- #

from nilearn import image, plotting
from nilearn.plotting import plot_glass_brain, plot_stat_map
from nilearn.image import new_img_like, load_img

mask_wb_filename='/work/816119/InSpePosNegData/BIDS_2023E/derivatives/sub-0117/anat/sub-0117_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
anat_filename='/work/816119/InSpePosNegData/BIDS_2023E/derivatives/sub-0117/anat/sub-0117_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz'

now = datetime.now()
print('Plotting and saving searchlight output (threshold:0.6):',now.strftime("%H:%M:%S"))

save_directory = '/work/807746/emma_folder/notebooks/fMRI/project_repo/notebooks/searchlight_result'

# Ensure the save_directory exists, create it if not
if not os.path.exists(save_directory):
    os.makedirs(save_directory)


#Create an image of the searchlight scores
searchlight_img = new_img_like(anat_filename, searchlight.scores_) # Create a new image with the same meta data as the anatomical data


fig_negpos_unthresholded = plot_glass_brain(searchlight_img, cmap='jet',colorbar=True, threshold=0.5,
                          title='negative vs pos (unthresholded)',
                          plot_abs=False)

fig_negpos_unthresholded.savefig("/work/807746/emma_folder/notebooks/fMRI/project_repo/notebooks/searchlight_result/searchlight_resultInSpe_neg_vs_but_searchlightNB_glass_unthresholded.png", dpi=300)

fig_negpos_06_thresh=plotting.plot_glass_brain(searchlight_img,cmap='prism',colorbar=True,threshold=0.60,title='negative vs pos (Acc>0.6')

fig_negpos_06_thresh.savefig("/work/807746/emma_folder/notebooks/fMRI/project_repo/notebooks/searchlight_result/searchlight_resultInSpe_neg_vs_but_searchlightNB_glass_thresh06.png", dpi=300)
#plt.show()


stat_map =  plot_stat_map(searchlight_img, cmap='jet',threshold=0.6, cut_coords=[-30,-20,-10,0,10,20,30],
              display_mode='z',  black_bg=False,
              title='neg vs pos (Acc>0.6)')

stat_map.savefig("/work/807746/emma_folder/notebooks/fMRI/project_repo/notebooks/searchlight_result/searchlight_resultInSpe_pos_vs_neg_searchlightNB_statmap_thresh06.png", dpi=300)