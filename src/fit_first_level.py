"""
The current scripts create a first level model to each of the subjects data (in a loop).

The function is based on the original script created by Emma Olsen and Sirid Wihlborg (https://github.com/emmarisgaardolsen/BSc_project_fMRI/blob/main/fmri_analysis_scripts/first_level_fit_function.py) as well as modifications made by Laura Bock Paulsen (https://github.com/laurabpaulsen/fMRI_analysis).

"""

from pathlib import Path
import os 
import numpy as np
import pandas as pd
from nilearn import masking
import nibabel as nib
from nilearn.glm.first_level import FirstLevelModel
import pickle

def load_prep_events(path): 
    """
    This function loads event files into a pandas dataframe 
    and modifies it to keep only the columns relevant to 
    perform out analysis. 

    Parameters
    ----------
    path : Path
        Path to a tsv file containing the events for a particular run
    
    Returns
    -------
    event_df : pd.DataFrame
        Pandas dataframe containing the events
    """

    # load the eventfile as a df
    event_df = pd.read_csv(path, sep='\t')

    # add button presses to the event dataframe, using the customised function add_button_presses
    event_df = add_button_presses(event_df)
    
    
    # Remove all IMG_BI events
    # event_df = event_df[event_df["trial_type"] != "IMG_BI"]

    # extract data corresponding to the needed columns only
    event_df = event_df.loc[:, ["onset", "duration", "trial_type"]]
    
    # change trial types (IMG_PO/IMG_PS gets positive, IMG_NO/IMG_NS gets negative, IMG_BI gets IMG_button)
    event_df["trial_type"] = event_df["trial_type"].apply(change_trial_type)

    # ensuring that the DataFrame index is a simple range index
    event_df = event_df.reset_index(drop = True)

    # return the modified df to be used for analysis
    return event_df


def load_prep_confounds(path, confound_cols=['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'csf', 'white_matter']):
    """
    Loads selected columns from a TSV file into a DataFrame.

    Parameters
    ----------
    path : Path
        File path to the TSV file.
    confound_cols : list of strings, optional
        Column names to load (default includes common confound variables).

    Returns
    -------
    confounds_df : pd.DataFrame
        DataFrame with specified confound columns.
    """

    confounds_df = pd.read_csv(path, sep='\t', usecols=confound_cols)
    return confounds_df



def add_button_presses(event_df, trial_type_col="trial_type", response_col="RT"):
    """
    Adds button presses to an event dataframe based on "IMG_BI" events and corresponding response times.

    Parameters
    ----------
    event_df : pd.DataFrame
        Dataframe containing the events.
    trial_type_col : str
        Column name for trial types.
    response_col : str
        Column name for response times.
    
    Returns
    -------
    pd.DataFrame
        Updated dataframe with button presses added.
    """

    # Filter out rows where response time is NaN and trial type is "IMG_BI"
    valid_presses = event_df[(event_df[trial_type_col] == "IMG_BI") & (~event_df[response_col].isna())]

    # Calculate onsets for button presses
    button_press_onsets = valid_presses[response_col] + valid_presses["onset"]

    # Create a DataFrame for new button press events
    button_press_df = pd.DataFrame({
        "onset": button_press_onsets,
        "duration": 0,
        "trial_type": "button_press"
    })

    # Concatenate the new button press events to the original DataFrame
    event_df = pd.concat([event_df, button_press_df], ignore_index=True).sort_values(by="onset")

    return event_df


def change_trial_type(trial_type):
    if trial_type in ['IMG_NS','IMG_NO']:
        return "negative"
    elif trial_type in ['IMG_PS', 'IMG_PO']:
        return "positive"
    elif trial_type in ['IMG_BI']:
        return "IMG_button"
    else:
        return trial_type

def fit_first_level_subject(subject, bids_dir, runs=[1, 2, 3, 4, 5, 6], space="MNI152NLin2009cAsym"):
    """
    Fit first level model for one subject.

    Parameters
    ----------
    subject : str
        Subject identifier e.g. "0102".
    bids_dir : Path
        Path to the root of the BIDS directory.
    runs : list of int
        List of runs to load.
    space : str
        Name of the space of the data to load.
    
    Returns
    -------
    first_level_model : FirstLevelModel
        First level model fitted for one subject.
    """
    
    bids_func_dir = os.path.join(bids_dir, f"sub-{subject}", "func")
    fprep_func_dir = os.path.join(bids_dir, "derivatives", f"sub-{subject}", "func")

    # Construct file paths
    fprep_func_paths = [os.path.join(fprep_func_dir, f"sub-{subject}_task-boldinnerspeech_run-{run}_echo-1_space-{space}_desc-preproc_bold.nii.gz") for run in runs]
    event_paths = [os.path.join(bids_func_dir, f"sub-{subject}_task-boldinnerspeech_run-{run}_echo-1_events.tsv") for run in runs]
    confounds_paths = [os.path.join(fprep_func_dir, f"sub-{subject}_task-boldinnerspeech_run-{run}_desc-confounds_timeseries.tsv") for run in runs]
    mask_paths = [os.path.join(fprep_func_dir, f"sub-{subject}_task-boldinnerspeech_run-{run}_space-{space}_desc-brain_mask.nii.gz") for run in runs]

    # Load data
    events = [load_prep_events(path) for path in event_paths]
    confounds = [load_prep_confounds(path) for path in confounds_paths]
    masks = [nib.load(path) for path in mask_paths]

    # Merge masks
    mask_img = masking.intersect_masks(masks, threshold=0.8)

    # Fit first level model
    t_r = int(nib.load(fprep_func_paths[0]).header['pixdim'][4])
    first_level_model = FirstLevelModel(t_r=t_r, mask_img=mask_img, slice_time_ref=0.5, hrf_model="glover", verbose=1)
    first_level_model.fit(fprep_func_paths, events, confounds)
    
    return first_level_model



if __name__ in "__main__":
    path = Path(__file__).parents[1]
    output_path = path / "flms"

    # make sure that output path exists
    if not output_path.exists():
        output_path.mkdir(parents = True)

    bids_dir = Path("/work/816119/InSpePosNegData/BIDS_2023E")
    subjects = ["0116", "0117", "0118", "0119", "0120", "0121", "0122", "0123"]
    
    for subject in subjects:
        flm = fit_first_level_subject(subject, bids_dir) 
        file_name = f"flm_{subject}.pkl"
        pickle.dump(flm, open(output_path / file_name, 'wb'))
            
