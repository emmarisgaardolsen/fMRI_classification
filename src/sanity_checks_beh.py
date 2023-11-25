"""
This script contains the code needed to do sanity checks of the behavioral data
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data_dir = '/work/816119/InSpePosNegData/BIDS_2023E'

def list_files(startpath):
    """ Simple function to show directory tree. 
    From: https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python. """
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in sorted(files):
            print('{}{}'.format(subindent, f))
            
            
# list_files(data_dir)

def load_prep_events(path): 
    """
    Loads the event tsv and modifies it to contain the events we want

    Parameters
    ----------
    path : Path
        Path to tsv file containing the events
    
    Returns
    -------
    event_df : pd.DataFrame
        Pandas dataframe containing the events
    """
    # load the data
    event_df = pd.read_csv(path, sep='\t')

    # add button presses to the event dataframe
    event_df = add_button_presses(event_df)

    # get the data corresponding to the events and only keep the needed columns
    event_df = event_df.loc[:, ["onset", "duration", "trial_type"]]

    event_df["trial_type"] = event_df["trial_type"].apply(update_trial_type)

    event_df = event_df.reset_index(drop = True)

    return event_df

def add_button_presses(event_df, trial_type_col = "trial_type", response_col = "RT"):
    """
    Adds button presses to the event dataframe.

    Parameters
    ----------
    event_df : pd.DataFrame
        Dataframe containing the events.
    
    trial_type_col : str
        Name of the column containing the trial types.

    response_col : str
        Name of the column containing the response times.
    
    Returns
    -------
    event_df : pd.DataFrame
        Dataframe containing the events with button presses added.
    """
    # get the indices of the button presses
    button_img_indices = event_df.index[event_df[trial_type_col] == "IMG_BI"].tolist()

    for index in button_img_indices:
        response_time = event_df.loc[index, response_col]
        onset = response_time + event_df.loc[index, "onset"]
        
        if not np.isnan(onset): # not including missed button presses where RT is NaN
            new_row = pd.DataFrame({"onset": [onset], "duration": [0], "trial_type": ["button_press"]})
            event_df = pd.concat([event_df, new_row], ignore_index=True)

    event_df = event_df.sort_values(by=["onset"])
    return event_df

def update_trial_type(trial_type):
    if trial_type in ['IMG_PO', 'IMG_PS']:
        return "positive"
    elif trial_type in ['IMG_NO', 'IMG_NS']:
        return "negative"
    elif trial_type == "IMG_BI":
        return "IMG_button"
    else:
        return trial_type

def count_button_presses(event_df):
    """
    Counts the number of button presses in the event dataframe.

    Parameters
    ----------
    event_df : pd.DataFrame
        Dataframe containing the events.

    Returns
    -------
    int
        Count of button presses.
    """
    return event_df[event_df['trial_type'] == 'button_press'].shape[0]


# Load data and count button presses

if __name__ == "__main__":
    bids_dir = Path("/work/816119/InSpePosNegData/BIDS_2023E")  # Replace with your BIDS directory path
    subjects = ["0116", "0117", "0118", "0119", "0120", "0121", "0122", "0123"]
    runs = [1, 2, 3, 4, 5, 6]

    # Dictionary to store button press counts
    button_press_data = {}

    for subject in subjects:
        button_press_data[subject] = []
        for run in runs:
            event_path = bids_dir / f"sub-{subject}" / "func" / f"sub-{subject}_task-boldinnerspeech_run-{run}_echo-1_events.tsv"  # Replace 'yourtask' with your actual task name
            event_df = load_prep_events(event_path)
            button_press_count = count_button_presses(event_df)
            button_press_data[subject].append(button_press_count)

    # Plotting the results
    # Determine the grid size
    num_subjects = len(subjects)
    cols = 3  # You can adjust this as needed
    rows = num_subjects // cols + (num_subjects % cols > 0)

    # Create a figure with subplots
    fig, axs = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axs = axs.flatten()  # Flatten in case of only one row

    # Plot data for each subject
    for i, subject in enumerate(subjects):
        axs[i].bar(runs, button_press_data[subject], color='skyblue')
        axs[i].set_xlabel('Run')
        axs[i].set_ylabel('Number of Button Presses')
        axs[i].set_title(f'Subject {subject}')
        axs[i].set_xticks(runs)

    # Hide any unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].set_visible(False)


    # Add a grand title
    fig.suptitle('Number of Button Presses per Subject per Trial', fontsize=16)

    # Adjust layout to accommodate the grand title
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('figs/button_presses_per_subject_per_trial.png', dpi=300)
