"""
This script is designed for generating plots used for sanity checks of fMRI data.
It loads first-level models and plots a specified contrast for all subjects.
"""

from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import nilearn
from nilearn import plotting
from nilearn.glm import threshold_stats_img
import numpy as np


def fetch_first_level_models(path: Path, return_subject_ids=False):
    """
    Loads the first level models from the given path. Assumes that the models are pickled and no other files are present in the directory.

    Parameters
    ----------
    path : Path
        Path to the directory containing the first level models.
    
    Returns
    -------
    flms : list
        List of first level models.
    """
    # Get all .pkl files in the directory
    flm_files = sorted([f for f in path.iterdir() if f.is_file() and f.suffix == ".pkl"])

    # Load models using list comprehension and context manager
    flms = [pickle.load(open(model, 'rb')) for model in flm_files]

    # Return models and subject IDs if requested
    if return_subject_ids:
        subject_ids = [model.stem[-4:] for model in flm_files]
        return flms, subject_ids
    else:
        return flms

def clean_contrast_name(contrast:str):
    """
    Cleans and formats the contrast name for plotting.
    """
    contrast = contrast.replace("_", " ")
    contrast = contrast.title()

    return contrast


def plot_contrast_subject_level(flm, subject_id, ax, threshold = False, contrast = "button_press", output_type = "z_score"):
    """
    Calculates and plots the contrast for the given first level model.

    Parameters
    ----------
    flm : FirstLevelModel
        First level model.
    subject_id : stre
        Subject ID.
    threshold : bool, optional
        if True, a bonferroni corrected threshold is applied. The default is False.
    ax : matplotlib.axes
        Axis to plot on. Defaults to None. If None, a new figure is created.
    contrast : str, optional
        Contrast to calculate and plot. The default is "button_press".
    
    Returns
    -------
    None.

    """

    contrast_map  = flm.compute_contrast(contrast, output_type = output_type)

    if threshold:
        contrast_map, threshold = threshold_stats_img(
            contrast_map, 
            alpha=0.05, 
            height_control='bonferroni')

    plotting.plot_glass_brain(
        contrast_map,
        colorbar=True,
        plot_abs=False, 
        cmap='PiYG_r',
        axes=ax)
    
    ax.set_title(f"Subject {subject_id}")

    
def plot_contrast_all_subjects(flms, subject_ids, threshold=False, save_path=None, contrast="button_press", output_type="z_score"):
    """
    Plots a given contrast for all subjects in the given list of first level models.

    Parameters
    ----------
    flms : list
        List of first level models.
    subject_ids : list
        List of subject IDs.
    threshold : bool, optional
        If True, a Bonferroni corrected threshold is applied. Default is False.
    save_path : str, optional
        Path to save the figure to. Default is None.
    contrast : str, optional
        Contrast to calculate and plot. Default is "button_press".
    output_type : str, optional
        Type of output to plot (e.g., "z_score"). Default is "z_score".

    Returns
    -------
    None.
    """
    nrows = int(np.ceil(len(flms) / 2))
    ncols = 2 if len(flms) > 1 else 1  # Use 1 column if there's only one plot
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 12))

    # Ensure axes is a 2D array for consistent indexing
    if nrows == 1:
        axes = np.array([axes])

    for i, (flm, subject_id) in enumerate(zip(flms, subject_ids)):
        ax = axes[i // ncols, i % ncols]
        plot_contrast_subject_level(flm, subject_id, ax, threshold, contrast=contrast, output_type=output_type)

    # Add super title in bold
    fig.suptitle(f"{clean_contrast_name(contrast)}", fontweight="bold", fontsize=20)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)


# Main execution block
if __name__ == "__main__":
    # Directly set the path to the first level models
    flm_path = Path("/work/807746/emma_folder/notebooks/fMRI/flms")
    flms, subject_ids = fetch_first_level_models(flm_path, return_subject_ids=True)

    # Check if any models were loaded
    if not flms:
        print("No first level models found. Please check the file path.")
        exit()

    # Set the output path to the specified directory
    output_path = Path("/work/807746/emma_folder/notebooks/fMRI/project_repo/fig")

    if not output_path.exists():
        output_path.mkdir(parents=True)

    plot_contrast_all_subjects(
        flms, subject_ids,threshold=True,
        save_path=output_path / "button_press_contrast.png",
        contrast="button_press",
        output_type = "z_score"
    )
