# fMRI Decoding Analysis
The current repository contains the code needed to perform the fMRI analysis related to the portfolio 4 assignment of the course `Advanced Cognitive Neuroscience` at the Master's Program in Cognitive Science, Aarhus University, Fall semester of 2023.

The current analysis was conducted by `Nanna Marie Steenholdt` and `Emma Risgaard Olsen`.

## Analysis Steps 

### Step 1: Set up virtual environment

```
bash setup_venv.sh
```

### Step 2: Activate virtual environment
```
source venv/bin/activate
```

### Step 3: Perform Sanity Checks on brain data

#### Step 3.1: Fit first level models 

```
python src/fit_first_level.py
```

#### Step 3.2: Plot sanity check plots

```
python src/plot_sanity_checks_brain.py
```

#### Step 3.3: Behavioral Sanity Checks

```
python src/plot_sanity_checks_beh.py
```

### Step 4: Analyse positive vs negative inner speech

#### Step 4.1: Compute and visualise contrast btw. positive and negative inner speech

```
python src/contrast_inner_speech.py
```

#### Step 4.2: Compute single trial betas for each participant

```
python src/single_trial_betas.py
```

#### Step 4.3: Perform searchlight decoding analysis 

```
python src/run_searchlight_decoding.py
```

## Project Organisation

The code cannot be run without access to the raw data which cannot be published on GitHub due to GDPR constraints, but all analysis steps can be seen by inspecting the code. The data was structured in the following manner, and running the code requires you to have the data organised in a corresponding folder structure.

```
├── Repository/directory containing the code
├── 816119                   <--- Folder containing fMRI data from several projects
│   └── InSpePosNegData      <--- Parent folder for fMRI data from current project
│       └── BIDS_2023E       <--- All fMRI data from current project
│           └── derivatives  <--- Folder containing preprocessed fMRI data (used in analysis)
│               └── sub-XXXX
│                   └── anat
│                       └── sub-0117_acq-T1sequence_run-1_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz
│                   └── figures
│                   └── func
│                       └── sub-0117_task-boldinnerspeech_run-1_desc-confounds_timeseries.tsv
│                       └── sub-0117_task-boldinnerspeech_run-1_echo-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
│                   └── log  
│               └── sub-XXXX.html <--- report from fMRIPrep explaining all preprocessing steps
│           └── sub-XXXX     <---- raw fMRI data, not preprocessed. Event-files used for analysis.
│               └── anat
│                   └── sub-XXXX_acq-T1sequence_run-1_T1w.json
│                   └── sub-XXXX_acq-T1sequence_run-1_T1w.nii.gz <---- Structural T1
│               └── func
│                   └── sub-XXXX_task-boldinnerspeech_run-1_echo-1_bold.nii.gz <---- fMRI data for run 1
│                   └── sub-XXXX_task-boldinnerspeech_run-1_echo-1_events.tsv  <---- event file for run 1 
│               └── sub-XXXX_scans.tsv
│           └── dataset_description.json
│           └── participants.json
│           └── participants.tsv

```