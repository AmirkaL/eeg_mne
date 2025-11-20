# EEG Go/No-Go Data Processing

Automated processing pipeline for EEG data from go/no-go experiments using MNE-Python.

## Features

- Automatic file detection and processing
- Individual participant analysis
- Group-level grand average analysis
- ERP component analysis (N2, P3)
- Visualization (topographic maps, comparison plots, butterfly plots)
- Export to MNE format (Evoked and Raw) for further analysis

## Requirements

- Python 3.7+
- pandas
- openpyxl
- mne
- numpy
- matplotlib

## Installation

```bash
pip install pandas openpyxl mne numpy matplotlib
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare your data

Create folders for each participant:
```
participant1/
  - data_1.xlsx      (GO condition data)
  - data_2.xlsx      (NO-GO condition data)
  - data_beh.xlsx    (behavioral data)
participant2/
  - ...
```

### 2. Process all participants

```bash
python process_all_participants.py
```

This script will:
- Find all `participant*` folders
- Process each participant's data
- Create individual results in each folder

### 3. Group analysis

```bash
python group_analysis.py
```

This script will:
- Load all processed participants
- Create grand average (group-level average)
- Generate group visualizations
- Calculate group statistics

## Output Files

### Individual Results (in each participant folder)

- `erp_*.png` - ERP visualizations
- `evoked_go-ave.fif` - GO condition data (MNE format)
- `evoked_nogo-ave.fif` - NO-GO condition data (MNE format)
- `go_raw.fif`, `nogo_raw.fif` - Raw format files (for MNELab)
- `erp_statistics.csv` - Component statistics

### Group Results (in root folder)

- `grand_average_comparison.png` - Main comparison plot
- `grand_average_go_topomap.png` - GO topographic maps
- `grand_average_nogo_topomap.png` - NO-GO topographic maps
- `grand_average_difference_topomap.png` - Difference wave maps
- `grand_average_butterfly.png` - All channels plot
- `grand_average_*.fif` - Group-level data files
- `grand_average_statistics.csv` - Group statistics

## Data Format

### Input Excel Files

**EEG Data Files** (`*_1.xlsx`, `*_2.xlsx`):
- Column `Time (ms)` - time points in milliseconds
- EEG channel columns (Fp1, Fp2, Fz, Cz, Pz, etc.)
- Standard 10-20 system electrode names

**Behavioral Data File** (`*_beh.xlsx`):
- Column `Name` - stimulus names
- Column `Total` - total number of trials
- Column `Averaged` - number of averaged epochs
- Column `RT1` - reaction time (optional)

## Scripts

- `process_eeg.py` - Processes data for a single participant
- `process_all_participants.py` - Batch processes all participants
- `group_analysis.py` - Creates group-level grand average

## Notes

- The pipeline automatically detects GO and NO-GO conditions based on trial counts
- Baseline correction is applied (-256 to 0 ms)
- ERP components N2 (200-300 ms) and P3 (300-500 ms) are analyzed
- All results are automatically saved in appropriate formats

## License

This project is provided as-is for research purposes.

