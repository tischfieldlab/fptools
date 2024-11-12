# Fiber Photometry Analysis: `fptools`
Codes/Notebooks to analyze Fiber Photometry data

See also related package for analysing event data from TDT and MedAssociate systems: [med-associates-utils](https://github.com/tischfieldlab/med-associates-utils)

## About
This package allows you to load data produced by TDT Synapse software, and eases working with the resulting data. Also includes a growing library of analysis routines.


## Install
We recommend you use anaconda virtual environments. Two main ways to install. The first, create the virtual environment, and install this package directly (more useful on "production" systems). The second, clone the repository, and then pass to anaconda the `environment.yaml` file during environment creation (more useful for development).
```
conda create -n fptools python=3.12
pip install git+https://github.com/tischfieldlab/fptools.git
```
OR
```
git clone https://github.com/tischfieldlab/fptools.git
cd fptools
conda env create -f environment.yml
```

## Usage
Please checkout the jupyter notebooks available in the `notebooks` directory.

### Loading Data
Please see the notebook [01_Data_Loading.ipynb](notebooks/01_Data_Loading.ipynb) for a full example of data loading and manipulation basics.

To load data from the tanks produced by the TDT software, use the function `load_data()` in the `fptools.io` module. `load_data()` takes a parameter `tank_path` which is the directory to search recursively for blocks.
```
> from fptools.io import load_data
> sessions = load_data(r'C:\path\to\data\tank')
100%|██████████| 30/30 [00:00<00:00, 2068.67it/s]
```
This will return to you a `SessionCollection` object, which in many ways behaves like a python list, but with some extra functionality.

We can ask the `SessionCollection` to describe itself, which will tell us the number of sessions, as well their contents.
```
> sessions.describe()
Number of sessions: 36

Signals present in data with counts:
(36) "Dopamine"
(36) "Isosbestic"

Epocs present in data with counts:
(36) "Cam1"
(36) "P1SC"
(36) "UnNP"
(36) "URM_"
(36) "Nose"
(36) "Tick"
(36) "RNP"
(36) "RMG"
```

### Working with `SessionCollection`

#### Filter Sessions
Assuming we added a genotype to each session metadata, we could filter to only WT sessions via the following code:
```
wt_sessions = sessions.filter(lambda s: s.metadata['genotype'] == 'WT')
```

#### 