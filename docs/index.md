# fptools
### Tools for Fiber Photometry Analysis
Codes/Notebooks to analyze Fiber Photometry and behavioral data

~~See also related package for analyzing event data from TDT and MedAssociate systems: [med-associates-utils](https://github.com/tischfieldlab/med-associates-utils)~~ The `med-associates-utils` package functionality has now largely been incorporated into `fptools`.

## About
This package allows you to load fiber photometry and behavioral data, with first-class support for data produced by TDT Synapse software and Med-Associates systems. The package eases working with the resulting data via a unified interface. Also includes a growing library of analysis routines.

## Features
- built-in support for TDT tanks and Med-Associates files
- interfaces to customize loading of arbitrary data
- metadata management and propagation
- powerful visualizations with only a few lines of code
- more being added with time

## Installation
We recommend you use anaconda virtual environments.

There are two main ways to install this package. The first, create the virtual environment, and install this package directly (more useful on "production" systems). The second, clone the repository, and then pass to anaconda the `environment.yaml` file during environment creation (more useful for development).
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

## Support
For technical inquiries specific to this package, please [open an Issue](https://github.com/tischfieldlab/fptools/issues)
with a description of your problem or request.

For general usage, see the [main website](https://github.com/tischfieldlab/fptools).

Other questions? Reach out to `thackray@rutgers.edu`.


