# Post-processing for CMG Coupled Flow-geomechanical Simulation (postCMGsim)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
The postCMGsim repo extracts results from CMG coupled flow-geomechanical simulation and performs fault slip modeling and importance sampling. The implementation in this repo is the second part of a physics-based workflow for seismic hazard assessment for carbon capture and storage (CCS) by integrating geomodeling, coupled flow-geomechanical simulation, fault slip modeling, global sensitivity analysis, and importance sampling.


## Author
Jihui Ding (jihuid@stanford.edu)

## How to Use

1. Extract grid coordinates and fault IDs

2. Extract CMG simulations results and save them as Numpy arrays

3. Perform fault slip analysis based on either pore pressure or simulated principal stresses

4. Visualize results in 2D or 3D

## Acknowledgements

This project builds on implementations from [pyCCUS](https://github.com/AndyStudio/pyCCUS-public).

