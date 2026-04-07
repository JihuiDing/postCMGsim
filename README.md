# Post-processing for CMG Coupled Flow-geomechanical Simulation (postCMGsim)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
The postCMGsim repo extracts results from CMG coupled flow-geomechanical simulation and performs fault slip modeling and importance sampling. The implementation in this repo is the second part of a physics-based workflow for seismic hazard assessment for carbon capture and storage (CCS) by integrating geomodeling, coupled flow-geomechanical simulation, fault slip modeling, global sensitivity analysis, and importance sampling.


## Author
Jihui Ding (jihuid@stanford.edu)

## How to Use

1. Perform geomodeling in Petrel and create grid files for CMG simulations

2. Run a Petrel Uncertainty & Optimization workflow to generate geological realizations that contain finit files, extract realizations (e.g., porosity and permeability) and convert them into CMG compatible format.

3. Set up a CMG dat file template in which all constant reservoir properties and simulation settings. In the template, set realization properties as variables. Finally, generate dat files by replacing variables with realizations (e.g., file paths to 3D properties, sampled values for flow and geomechanical properties).

4. Run simulations on these dat files in batches (e.g., batch of 100) on a computing cluster.

## Acknowledgements

This project builds on implementations from [Yuna Li/pyCCUS](https://github.com/AndyStudio/pyCCUS-public).

