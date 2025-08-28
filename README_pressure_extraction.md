# Pressure Data Extraction from .rwo Files

This repository contains code to extract pressure data from reservoir simulation `.rwo` files and organize it into numpy arrays for analysis.

## Overview

The `.rwo` file contains reservoir simulation results with pressure values for each grid cell (i, j, k) at different time steps. The extraction code parses this file and creates a structured numpy array that can be easily used for analysis and visualization.

## Files

### Main Scripts

1. **`extract_pressure_rwo.py`** - Main extraction script that reads the .rwo file and creates numpy arrays
2. **`verify_pressure_data.py`** - Verification script that loads and checks the extracted data
3. **`example_usage.py`** - Examples showing various ways to use the extracted pressure data

### Data Files

- **`data/case1_PRES.rwo`** - Input file containing pressure simulation results (76MB)
- **`results/pressure_data.npy`** - Extracted pressure data as numpy array
- **`results/time_info.npy`** - Time step information
- **`results/time_info.txt`** - Human-readable time information

## Data Structure

The extracted pressure data has the following structure:

- **Shape**: `(n_i, n_j, n_k, n_time)`
- **Dimensions**:
  - `n_i = 107` (I-direction cells)
  - `n_j = 117` (J-direction cells) 
  - `n_k = 79` (K-direction layers)
  - `n_time = 6` (time steps)

### Time Steps

1. **Step 1**: 0 days (2030-Jan-01) - Initial condition
2. **Step 2**: 3,652 days (2040-Jan-01) - 10 years
3. **Step 3**: 7,305 days (2050-Jan-01) - 20 years
4. **Step 4**: 10,957 days (2060-Jan-01) - 30 years
5. **Step 5**: 189,926 days (2550-Jan-01) - 520 years
6. **Step 6**: 372,547 days (3050-Jan-01) - 1,020 years

### Data Characteristics

- **Units**: Pressure in kPa
- **Range**: 0.00 to 32,418.50 kPa
- **Sparsity**: 73.4% of cells contain zero values
- **Non-zero cells**: 263,528 cells contain actual pressure data per time step

## Usage

### 1. Extract Pressure Data

```bash
python extract_pressure_rwo.py
```

This will:
- Parse the `data/case1_PRES.rwo` file
- Extract pressure values for all grid cells and time steps
- Save the data as `results/pressure_data.npy`
- Save time information as `results/time_info.npy` and `results/time_info.txt`

### 2. Verify Extracted Data

```bash
python verify_pressure_data.py
```

This will:
- Load the extracted data
- Display data statistics
- Create visualization plots
- Save plots to the `results/` directory

### 3. Run Examples

```bash
python example_usage.py
```

This demonstrates various analysis techniques:
- Basic array access and slicing
- Spatial analysis at specific time steps
- Temporal analysis for specific regions
- Statistical analysis across the dataset
- Data export in different formats

## Data Access Examples

### Basic Access

```python
import numpy as np

# Load the data
pressure_data = np.load("results/pressure_data.npy")
time_info = np.load("results/time_info.npy", allow_pickle=True).item()

# Get dimensions
n_i, n_j, n_k, n_time = pressure_data.shape

# Access specific cell pressure over time
i, j, k = 50, 50, 39
pressure_history = pressure_data[i, j, k, :]

# Access pressure at specific time and location
pressure_at_time = pressure_data[:, :, :, 0]  # First time step
```

### Spatial Analysis

```python
# Get pressure distribution at last time step
final_pressure = pressure_data[:, :, :, -1]

# Calculate statistics
min_pressure = np.min(final_pressure)
max_pressure = np.max(final_pressure)
mean_pressure = np.mean(final_pressure)

# Find cells with highest pressure
max_indices = np.where(final_pressure == np.max(final_pressure))
```

### Temporal Analysis

```python
# Analyze pressure evolution in a region
region_pressure = pressure_data[40:60, 40:60, 30:50, :]
mean_pressure_time = np.mean(region_pressure, axis=(0, 1, 2))

# Calculate pressure change rates
time_days = np.array(time_info['time_values'])
pressure_change_rate = np.gradient(mean_pressure_time, time_days)
```

## Output Files

After running the extraction, you'll find these files in the `results/` directory:

- **`pressure_data.npy`** - Main pressure array (45MB)
- **`time_info.npy`** - Time step information
- **`time_info.txt`** - Human-readable time information
- **`pressure_evolution_slice.png`** - 2D pressure evolution plot
- **`pressure_3d_surface_t0.png`** - 3D pressure surface plot
- **`pressure_2d_K40_T0.csv`** - 2D pressure slice export
- **`cell_pressure_time_series.csv`** - Cell pressure time series export

## Requirements

- Python 3.6+
- NumPy
- Matplotlib (for visualization)

## Notes

- The data contains many zero values (73.4% sparsity), which is common in reservoir simulation results
- Non-zero pressure values are concentrated in specific regions of the grid
- The simulation covers a very long time period (1,020 years)
- Pressure values increase over time in most active regions
- The grid dimensions (107×117×79) represent a substantial 3D reservoir model

## Troubleshooting

- Ensure the `data/case1_PRES.rwo` file exists and is readable
- Check that you have sufficient disk space for the output files (~45MB for pressure data)
- If matplotlib plotting fails, the data extraction will still work
- The script handles large files efficiently by processing line by line

## Customization

You can modify the scripts to:
- Extract different properties (temperature, saturation, etc.)
- Change the output format (CSV, HDF5, etc.)
- Add additional analysis functions
- Modify the visualization options
- Extract data for specific regions or time periods 