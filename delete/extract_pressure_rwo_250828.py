#!/usr/bin/env python3
"""
Extract pressure data from .rwo file and organize into numpy array.
The .rwo file contains reservoir simulation results with pressure for i,j,k cells at different times.
"""

import numpy as np
import re
from typing import Tuple, List
import os

def parse_rwo_file(filepath: str) -> Tuple[np.ndarray, List[float], List[str]]:
    """
    Parse the .rwo file and extract pressure data.
    
    Args:
        filepath: Path to the .rwo file
        
    Returns:
        pressure_array: numpy array with shape (n_i, n_j, n_k, n_time)
        time_values: List of time values in days
        time_dates: List of time dates as strings
    """
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    print(f"Parsing file: {filepath}")
    
    # Initialize lists to store data
    time_values = []
    time_dates = []
    pressure_data = []
    
    current_time = None
    current_k = None
    current_j = None
    current_data = []
    
    with open(filepath, 'r') as file:
        for line_num, line in enumerate(file):
            line = line.strip()
            
            # Check for time step
            if line.startswith("**  TIME ="):
                # Save previous time step data if exists
                if current_time is not None and current_data:
                    pressure_data.append({
                        'time': current_time,
                        'date': time_dates[-1] if time_dates else '',
                        'data': current_data.copy()
                    })
                
                # Parse new time step
                match = re.match(r'\*\*  TIME = (\d+(?:\.\d+)?)\s+(.+)', line)
                if match:
                    current_time = float(match.group(1))
                    current_date = match.group(2)
                    time_values.append(current_time)
                    time_dates.append(current_date)
                    current_data = []
                    print(f"Processing time step: {current_time} ({current_date})")
                
            # Check for K, J header
            elif line.startswith("** K ="):
                match = re.match(r'\*\* K = (\d+), J = (\d+)', line)
                if match:
                    current_k = int(match.group(1))
                    current_j = int(match.group(2))
                    current_data.append({
                        'k': current_k,
                        'j': current_j,
                        'values': []
                    })
            
            # Parse pressure values (skip empty lines and headers)
            elif line and not line.startswith("**") and not line.startswith("RESULTS") and not line.startswith("PRES"):
                # Split line and convert to float
                try:
                    values = [float(x) for x in line.split()]
                    if current_data and 'values' in current_data[-1]:
                        current_data[-1]['values'].extend(values)
                except ValueError:
                    # Skip lines that can't be parsed as numbers
                    continue
    
    # Add the last time step
    if current_time is not None and current_data:
        pressure_data.append({
            'time': current_time,
            'date': time_dates[-1] if time_dates else '',
            'data': current_data
        })
    
    print(f"Found {len(time_values)} time steps")
    
    # Determine grid dimensions
    k_values = set()
    j_values = set()
    i_count = 0
    
    for time_data in pressure_data:
        for cell_data in time_data['data']:
            k_values.add(cell_data['k'])
            j_values.add(cell_data['j'])
            if i_count == 0:
                i_count = len(cell_data['values'])
    
    n_k = max(k_values)
    n_j = max(j_values)
    n_i = i_count
    n_time = len(time_values)
    
    print(f"Grid dimensions: I={n_i}, J={n_j}, K={n_k}, Time={n_time}")
    
    # Create the pressure array
    pressure_array = np.zeros((n_i, n_j, n_k, n_time))
    
    # Fill the array
    for time_idx, time_data in enumerate(pressure_data):
        for cell_data in time_data['data']:
            k = cell_data['k'] - 1  # Convert to 0-based indexing
            j = cell_data['j'] - 1  # Convert to 0-based indexing
            
            if len(cell_data['values']) == n_i:
                pressure_array[:, j, k, time_idx] = cell_data['values']
            else:
                print(f"Warning: Expected {n_i} values, got {len(cell_data['values'])} for K={k+1}, J={j+1}, Time={time_data['time']}")
    
    return pressure_array, time_values, time_dates

def save_pressure_data(pressure_array: np.ndarray, time_values: List[float], 
                      time_dates: List[str], output_dir: str = "results"):
    """
    Save the extracted pressure data to files.
    
    Args:
        pressure_array: Pressure data array
        time_values: List of time values
        time_dates: List of time dates
        output_dir: Output directory
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save numpy array
    np.save(os.path.join(output_dir, "pressure_data.npy"), pressure_array)
    
    # Save time information
    time_info = {
        'time_values': time_values,
        'time_dates': time_dates,
        'shape': pressure_array.shape
    }
    np.save(os.path.join(output_dir, "time_info.npy"), time_info)
    
    # Save as text file for human readability
    with open(os.path.join(output_dir, "time_info.txt"), 'w') as f:
        f.write("Time Step Information:\n")
        f.write("=" * 30 + "\n")
        for i, (time, date) in enumerate(zip(time_values, time_dates)):
            f.write(f"Step {i+1}: Time = {time} days, Date = {date}\n")
        f.write(f"\nArray shape: {pressure_array.shape}\n")
        f.write(f"Grid dimensions: I={pressure_array.shape[0]}, J={pressure_array.shape[1]}, K={pressure_array.shape[2]}\n")
    
    print(f"Data saved to {output_dir}/")
    print(f"Pressure array shape: {pressure_array.shape}")
    print(f"Time steps: {len(time_values)}")

def main():
    """Main function to extract pressure data from .rwo file."""
    
    # File path
    rwo_file = "data/case1_PRES.rwo"
    
    try:
        # Parse the .rwo file
        pressure_array, time_values, time_dates = parse_rwo_file(rwo_file)
        
        # Save the data
        save_pressure_data(pressure_array, time_values, time_dates)
        
        # Print some statistics
        print("\nData Statistics:")
        print(f"Min pressure: {np.min(pressure_array):.2f} kPa")
        print(f"Max pressure: {np.max(pressure_array):.2f} kPa")
        print(f"Mean pressure: {np.mean(pressure_array):.2f} kPa")
        
        # Show sample data for first time step
        print(f"\nSample data for first time step (K=1, J=1):")
        print(pressure_array[:5, 0, 0, 0])
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 