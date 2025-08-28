#!/usr/bin/env python3
"""
Example usage of the extracted pressure data.
This script demonstrates various ways to work with the pressure array.
"""

import numpy as np
import matplotlib.pyplot as plt

def load_data():
    """Load the extracted pressure data."""
    pressure_data = np.load("results/pressure_data.npy")
    time_info = np.load("results/time_info.npy", allow_pickle=True).item()
    return pressure_data, time_info

def example_1_basic_access():
    """Example 1: Basic array access and slicing."""
    print("=== Example 1: Basic Array Access ===")
    
    pressure_data, time_info = load_data()
    
    # Get array dimensions
    n_i, n_j, n_k, n_time = pressure_data.shape
    print(f"Array dimensions: I={n_i}, J={n_j}, K={n_k}, Time={n_time}")
    
    # Access specific cell pressure over time
    i, j, k = 50, 50, 39  # Example cell
    pressure_history = pressure_data[i, j, k, :]
    
    print(f"\nPressure history for cell (I={i+1}, J={j+1}, K={k+1}):")
    for t, (time, date) in enumerate(zip(time_info['time_values'], time_info['time_dates'])):
        print(f"  {date}: {pressure_history[t]:.2f} kPa")
    
    # Calculate pressure change
    initial_pressure = pressure_history[0]
    final_pressure = pressure_history[-1]
    pressure_change = final_pressure - initial_pressure
    
    print(f"\nPressure change: {pressure_change:.2f} kPa")
    print(f"Percentage change: {(pressure_change/initial_pressure)*100:.1f}%" if initial_pressure > 0 else "N/A (initial pressure is 0)")

def example_2_spatial_analysis():
    """Example 2: Spatial analysis at a specific time."""
    print("\n=== Example 2: Spatial Analysis ===")
    
    pressure_data, time_info = load_data()
    
    # Analyze pressure distribution at the last time step
    time_idx = -1
    pressure_slice = pressure_data[:, :, :, time_idx]
    
    print(f"Pressure analysis for {time_info['time_dates'][time_idx]}:")
    print(f"  Min pressure: {np.min(pressure_slice):.2f} kPa")
    print(f"  Max pressure: {np.max(pressure_slice):.2f} kPa")
    print(f"  Mean pressure: {np.mean(pressure_slice):.2f} kPa")
    print(f"  Std deviation: {np.std(pressure_slice):.2f} kPa")
    
    # Find cells with highest pressure
    max_pressure = np.max(pressure_slice)
    max_indices = np.where(pressure_slice == max_pressure)
    print(f"\nHighest pressure: {max_pressure:.2f} kPa at:")
    for i, j, k in zip(max_indices[0], max_indices[1], max_indices[2]):
        print(f"  I={i+1}, J={j+1}, K={k+1}")
    
    # Calculate pressure gradient in I direction for middle layer
    k_middle = pressure_data.shape[2] // 2
    pressure_2d = pressure_data[:, :, k_middle, time_idx]
    
    # Calculate gradient (simple finite difference)
    gradient_i = np.gradient(pressure_2d, axis=0)
    gradient_j = np.gradient(pressure_2d, axis=1)
    
    print(f"\nPressure gradients in middle layer (K={k_middle+1}):")
    print(f"  Max I-gradient: {np.max(np.abs(gradient_i)):.2f} kPa/cell")
    print(f"  Max J-gradient: {np.max(np.abs(gradient_j)):.2f} kPa/cell")

def example_3_temporal_analysis():
    """Example 3: Temporal analysis for specific regions."""
    print("\n=== Example 3: Temporal Analysis ===")
    
    pressure_data, time_info = load_data()
    
    # Analyze pressure evolution in a region (e.g., center of the grid)
    i_start, i_end = 40, 60
    j_start, j_end = 40, 60
    k_start, k_end = 30, 50
    
    region_pressure = pressure_data[i_start:i_end, j_start:j_end, k_start:k_end, :]
    
    print(f"Pressure evolution in region:")
    print(f"  I: {i_start+1} to {i_end}")
    print(f"  J: {j_start+1} to {j_end}")
    print(f"  K: {k_start+1} to {k_end}")
    
    # Calculate mean pressure for the region over time
    mean_pressure_time = np.mean(region_pressure, axis=(0, 1, 2))
    
    print(f"\nMean pressure in region over time:")
    for t, (time, date) in enumerate(zip(time_info['time_values'], time_info['time_dates'])):
        print(f"  {date}: {mean_pressure_time[t]:.2f} kPa")
    
    # Calculate pressure change rate
    time_days = np.array(time_info['time_values'])
    pressure_change_rate = np.gradient(mean_pressure_time, time_days)
    
    print(f"\nPressure change rates:")
    for t, (time, date) in enumerate(zip(time_info['time_values'], time_info['time_dates'])):
        print(f"  {date}: {pressure_change_rate[t]:.4f} kPa/day")

def example_4_statistical_analysis():
    """Example 4: Statistical analysis across the entire dataset."""
    print("\n=== Example 4: Statistical Analysis ===")
    
    pressure_data, time_info = load_data()
    
    # Overall statistics
    print("Overall dataset statistics:")
    print(f"  Total cells: {pressure_data.size}")
    print(f"  Non-zero cells: {np.count_nonzero(pressure_data)}")
    print(f"  Zero cells: {np.sum(pressure_data == 0)}")
    print(f"  Sparsity: {(np.sum(pressure_data == 0) / pressure_data.size) * 100:.1f}%")
    
    # Statistics by time step
    print(f"\nStatistics by time step:")
    for t, (time, date) in enumerate(zip(time_info['time_values'], time_info['time_dates'])):
        time_slice = pressure_data[:, :, :, t]
        non_zero = time_slice[time_slice > 0]
        if len(non_zero) > 0:
            print(f"  {date}:")
            print(f"    Non-zero cells: {len(non_zero)}")
            print(f"    Mean (non-zero): {np.mean(non_zero):.2f} kPa")
            print(f"    Std (non-zero): {np.std(non_zero):.2f} kPa")
        else:
            print(f"  {date}: All cells are zero")

def example_5_data_export():
    """Example 5: Export data in different formats."""
    print("\n=== Example 5: Data Export ===")
    
    pressure_data, time_info = load_data()
    
    # Export a 2D slice as CSV
    k_layer = 39  # Middle layer
    time_idx = 0  # First time step
    
    pressure_2d = pressure_data[:, :, k_layer, time_idx]
    
    # Save as CSV
    np.savetxt(f'results/pressure_2d_K{k_layer+1}_T{time_idx}.csv', pressure_2d, 
               delimiter=',', fmt='%.2f')
    
    print(f"Exported 2D pressure slice to: results/pressure_2d_K{k_layer+1}_T{time_idx}.csv")
    
    # Export time series for specific cells
    cells_of_interest = [(50, 50, 39), (25, 25, 20), (75, 75, 60)]
    
    time_series_data = []
    for i, j, k in cells_of_interest:
        cell_pressure = pressure_data[i, j, k, :]
        time_series_data.append(cell_pressure)
    
    time_series_array = np.array(time_series_data).T
    header = f"I,J,K,{','.join([f'Time_{i+1}' for i in range(len(time_info['time_values']))])}"
    
    np.savetxt('results/cell_pressure_time_series.csv', time_series_array, 
               delimiter=',', fmt='%.2f', header=header, comments='')
    
    print("Exported cell pressure time series to: results/cell_pressure_time_series.csv")

def main():
    """Run all examples."""
    print("Pressure Data Analysis Examples")
    print("=" * 40)
    
    try:
        example_1_basic_access()
        example_2_spatial_analysis()
        example_3_temporal_analysis()
        example_4_statistical_analysis()
        example_5_data_export()
        
        print("\n" + "=" * 40)
        print("All examples completed successfully!")
        print("Check the results/ directory for exported files.")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 