#!/usr/bin/env python3
"""
Extract grid coordinates from .fgrid file and save to numpy array.
The .fgrid file contains grid cell definitions with i,j,k indices and corner coordinates.
This script extracts the average x,y,z coordinates for each cell and organizes them into a numpy array.
"""

import numpy as np
import re
import os
from typing import Tuple

def extract_fgrid_coordinates(filepath: str) -> Tuple[np.ndarray, dict]:
    """
    Extract grid coordinates from .fgrid file and organize into numpy array.
    
    Args:
        filepath: Path to the .fgrid file
        
    Returns:
        coordinates_array: numpy array with shape (n_i, n_j, n_k, 3) containing [x_ave, y_ave, z_ave]
        grid_info: Dictionary containing grid dimensions and metadata
    """
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    print(f"Parsing .fgrid file: {filepath}")
    
    # Initialize variables
    grid_info = {}
    coordinates_data = []
    
    with open(filepath, 'r') as file:
        content = file.read()
    
    # Extract grid dimensions
    dims_match = re.search(r"'DIMENS  '\s+3\s+'INTE'\s*\n\s*(\d+)\s+(\d+)\s+(\d+)", content)
    if dims_match:
        n_i, n_j, n_k = map(int, dims_match.groups())
        grid_info['dimensions'] = (n_i, n_j, n_k)
        print(f"Grid dimensions: I={n_i}, J={n_j}, K={n_k}")
    else:
        raise ValueError("Could not find grid dimensions in .fgrid file")
    
    # Extract map units
    units_match = re.search(r"'MAPUNITS'\s+1\s+'CHAR'\s*\n\s*'([^']+)'", content)
    if units_match:
        grid_info['units'] = units_match.group(1)
        print(f"Map units: {grid_info['units']}")
    
    # Split content into COORDS sections
    sections = content.split("'COORDS  '")
    
    print(f"Processing {len(sections)-1} coordinate sections...")
    
    for i, section in enumerate(sections[1:], 1):
        # Extract COORDS values (i, j, k, cell_id)
        coords_match = re.search(r"'INTE'\s*\n\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)", section)
        if coords_match:
            i, j, k, cell_id = map(int, coords_match.groups())
        else:
            continue
        
        # Extract CORNERS values
        corners_match = re.search(r"'CORNERS '\s+24\s+'REAL'\s*\n(.*?)(?='COORDS  '|$)", section, re.DOTALL)
        if corners_match:
            corners_text = corners_match.group(1).strip()
            # Extract all numeric values from corners
            corners_values = re.findall(r'-?\d+\.?\d*E?[+-]?\d*', corners_text)
            corners_values = [float(val) for val in corners_values]
            
            if len(corners_values) >= 24:
                # Calculate averages for x, y, z
                # Corners are organized as: [x1,y1,z1, x2,y2,z2, x3,y3,z3, x4,y4,z4, x5,y5,z5, x6,y6,z6, x7,y7,z7, x8,y8,z8]
                x_ave = np.mean(corners_values[0::3])  # Every 3rd value starting from 0
                y_ave = np.mean(corners_values[1::3])  # Every 3rd value starting from 1
                z_ave = np.mean(corners_values[2::3])  # Every 3rd value starting from 2
                
                coordinates_data.append({
                    'i': i,
                    'j': j,
                    'k': k,
                    'cell_id': cell_id,
                    'x_ave': x_ave,
                    'y_ave': y_ave,
                    'z_ave': z_ave
                })
    
    print(f"Total cells processed: {len(coordinates_data)}")
    
    # Create the coordinates array with shape (n_i, n_j, n_k, 3)
    coordinates_array = np.zeros((n_i, n_j, n_k, 3))
    
    # Fill the array
    for cell_data in coordinates_data:
        i, j, k = cell_data['i'] - 1, cell_data['j'] - 1, cell_data['k'] - 1  # Convert to 0-based indexing
        
        # Check bounds
        if 0 <= i < n_i and 0 <= j < n_j and 0 <= k < n_k:
            coordinates_array[i, j, k, 0] = cell_data['x_ave']  # x coordinate
            coordinates_array[i, j, k, 1] = cell_data['y_ave']  # y coordinate
            coordinates_array[i, j, k, 2] = cell_data['z_ave']  # z coordinate
        else:
            print(f"Warning: Cell indices ({i+1}, {j+1}, {k+1}) out of bounds for grid ({n_i}, {n_j}, {n_k})")
    
    # Check for missing cells
    total_expected = n_i * n_j * n_k
    cells_filled = np.count_nonzero(np.any(coordinates_array != 0, axis=3))
    print(f"Cells filled: {cells_filled}/{total_expected} ({cells_filled/total_expected*100:.1f}%)")
    
    return coordinates_array, grid_info

def save_coordinates_data(coordinates_array: np.ndarray, grid_info: dict, output_dir: str = "results"):
    """
    Save the extracted coordinates data to files.
    
    Args:
        coordinates_array: Coordinates array with shape (n_i, n_j, n_k, 3)
        grid_info: Grid information dictionary
        output_dir: Output directory
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save numpy array
    np.save(os.path.join(output_dir, "grid_coordinates.npy"), coordinates_array)
    
    # Save grid information
    grid_info_file = os.path.join(output_dir, "grid_info.npy")
    np.save(grid_info_file, grid_info)
    
    # Save as text file for human readability
    with open(os.path.join(output_dir, "grid_info.txt"), 'w') as f:
        f.write("Grid Information:\n")
        f.write("=" * 20 + "\n")
        f.write(f"Dimensions: I={grid_info['dimensions'][0]}, J={grid_info['dimensions'][1]}, K={grid_info['dimensions'][2]}\n")
        if 'units' in grid_info:
            f.write(f"Units: {grid_info['units']}\n")
        f.write(f"\nCoordinates array shape: {coordinates_array.shape}\n")
        f.write(f"Coordinates array dtype: {coordinates_array.dtype}\n")
        
        # Calculate statistics
        non_zero_coords = coordinates_array[coordinates_array != 0]
        if len(non_zero_coords) > 0:
            f.write(f"\nCoordinate Statistics:\n")
            f.write(f"  X range: {np.min(coordinates_array[:,:,:,0]):.2f} to {np.max(coordinates_array[:,:,:,0]):.2f}\n")
            f.write(f"  Y range: {np.min(coordinates_array[:,:,:,1]):.2f} to {np.max(coordinates_array[:,:,:,1]):.2f}\n")
            f.write(f"  Z range: {np.min(coordinates_array[:,:,:,2]):.2f} to {np.max(coordinates_array[:,:,:,2]):.2f}\n")
    
    print(f"Data saved to {output_dir}/")
    print(f"Coordinates array shape: {coordinates_array.shape}")

def verify_coordinates_data(coordinates_array: np.ndarray, grid_info: dict):
    """
    Verify the extracted coordinates data.
    
    Args:
        coordinates_array: Coordinates array
        grid_info: Grid information
    """
    
    print("\nData Verification:")
    print(f"  Array shape: {coordinates_array.shape}")
    print(f"  Expected shape: ({grid_info['dimensions'][0]}, {grid_info['dimensions'][1]}, {grid_info['dimensions'][2]}, 3)")
    
    # Check if shapes match
    expected_shape = (grid_info['dimensions'][0], grid_info['dimensions'][1], grid_info['dimensions'][2], 3)
    if coordinates_array.shape == expected_shape:
        print("  ✓ Array shape matches expected dimensions")
    else:
        print("  ✗ Array shape does not match expected dimensions")
        return False
    
    # Check for non-zero values
    non_zero_cells = np.count_nonzero(np.any(coordinates_array != 0, axis=3))
    total_cells = coordinates_array.shape[0] * coordinates_array.shape[1] * coordinates_array.shape[2]
    print(f"  Non-zero coordinate cells: {non_zero_cells}/{total_cells} ({non_zero_cells/total_cells*100:.1f}%)")
    
    # Check coordinate ranges
    x_coords = coordinates_array[:,:,:,0]
    y_coords = coordinates_array[:,:,:,1]
    z_coords = coordinates_array[:,:,:,2]
    
    print(f"  X coordinate range: {np.min(x_coords):.2f} to {np.max(x_coords):.2f}")
    print(f"  Y coordinate range: {np.min(y_coords):.2f} to {np.max(y_coords):.2f}")
    print(f"  Z coordinate range: {np.min(z_coords):.2f} to {np.max(z_coords):.2f}")
    
    return True

def main():
    """Main function to extract grid coordinates from .fgrid file."""
    
    # File path
    fgrid_file = "data/JD_Sula_2025_flow.fgrid"
    
    try:
        # Extract coordinates from .fgrid file
        coordinates_array, grid_info = extract_fgrid_coordinates(fgrid_file)
        
        # Verify the data
        if not verify_coordinates_data(coordinates_array, grid_info):
            print("Warning: Data verification failed!")
        
        # Save the data
        save_coordinates_data(coordinates_array, grid_info)
        
        # Print some sample data
        print(f"\nSample coordinate data:")
        print(f"  Cell (1,1,1): [{coordinates_array[0,0,0,0]:.2f}, {coordinates_array[0,0,0,1]:.2f}, {coordinates_array[0,0,0,2]:.2f}]")
        print(f"  Cell (50,50,39): [{coordinates_array[49,49,38,0]:.2f}, {coordinates_array[49,49,38,1]:.2f}, {coordinates_array[49,49,38,2]:.2f}]")
        print(f"  Cell (107,117,79): [{coordinates_array[106,116,78,0]:.2f}, {coordinates_array[106,116,78,1]:.2f}, {coordinates_array[106,116,78,2]:.2f}]")
        
        print(f"\nExtraction completed successfully!")
        print(f"Coordinates array saved as: results/grid_coordinates.npy")
        print(f"Grid info saved as: results/grid_info.npy")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 