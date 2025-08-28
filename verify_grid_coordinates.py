#!/usr/bin/env python3
"""
Verify and demonstrate the extracted grid coordinates data.
"""

import numpy as np
import matplotlib.pyplot as plt

def load_and_verify_grid_data():
    """Load the extracted grid coordinates and verify its structure."""
    
    try:
        # Load the grid coordinates
        grid_coordinates = np.load("results/grid_coordinates.npy")
        grid_info = np.load("results/grid_info.npy", allow_pickle=True).item()
        
        print("Grid coordinates loaded successfully!")
        print(f"Coordinates array shape: {grid_coordinates.shape}")
        print(f"Grid dimensions: I={grid_info['dimensions'][0]}, J={grid_info['dimensions'][1]}, K={grid_info['dimensions'][2]}")
        print(f"Units: {grid_info.get('units', 'Unknown')}")
        
        # Check data statistics
        print(f"\nCoordinate Statistics:")
        print(f"  X range: {np.min(grid_coordinates[:,:,:,0]):.2f} to {np.max(grid_coordinates[:,:,:,0]):.2f}")
        print(f"  Y range: {np.min(grid_coordinates[:,:,:,1]):.2f} to {np.max(grid_coordinates[:,:,:,1]):.2f}")
        print(f"  Z range: {np.min(grid_coordinates[:,:,:,2]):.2f} to {np.max(grid_coordinates[:,:,:,2]):.2f}")
        
        # Check for any zero values
        zero_coords = np.sum(grid_coordinates == 0)
        total_coords = grid_coordinates.size
        print(f"  Zero coordinates: {zero_coords}/{total_coords} ({zero_coords/total_coords*100:.1f}%)")
        
        return grid_coordinates, grid_info
        
    except Exception as e:
        print(f"Error loading grid data: {e}")
        return None, None

def demonstrate_coordinate_access(grid_coordinates, grid_info):
    """Demonstrate various ways to access coordinate data."""
    
    print("\n=== Coordinate Access Examples ===")
    
    # Get dimensions
    n_i, n_j, n_k, n_coords = grid_coordinates.shape
    
    # Example 1: Access specific cell coordinates
    i, j, k = 50, 50, 39
    cell_coords = grid_coordinates[i, j, k, :]
    print(f"Coordinates for cell (I={i+1}, J={j+1}, K={k+1}):")
    print(f"  X: {cell_coords[0]:.2f} m")
    print(f"  Y: {cell_coords[1]:.2f} m")
    print(f"  Z: {cell_coords[2]:.2f} m")
    
    # Example 2: Get all X coordinates for a specific layer
    k_layer = 39
    x_coords_layer = grid_coordinates[:, :, k_layer, 0]
    print(f"\nX coordinates for layer K={k_layer+1}:")
    print(f"  Shape: {x_coords_layer.shape}")
    print(f"  Range: {np.min(x_coords_layer):.2f} to {np.max(x_coords_layer):.2f} m")
    
    # Example 3: Get coordinate differences between adjacent cells
    i1, j1, k1 = 50, 50, 39
    i2, j2, k2 = 51, 50, 39
    
    coords1 = grid_coordinates[i1, j1, k1, :]
    coords2 = grid_coordinates[i2, j2, k2, :]
    
    dx = coords2[0] - coords1[0]
    dy = coords2[1] - coords1[1]
    dz = coords2[2] - coords1[2]
    
    print(f"\nCoordinate differences between adjacent cells:")
    print(f"  dX: {dx:.2f} m")
    print(f"  dY: {dy:.2f} m")
    print(f"  dZ: {dz:.2f} m")
    
    # Example 4: Calculate cell volumes (approximate)
    # Get cell dimensions from adjacent cells
    i3, j3, k3 = 50, 51, 39
    i4, j4, k4 = 50, 50, 40
    
    coords3 = grid_coordinates[i3, j3, k3, :]
    coords4 = grid_coordinates[i4, j4, k4, :]
    
    dx_cell = abs(coords2[0] - coords1[0])
    dy_cell = abs(coords3[1] - coords1[1])
    dz_cell = abs(coords4[2] - coords1[2])
    
    cell_volume = dx_cell * dy_cell * dz_cell
    print(f"\nApproximate cell volume: {cell_volume:.2f} m³")

def plot_coordinate_slices(grid_coordinates, grid_info):
    """Create visualization plots of coordinate data."""
    
    try:
        print("\nCreating visualization plots...")
        
        # Create a 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Grid Coordinate Visualization', fontsize=16)
        
        # Plot 1: X coordinates for middle layer
        k_middle = grid_coordinates.shape[2] // 2
        x_coords = grid_coordinates[:, :, k_middle, 0]
        im1 = axes[0, 0].imshow(x_coords.T, cmap='viridis', aspect='auto')
        axes[0, 0].set_title(f'X Coordinates - Layer K={k_middle+1}')
        axes[0, 0].set_xlabel('I index')
        axes[0, 0].set_ylabel('J index')
        plt.colorbar(im1, ax=axes[0, 0], label='X (m)')
        
        # Plot 2: Y coordinates for middle layer
        y_coords = grid_coordinates[:, :, k_middle, 1]
        im2 = axes[0, 1].imshow(y_coords.T, cmap='plasma', aspect='auto')
        axes[0, 1].set_title(f'Y Coordinates - Layer K={k_middle+1}')
        axes[0, 1].set_xlabel('I index')
        axes[0, 1].set_ylabel('J index')
        plt.colorbar(im2, ax=axes[0, 1], label='Y (m)')
        
        # Plot 3: Z coordinates for middle layer
        z_coords = grid_coordinates[:, :, k_middle, 2]
        im3 = axes[1, 0].imshow(z_coords.T, cmap='coolwarm', aspect='auto')
        axes[1, 0].set_title(f'Z Coordinates - Layer K={k_middle+1}')
        axes[1, 0].set_xlabel('I index')
        axes[1, 0].set_ylabel('J index')
        plt.colorbar(im3, ax=axes[1, 0], label='Z (m)')
        
        # Plot 4: Z coordinates along I direction for middle J
        j_middle = grid_coordinates.shape[1] // 2
        z_profile = grid_coordinates[:, j_middle, :, 2]
        im4 = axes[1, 1].imshow(z_profile.T, cmap='terrain', aspect='auto')
        axes[1, 1].set_title(f'Z Coordinates - J={j_middle+1} (I-K plane)')
        axes[1, 1].set_xlabel('I index')
        axes[1, 1].set_ylabel('K index')
        plt.colorbar(im4, ax=axes[1, 1], label='Z (m)')
        
        plt.tight_layout()
        plt.savefig('results/grid_coordinates_visualization.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("Plots saved to results/grid_coordinates_visualization.png")
        
    except Exception as e:
        print(f"Warning: Could not create plots: {e}")

def export_coordinate_samples(grid_coordinates, grid_info):
    """Export sample coordinate data in different formats."""
    
    print("\n=== Exporting Sample Data ===")
    
    # Export a 2D slice as CSV
    k_layer = 39  # Middle layer
    x_coords_2d = grid_coordinates[:, :, k_layer, 0]
    y_coords_2d = grid_coordinates[:, :, k_layer, 1]
    z_coords_2d = grid_coordinates[:, :, k_layer, 2]
    
    # Combine into a single array
    coords_2d = np.stack([x_coords_2d, y_coords_2d, z_coords_2d], axis=2)
    
    # Save as CSV
    np.savetxt(f'results/coordinates_2d_K{k_layer+1}.csv', 
               coords_2d.reshape(-1, 3), 
               delimiter=',', 
               fmt='%.2f',
               header='X,Y,Z',
               comments='')
    
    print(f"Exported 2D coordinates slice to: results/coordinates_2d_K{k_layer+1}.csv")
    
    # Export coordinate profiles along different directions
    i_middle = grid_coordinates.shape[0] // 2
    j_middle = grid_coordinates.shape[1] // 2
    
    # I-direction profile
    i_profile = grid_coordinates[i_middle, :, :, :]
    np.savetxt('results/coordinates_profile_I.csv', 
               i_profile.reshape(-1, 3), 
               delimiter=',', 
               fmt='%.2f',
               header='X,Y,Z',
               comments='')
    
    # J-direction profile
    j_profile = grid_coordinates[:, j_middle, :, :]
    np.savetxt('results/coordinates_profile_J.csv', 
               j_profile.reshape(-1, 3), 
               delimiter=',', 
               fmt='%.2f',
               header='X,Y,Z',
               comments='')
    
    print("Exported coordinate profiles to results/ directory")

def main():
    """Main function to verify and demonstrate grid coordinates."""
    
    print("Grid Coordinates Verification and Demonstration")
    print("=" * 50)
    
    # Load and verify data
    grid_coordinates, grid_info = load_and_verify_grid_data()
    
    if grid_coordinates is None:
        return 1
    
    print("\nData verification completed successfully!")
    
    # Demonstrate coordinate access
    demonstrate_coordinate_access(grid_coordinates, grid_info)
    
    # Create visualization plots
    plot_coordinate_slices(grid_coordinates, grid_info)
    
    # Export sample data
    export_coordinate_samples(grid_coordinates, grid_info)
    
    print("\n" + "=" * 50)
    print("Grid coordinates verification completed!")
    print("Check the results/ directory for exported files and plots.")
    
    return 0

if __name__ == "__main__":
    exit(main()) 