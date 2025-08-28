#!/usr/bin/env python3
"""
Verify and demonstrate the extracted pressure data.
"""

import numpy as np
import matplotlib.pyplot as plt

def load_and_verify_data():
    """Load the extracted pressure data and verify its structure."""
    
    try:
        # Load the pressure data
        pressure_data = np.load("results/pressure_data.npy")
        time_info = np.load("results/time_info.npy", allow_pickle=True).item()
        
        print("Data loaded successfully!")
        print(f"Pressure array shape: {pressure_data.shape}")
        print(f"Time steps: {len(time_info['time_values'])}")
        print(f"Grid dimensions: I={pressure_data.shape[0]}, J={pressure_data.shape[1]}, K={pressure_data.shape[2]}")
        
        # Print time information
        print("\nTime steps:")
        for i, (time, date) in enumerate(zip(time_info['time_values'], time_info['time_dates'])):
            print(f"  Step {i+1}: {time} days ({date})")
        
        # Check data statistics
        print(f"\nData Statistics:")
        print(f"  Min pressure: {np.min(pressure_data):.2f} kPa")
        print(f"  Max pressure: {np.max(pressure_data):.2f} kPa")
        print(f"  Mean pressure: {np.mean(pressure_data):.2f} kPa")
        print(f"  Non-zero values: {np.count_nonzero(pressure_data)}")
        print(f"  Zero values: {np.sum(pressure_data == 0)}")
        
        return pressure_data, time_info
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def plot_pressure_slice(pressure_data, time_info, k_layer=0, j_row=0):
    """Plot a 2D slice of pressure data for a specific K and J."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Pressure Evolution: K={k_layer+1}, J={j_row+1}', fontsize=16)
    
    for time_idx in range(6):
        row = time_idx // 3
        col = time_idx % 3
        
        # Extract the 2D slice
        pressure_slice = pressure_data[:, j_row, k_layer, time_idx]
        
        # Plot
        axes[row, col].plot(pressure_slice, 'b-', linewidth=1)
        axes[row, col].set_title(f'Time: {time_info["time_values"][time_idx]:.0f} days\n{time_info["time_dates"][time_idx]}')
        axes[row, col].set_xlabel('I index')
        axes[row, col].set_ylabel('Pressure (kPa)')
        axes[row, col].grid(True, alpha=0.3)
        
        # Add statistics
        mean_p = np.mean(pressure_slice)
        max_p = np.max(pressure_slice)
        axes[row, col].text(0.02, 0.98, f'Mean: {mean_p:.0f}\nMax: {max_p:.0f}', 
                           transform=axes[row, col].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/pressure_evolution_slice.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_pressure_3d_surface(pressure_data, time_info, time_idx=0):
    """Plot a 3D surface plot of pressure for a specific time step."""
    
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create meshgrid for I and J
    i_indices = np.arange(pressure_data.shape[0])
    j_indices = np.arange(pressure_data.shape[1])
    I, J = np.meshgrid(i_indices, j_indices, indexing='ij')
    
    # Extract pressure data for middle K layer and specified time
    k_middle = pressure_data.shape[2] // 2
    pressure_slice = pressure_data[:, :, k_middle, time_idx]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create surface plot
    surf = ax.plot_surface(I, J, pressure_slice, cmap='viridis', alpha=0.8)
    
    ax.set_xlabel('I index')
    ax.set_ylabel('J index')
    ax.set_zlabel('Pressure (kPa)')
    ax.set_title(f'Pressure Surface: K={k_middle+1}, Time={time_info["time_values"][time_idx]:.0f} days\n{time_info["time_dates"][time_idx]}')
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.savefig(f'results/pressure_3d_surface_t{time_idx}.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main function to verify and demonstrate the data."""
    
    print("Loading and verifying pressure data...")
    pressure_data, time_info = load_and_verify_data()
    
    if pressure_data is None:
        return 1
    
    print("\nData verification completed successfully!")
    
    # Demonstrate data access
    print(f"\nExample data access:")
    print(f"  pressure_data[0, 0, 0, 0] = {pressure_data[0, 0, 0, 0]:.2f} kPa")
    print(f"  pressure_data[50, 50, 39, 2] = {pressure_data[50, 50, 39, 2]:.2f} kPa")
    
    # Show some sample slices
    print(f"\nSample pressure values for K=1, J=1, first time step:")
    print(pressure_data[:10, 0, 0, 0])
    
    print(f"\nSample pressure values for K=40, J=60, last time step:")
    print(pressure_data[:10, 59, 39, 5])
    
    # Create plots
    try:
        print("\nCreating visualization plots...")
        plot_pressure_slice(pressure_data, time_info, k_layer=0, j_row=0)
        plot_pressure_3d_surface(pressure_data, time_info, time_idx=0)
        print("Plots saved to results/ directory")
    except Exception as e:
        print(f"Warning: Could not create plots: {e}")
    
    return 0

if __name__ == "__main__":
    exit(main()) 