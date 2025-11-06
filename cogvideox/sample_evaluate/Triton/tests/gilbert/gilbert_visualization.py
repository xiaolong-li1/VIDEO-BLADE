#!/usr/bin/env python3
"""
Gilbert Curve Rearrangement Visualization

This script visualizes how the Gilbert 3D space-filling curve rearranges elements
and analyzes whether neighboring elements in memory remain spatially adjacent
after the rearrangement.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import sys
import os

# Add the Triton directory to path to import gilbert3d
sys.path.append('/workspace/Triton')

# Import the gilbert3d function (assuming it exists)
try:
    from utils.gilbert3d import gilbert3d
except ImportError:
    print("Warning: Could not import gilbert3d. Using placeholder implementation.")
    def gilbert3d(width, height, depth):
        """Placeholder gilbert3d implementation - returns sequential coordinates"""
        coords = []
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    coords.append((x, y, z))
        return coords

class GilbertVisualizer:
    def __init__(self, width=45, height=30, depth=13, text_length=226):
        self.width = width
        self.height = height
        self.depth = depth
        self.text_length = text_length
        self.total_elements = width * height * depth
        
        # Generate Gilbert curve mapping
        self._generate_mappings()
        
    def _generate_mappings(self):
        """Generate the Gilbert curve coordinate mappings."""
        print(f"Generating Gilbert mappings for {self.width}x{self.height}x{self.depth} cube...")
        
        coord_to_index = {}
        gilbert_coords = []
        
        index = 0
        for x, y, z in gilbert3d(self.width, self.height, self.depth):
            coord_index = x + self.width * (y + self.height * z)
            coord_to_index[coord_index] = index
            gilbert_coords.append((x, y, z))
            index += 1
        
        # Create mapping arrays
        self.original_order2gilbert_order = [0] * self.total_elements
        self.gilbert_order2original_order = [0] * self.total_elements
        self.gilbert_coords = gilbert_coords
        
        for coord_idx, gilbert_idx in coord_to_index.items():
            self.original_order2gilbert_order[coord_idx] = gilbert_idx
            self.gilbert_order2original_order[gilbert_idx] = coord_idx
            
        print(f"Generated {len(self.gilbert_coords)} coordinate mappings")
    
    def coord_to_index(self, x, y, z):
        """Convert 3D coordinates to linear index."""
        return x + self.width * (y + self.height * z)
    
    def index_to_coord(self, idx):
        """Convert linear index to 3D coordinates."""
        z = idx // (self.width * self.height)
        remainder = idx % (self.width * self.height)
        y = remainder // self.width
        x = remainder % self.width
        return x, y, z
    
    def analyze_spatial_locality(self, sample_size=1000):
        """Analyze spatial locality of Gilbert rearrangement."""
        print("\nAnalyzing spatial locality...")
        
        # Sample random consecutive pairs in Gilbert order
        np.random.seed(42)
        sample_indices = np.random.choice(
            range(self.total_elements - 1), 
            min(sample_size, self.total_elements - 1), 
            replace=False
        )
        
        distances = []
        original_distances = []
        
        for i in sample_indices:
            # Get consecutive Gilbert indices
            gilbert_idx1 = i
            gilbert_idx2 = i + 1
            
            # Get corresponding original indices
            orig_idx1 = self.gilbert_order2original_order[gilbert_idx1]
            orig_idx2 = self.gilbert_order2original_order[gilbert_idx2]
            
            # Convert to 3D coordinates
            coord1 = self.index_to_coord(orig_idx1)
            coord2 = self.index_to_coord(orig_idx2)
            
            # Calculate Euclidean distance
            dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))
            distances.append(dist)
            
            # For comparison, calculate distance between consecutive original indices
            orig_coord1 = self.index_to_coord(i)
            orig_coord2 = self.index_to_coord(i + 1)
            orig_dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(orig_coord1, orig_coord2)))
            original_distances.append(orig_dist)
        
        return np.array(distances), np.array(original_distances)
    
    def visualize_rearrangement(self, num_points=500):
        """Visualize the Gilbert rearrangement effect."""
        print(f"\nCreating visualization with {num_points} sample points...")
        
        # Sample points for visualization
        np.random.seed(42)
        sample_indices = np.random.choice(self.total_elements, num_points, replace=False)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 8))
        
        # Original order visualization
        ax1 = fig.add_subplot(131, projection='3d')
        colors = plt.cm.viridis(np.linspace(0, 1, num_points))
        
        for i, orig_idx in enumerate(sample_indices):
            x, y, z = self.index_to_coord(orig_idx)
            ax1.scatter(x, y, z, c=[colors[i]], s=20, alpha=0.6)
        
        ax1.set_title('Original Linear Order\n(Sequential memory layout)', fontsize=12)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Gilbert order visualization
        ax2 = fig.add_subplot(132, projection='3d')
        
        # Sort sample indices by their Gilbert order
        gilbert_sorted_indices = sorted(sample_indices, 
                                      key=lambda x: self.original_order2gilbert_order[x])
        
        for i, orig_idx in enumerate(gilbert_sorted_indices):
            x, y, z = self.index_to_coord(orig_idx)
            ax2.scatter(x, y, z, c=[colors[i]], s=20, alpha=0.6)
        
        ax2.set_title('Gilbert Curve Order\n(Space-filling curve)', fontsize=12)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # Distance analysis subplot
        ax3 = fig.add_subplot(133)
        
        distances, original_distances = self.analyze_spatial_locality()
        
        ax3.hist(distances, bins=50, alpha=0.7, label='Gilbert Order', density=True)
        ax3.hist(original_distances, bins=50, alpha=0.7, label='Original Order', density=True)
        ax3.set_xlabel('Euclidean Distance Between Consecutive Elements')
        ax3.set_ylabel('Density')
        ax3.set_title('Spatial Distance Distribution\n(Consecutive Memory Positions)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/workspace/gilbert_visualization.png', dpi=300, bbox_inches='tight')
        print("Visualization saved as gilbert_visualization.png")
        
        return fig
    
    def print_locality_analysis(self):
        """Print detailed locality analysis."""
        distances, original_distances = self.analyze_spatial_locality()
        
        print("\n" + "="*60)
        print("GILBERT REARRANGEMENT SPATIAL LOCALITY ANALYSIS")
        print("="*60)
        print(f"Cube dimensions: {self.width} × {self.height} × {self.depth}")
        print(f"Total elements: {self.total_elements:,}")
        print(f"Text length: {self.text_length}")
        print()
        
        print("Distance between consecutive elements in memory:")
        print(f"Gilbert Order - Mean: {distances.mean():.3f}, Std: {distances.std():.3f}")
        print(f"Original Order - Mean: {original_distances.mean():.3f}, Std: {original_distances.std():.3f}")
        print()
        
        # Analyze percentage of close neighbors
        close_threshold = 2.0  # Within 2 units distance
        gilbert_close = (distances <= close_threshold).mean() * 100
        original_close = (original_distances <= close_threshold).mean() * 100
        
        print(f"Percentage of consecutive pairs within {close_threshold} units:")
        print(f"Gilbert Order: {gilbert_close:.1f}%")
        print(f"Original Order: {original_close:.1f}%")
        print()
        
        improvement = gilbert_close / original_close if original_close > 0 else float('inf')
        print(f"Gilbert curve improves spatial locality by {improvement:.2f}x")
        print("="*60)

def main():
    """Main function to run the visualization."""
    print("Gilbert 3D Curve Visualization")
    print("="*50)
    
    # Use the same parameters as in cogvideo_newattn.py
    width = 45
    height = 30  
    depth = 13
    text_length = 226
    
    # Create visualizer
    visualizer = GilbertVisualizer(width, height, depth, text_length)
    
    # Generate and save visualization
    fig = visualizer.visualize_rearrangement(num_points=1000)
    
    # Print analysis
    visualizer.print_locality_analysis()
    
    # Show plot if running interactively
    try:
        plt.show()
    except:
        print("Note: Running in non-interactive mode. Plot saved to file.")

if __name__ == "__main__":
    main()