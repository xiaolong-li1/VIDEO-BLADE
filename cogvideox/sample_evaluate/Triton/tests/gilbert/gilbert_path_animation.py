#!/usr/bin/env python3
"""
Gilbert Curve Path Animation

Creates animated videos showing how memory access patterns differ
between original linear order and Gilbert curve order in 3D space.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import sys
import os

# Add the Triton directory to path
sys.path.append('/workspace/Triton')

try:
    from utils.gilbert3d import gilbert3d
except ImportError:
    print("Warning: Could not import gilbert3d. Using placeholder implementation.")
    def gilbert3d(width, height, depth):
        """Placeholder implementation - returns simple sequential order"""
        coords = []
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    coords.append((x, y, z))
        return coords

class GilbertPathAnimator:
    def __init__(self, width=8, height=6, depth=4):  # Smaller for clearer visualization
        self.width = width
        self.height = height
        self.depth = depth
        self.total_elements = width * height * depth
        
        print(f"Creating path animation for {width}x{height}x{depth} cube ({self.total_elements} elements)")
        
        # Generate Gilbert curve mapping
        self._generate_mappings()
        
    def _generate_mappings(self):
        """Generate Gilbert curve coordinate mappings."""
        coord_to_index = {}
        self.gilbert_coords = []
        
        index = 0
        for x, y, z in gilbert3d(self.width, self.height, self.depth):
            coord_index = x + self.width * (y + self.height * z)
            coord_to_index[coord_index] = index
            self.gilbert_coords.append((x, y, z))
            index += 1
        
        # Create mapping arrays
        self.original_order2gilbert_order = [0] * self.total_elements
        self.gilbert_order2original_order = [0] * self.total_elements
        
        for coord_idx, gilbert_idx in coord_to_index.items():
            self.original_order2gilbert_order[coord_idx] = gilbert_idx
            self.gilbert_order2original_order[gilbert_idx] = coord_idx
    
    def index_to_coord(self, idx):
        """Convert linear index to 3D coordinates."""
        z = idx // (self.width * self.height)
        remainder = idx % (self.width * self.height)
        y = remainder // self.width
        x = remainder % self.width
        return x, y, z
    
    def create_path_animation(self, order_type='both', max_points=None, fps=10):
        """
        Create animated visualization of memory access paths.
        
        Args:
            order_type: 'original', 'gilbert', or 'both'
            max_points: Maximum number of points to show (None for all)
            fps: Frames per second for animation
        """
        if max_points is None:
            max_points = min(self.total_elements, 200)  # Limit for performance
        
        # Prepare coordinate sequences
        if order_type in ['original', 'both']:
            original_coords = [self.index_to_coord(i) for i in range(max_points)]
        
        if order_type in ['gilbert', 'both']:
            gilbert_coords = []
            for gilbert_idx in range(max_points):
                orig_idx = self.gilbert_order2original_order[gilbert_idx]
                gilbert_coords.append(self.index_to_coord(orig_idx))
        
        # Create figure and subplots
        if order_type == 'both':
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': '3d'})
            axes = [ax1, ax2]
            titles = ['Original Linear Order\n(Sequential Memory Access)', 
                     'Gilbert Curve Order\n(Space-Filling Access)']
            coord_sequences = [original_coords, gilbert_coords]
        else:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            axes = [ax]
            if order_type == 'original':
                titles = ['Original Linear Order (Sequential Memory Access)']
                coord_sequences = [original_coords]
            else:
                titles = ['Gilbert Curve Order (Space-Filling Access)']
                coord_sequences = [gilbert_coords]
        
        # Initialize plots
        lines = []
        points = []
        texts = []
        
        for i, ax in enumerate(axes):
            ax.set_xlim(0, self.width-1)
            ax.set_ylim(0, self.height-1)
            ax.set_zlim(0, self.depth-1)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(titles[i], fontsize=12, pad=20)
            
            # Initialize empty line and point plots
            line, = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.7, label='Path')
            point = ax.scatter([], [], [], c='red', s=100, alpha=0.8, label='Current')
            text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            lines.append(line)
            points.append(point)
            texts.append(text)
        
        # Animation function
        def animate(frame):
            if frame >= max_points:
                return lines + points + texts
            
            for i, (line, point, text, coords) in enumerate(zip(lines, points, texts, coord_sequences)):
                # Get coordinates up to current frame
                current_coords = coords[:frame+1]
                
                if current_coords:
                    # Update path line
                    xs, ys, zs = zip(*current_coords)
                    line.set_data_3d(xs, ys, zs)
                    
                    # Update current point
                    current_x, current_y, current_z = current_coords[-1]
                    point._offsets3d = ([current_x], [current_y], [current_z])
                    
                    # Update text
                    text.set_text(f'Step: {frame+1}/{max_points}\nPosition: ({current_x}, {current_y}, {current_z})')
            
            return lines + points + texts
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=max_points+10, 
            interval=1000//fps, blit=False, repeat=True
        )
        
        return fig, anim
    
    def save_animations(self, max_points=100, fps=5):
        """Save both original and Gilbert animations as videos."""
        print(f"Creating animations with {max_points} points at {fps} fps...")
        
        try:
            # Create both animations
            print("Creating original order animation...")
            fig1, anim1 = self.create_path_animation('original', max_points, fps)
            
            print("Saving original_path.mp4...")
            anim1.save('/workspace/original_path.mp4', writer='ffmpeg', fps=fps, bitrate=1800)
            plt.close(fig1)
            
            print("Creating Gilbert order animation...")
            fig2, anim2 = self.create_path_animation('gilbert', max_points, fps)
            
            print("Saving gilbert_path.mp4...")
            anim2.save('/workspace/gilbert_path.mp4', writer='ffmpeg', fps=fps, bitrate=1800)
            plt.close(fig2)
            
            print("Creating comparison animation...")
            fig3, anim3 = self.create_path_animation('both', max_points, fps)
            
            print("Saving gilbert_comparison.mp4...")
            anim3.save('/workspace/gilbert_comparison.mp4', writer='ffmpeg', fps=fps, bitrate=1800)
            plt.close(fig3)
            
            print("All animations saved successfully!")
            
        except Exception as e:
            print(f"Error creating video with ffmpeg: {e}")
            print("Trying with pillow writer...")
            
            try:
                # Fallback to GIF
                fig1, anim1 = self.create_path_animation('original', max_points, fps)
                anim1.save('/workspace/original_path.gif', writer='pillow', fps=fps)
                plt.close(fig1)
                
                fig2, anim2 = self.create_path_animation('gilbert', max_points, fps)
                anim2.save('/workspace/gilbert_path.gif', writer='pillow', fps=fps)
                plt.close(fig2)
                
                fig3, anim3 = self.create_path_animation('both', max_points, fps)
                anim3.save('/workspace/gilbert_comparison.gif', writer='pillow', fps=fps)
                plt.close(fig3)
                
                print("Animations saved as GIFs!")
                
            except Exception as e2:
                print(f"Error creating GIFs: {e2}")
                print("Saving static frames instead...")
                self._save_static_comparison(max_points)
    
    def _save_static_comparison(self, max_points):
        """Save static comparison images showing the complete paths."""
        # Original path
        original_coords = [self.index_to_coord(i) for i in range(max_points)]
        
        # Gilbert path
        gilbert_coords = []
        for gilbert_idx in range(max_points):
            orig_idx = self.gilbert_order2original_order[gilbert_idx]
            gilbert_coords.append(self.index_to_coord(orig_idx))
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': '3d'})
        
        # Plot original path
        if original_coords:
            xs, ys, zs = zip(*original_coords)
            ax1.plot(xs, ys, zs, 'b-', linewidth=2, alpha=0.7)
            ax1.scatter(xs, ys, zs, c=range(len(xs)), cmap='viridis', s=20)
        
        ax1.set_xlim(0, self.width-1)
        ax1.set_ylim(0, self.height-1)
        ax1.set_zlim(0, self.depth-1)
        ax1.set_title('Original Linear Order\n(Sequential Memory Access)', fontsize=12)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Plot Gilbert path
        if gilbert_coords:
            xs, ys, zs = zip(*gilbert_coords)
            ax2.plot(xs, ys, zs, 'r-', linewidth=2, alpha=0.7)
            ax2.scatter(xs, ys, zs, c=range(len(xs)), cmap='viridis', s=20)
        
        ax2.set_xlim(0, self.width-1)
        ax2.set_ylim(0, self.height-1)
        ax2.set_zlim(0, self.depth-1)
        ax2.set_title('Gilbert Curve Order\n(Space-Filling Access)', fontsize=12)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        plt.tight_layout()
        plt.savefig('/workspace/gilbert_path_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Static comparison saved as gilbert_path_comparison.png")

def main():
    """Main function to create and save animations."""
    print("Gilbert 3D Path Animation Creator")
    print("=" * 50)
    
    # Create animator with smaller dimensions for clearer visualization
    animator = GilbertPathAnimator(width=8, height=6, depth=4)
    
    # Save animations
    animator.save_animations(max_points=80, fps=8)
    
    print("\nAnimation files created:")
    print("- gilbert_comparison.mp4/gif - Side-by-side comparison")
    print("- original_path.mp4/gif - Original linear order only") 
    print("- gilbert_path.mp4/gif - Gilbert curve order only")
    print("- gilbert_path_comparison.png - Static comparison image")

if __name__ == "__main__":
    main()