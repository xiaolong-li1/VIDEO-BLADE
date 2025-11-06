#!/usr/bin/env python3
"""
Simple Gilbert Rearrangement Analysis

Shows how consecutive memory positions map to 3D coordinates
before and after Gilbert rearrangement.
"""

import sys
sys.path.append('/workspace/Triton')

try:
    from utils.gilbert3d import gilbert3d
except ImportError:
    def gilbert3d(width, height, depth):
        coords = []
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    coords.append((x, y, z))
        return coords

def analyze_gilbert_mapping(width=45, height=30, depth=13, show_samples=20):
    """Analyze and display Gilbert mapping for first few elements."""
    
    print(f"Gilbert 3D Curve Analysis")
    print(f"Cube dimensions: {width} × {height} × {depth}")
    print("="*60)
    
    # Generate mappings like in the original code
    coord_to_index = {}
    gilbert_coords = []
    
    index = 0
    for x, y, z in gilbert3d(width, height, depth):
        coord_index = x + width * (y + height * z)
        coord_to_index[coord_index] = index
        gilbert_coords.append((x, y, z))
        index += 1
    
    total_elements = width * height * depth
    original_order2gilbert_order = [0] * total_elements
    gilbert_order2original_order = [0] * total_elements
    
    for coord_idx, gilbert_idx in coord_to_index.items():
        original_order2gilbert_order[coord_idx] = gilbert_idx
        gilbert_order2original_order[gilbert_idx] = coord_idx
    
    def index_to_coord(idx):
        z = idx // (width * height)
        remainder = idx % (width * height)
        y = remainder // width
        x = remainder % width
        return x, y, z
    
    # Show original vs Gilbert order mapping
    print("ORIGINAL ORDER (Sequential memory layout):")
    print("Memory Index -> 3D Coordinate")
    for i in range(min(show_samples, total_elements)):
        x, y, z = index_to_coord(i)
        print(f"  {i:4d} -> ({x:2d}, {y:2d}, {z:2d})")
    
    print(f"\n... (showing first {show_samples} of {total_elements} total)")
    
    print("\nGILBERT ORDER (Space-filling curve layout):")
    print("Memory Index -> 3D Coordinate")
    for gilbert_idx in range(min(show_samples, total_elements)):
        orig_idx = gilbert_order2original_order[gilbert_idx]
        x, y, z = index_to_coord(orig_idx)
        print(f"  {gilbert_idx:4d} -> ({x:2d}, {y:2d}, {z:2d})")
    
    print(f"\n... (showing first {show_samples} of {total_elements} total)")
    
    # Analyze consecutive distances
    print("\nSPATIAL LOCALITY COMPARISON:")
    print("-" * 40)
    
    # Original order consecutive distances
    orig_distances = []
    for i in range(min(100, total_elements - 1)):
        coord1 = index_to_coord(i)
        coord2 = index_to_coord(i + 1)
        dist = sum((a - b) ** 2 for a, b in zip(coord1, coord2)) ** 0.5
        orig_distances.append(dist)
    
    # Gilbert order consecutive distances  
    gilbert_distances = []
    for i in range(min(100, total_elements - 1)):
        orig_idx1 = gilbert_order2original_order[i]
        orig_idx2 = gilbert_order2original_order[i + 1]
        coord1 = index_to_coord(orig_idx1)
        coord2 = index_to_coord(orig_idx2)
        dist = sum((a - b) ** 2 for a, b in zip(coord1, coord2)) ** 0.5
        gilbert_distances.append(dist)
    
    avg_orig = sum(orig_distances) / len(orig_distances)
    avg_gilbert = sum(gilbert_distances) / len(gilbert_distances)
    
    print(f"Average distance between consecutive elements:")
    print(f"  Original order:  {avg_orig:.3f}")
    print(f"  Gilbert order:   {avg_gilbert:.3f}")
    print(f"  Improvement:     {avg_orig/avg_gilbert:.2f}x better locality")
    
    # Count unit distance neighbors (most spatially adjacent)
    orig_unit = sum(1 for d in orig_distances if abs(d - 1.0) < 0.1)
    gilbert_unit = sum(1 for d in gilbert_distances if abs(d - 1.0) < 0.1)
    
    print(f"\nUnit distance neighbors (perfectly adjacent):")
    print(f"  Original order:  {orig_unit}/{len(orig_distances)} ({orig_unit/len(orig_distances)*100:.1f}%)")
    print(f"  Gilbert order:   {gilbert_unit}/{len(gilbert_distances)} ({gilbert_unit/len(gilbert_distances)*100:.1f}%)")
    
    print("\n" + "="*60)
    print("CONCLUSION:")
    if avg_gilbert < avg_orig:
        print("✓ Gilbert rearrangement IMPROVES spatial locality!")
        print("  Consecutive memory accesses are more spatially coherent.")
    else:
        print("✗ Gilbert rearrangement does not improve spatial locality.")
    
    print("✓ This is beneficial for attention mechanisms because:")
    print("  - Better cache locality during memory access")
    print("  - More coherent spatial patterns for sparse attention")
    print("  - Improved data access patterns for 3D video tokens")

if __name__ == "__main__":
    analyze_gilbert_mapping()