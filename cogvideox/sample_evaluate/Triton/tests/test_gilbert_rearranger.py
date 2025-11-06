#!/usr/bin/env python3
"""
Test script to verify GilbertRearranger implementation correctness
"""

import torch
import numpy as np
import sys
import os

# Add the Triton directory to path
sys.path.append('/workspace/Triton')

from utils.gilbert3d import gilbert3d

class GilbertRearranger:
    """基于 Gilbert 曲线的序列重排器，用于视频和文本数据的重新排列。"""
    def __init__(self, width, height, depth, text_length=224):
        self.width = width
        self.height = height
        self.depth = depth
        self.total_elements = width * height * depth
        self.text_length = text_length

        coord_to_index = self._gilbert3d_with_index(width, height, depth)
        original_order2gilbert_order = [0] * self.total_elements
        gilbert_order2original_order = [0] * self.total_elements

        for coord_idx, org_idx in coord_to_index.items():
            original_order2gilbert_order[org_idx] = coord_idx
            gilbert_order2original_order[coord_idx] = org_idx

        self.original_order2gilbert_order = torch.tensor(original_order2gilbert_order, dtype=torch.long)
        self.gilbert_order2original_order = torch.tensor(gilbert_order2original_order, dtype=torch.long)

    def _gilbert3d_with_index(self, width, height, depth):
        """生成 Gilbert 曲线的坐标到索引映射。"""
        coord_to_index = {}
        index = 0
        def coord_to_index_func(x, y, z):
            return x + width * (y + height * z)
        for x, y, z in gilbert3d(width, height, depth):
            coord_index = coord_to_index_func(x, y, z)
            coord_to_index[coord_index] = index
            index += 1
        return coord_to_index
        
    def rearrange(self, q, k, v):
        """将 q、k、v 张量的视频部分按 Gilbert 曲线顺序重排。"""
        seq_dim = -2
        text_part_q, video_part_q = q[..., :self.text_length, :], q[..., self.text_length:, :]
        text_part_k, video_part_k = k[..., :self.text_length, :], k[..., self.text_length:, :]
        text_part_v, video_part_v = v[..., :self.text_length, :], v[..., self.text_length:, :]

        q_rearranged = video_part_q.index_select(seq_dim, self.original_order2gilbert_order)
        k_rearranged = video_part_k.index_select(seq_dim, self.original_order2gilbert_order)
        v_rearranged = video_part_v.index_select(seq_dim, self.original_order2gilbert_order)

        return (torch.cat((q_rearranged,text_part_q), dim=seq_dim),
                torch.cat((k_rearranged,text_part_k), dim=seq_dim),
                torch.cat((v_rearranged,text_part_v), dim=seq_dim))
    
    def reversed_rearrange(self, out):
        """将输出张量的视频部分从 Gilbert 曲线顺序恢复到原始顺序。"""
        seq_dim = -2
        video_part,text_part= out[..., :-self.text_length, :], out[..., -self.text_length:, :]
        out_reversed = video_part.index_select(seq_dim, self.gilbert_order2original_order)
        return torch.cat((text_part, out_reversed), dim=seq_dim)

def test_gilbert_rearranger():
    """Test the GilbertRearranger implementation"""
    print("Testing GilbertRearranger Implementation")
    print("=" * 50)
    
    # Test parameters
    width, height, depth = 8, 6, 4
    text_length = 10
    batch_size = 2
    num_heads = 4
    head_dim = 64
    
    total_video_tokens = width * height * depth
    total_seq_length = text_length + total_video_tokens
    
    print(f"Test dimensions: {width}x{height}x{depth}")
    print(f"Video tokens: {total_video_tokens}, Text tokens: {text_length}")
    print(f"Total sequence length: {total_seq_length}")
    
    # Create test rearranger
    rearranger = GilbertRearranger(width, height, depth, text_length)
    
    print(f"Mapping arrays created successfully")
    print(f"original_order2gilbert_order length: {len(rearranger.original_order2gilbert_order)}")
    print(f"gilbert_order2original_order length: {len(rearranger.gilbert_order2original_order)}")
    
    # Test 1: Check mapping consistency
    print("\nTest 1: Mapping Consistency")
    print("-" * 30)
    
    # Verify that the mappings are inverse of each other
    errors = 0
    for i in range(total_video_tokens):
        gilbert_idx = rearranger.original_order2gilbert_order[i].item()
        recovered_orig_idx = rearranger.gilbert_order2original_order[gilbert_idx].item()
        if recovered_orig_idx != i:
            print(f"  Error: orig {i} -> gilbert {gilbert_idx} -> recovered {recovered_orig_idx}")
            errors += 1
    
    if errors == 0:
        print("  ✓ Mapping consistency test passed")
    else:
        print(f"  ✗ Found {errors} mapping errors")
    
    # Test 2: Index range validation
    print("\nTest 2: Index Range Validation")
    print("-" * 30)
    
    orig_min, orig_max = rearranger.original_order2gilbert_order.min().item(), rearranger.original_order2gilbert_order.max().item()
    gilbert_min, gilbert_max = rearranger.gilbert_order2original_order.min().item(), rearranger.gilbert_order2original_order.max().item()
    
    print(f"  original_order2gilbert_order range: [{orig_min}, {orig_max}]")
    print(f"  gilbert_order2original_order range: [{gilbert_min}, {gilbert_max}]")
    
    range_ok = (orig_min == 0 and orig_max == total_video_tokens - 1 and 
                gilbert_min == 0 and gilbert_max == total_video_tokens - 1)
    
    if range_ok:
        print("  ✓ Index ranges are correct")
    else:
        print("  ✗ Index ranges are incorrect")
    
    # Test 3: Create test tensors and test rearrangement
    print("\nTest 3: Rearrangement Functionality")  
    print("-" * 30)
    
    # Create test tensors
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, total_seq_length, head_dim)
    k = torch.randn(batch_size, num_heads, total_seq_length, head_dim)  
    v = torch.randn(batch_size, num_heads, total_seq_length, head_dim)
    
    print(f"  Input tensor shapes: q={q.shape}, k={k.shape}, v={v.shape}")
    
    try:
        # Test rearrange
        q_rearranged, k_rearranged, v_rearranged = rearranger.rearrange(q, k, v)
        print(f"  Rearranged tensor shapes: q={q_rearranged.shape}, k={k_rearranged.shape}, v={v_rearranged.shape}")
        
        # Verify shapes are unchanged
        if (q_rearranged.shape == q.shape and 
            k_rearranged.shape == k.shape and 
            v_rearranged.shape == v.shape):
            print("  ✓ Rearranged tensor shapes are correct")
        else:
            print("  ✗ Rearranged tensor shapes are incorrect")
        
        # Test that text parts are preserved
        text_preserved = (torch.equal(q[..., :text_length, :], q_rearranged[..., -text_length:, :]) and
                         torch.equal(k[..., :text_length, :], k_rearranged[..., -text_length:, :]) and
                         torch.equal(v[..., :text_length, :], v_rearranged[..., -text_length:, :]))
        
        if text_preserved:
            print("  ✓ Text parts are correctly preserved")
        else:
            print("  ✗ Text parts are not preserved correctly")
        
    except Exception as e:
        print(f"  ✗ Rearrange failed: {e}")
        return
    
    # Test 4: Test reverse rearrangement
    print("\nTest 4: Reverse Rearrangement")
    print("-" * 30)
    
    try:
        # Create a dummy output tensor (same shape as input)
        out = torch.randn_like(q_rearranged)
        
        # Test reverse rearrangement
        out_reversed = rearranger.reversed_rearrange(out)
        
        print(f"  Reversed output shape: {out_reversed.shape}")
        
        if out_reversed.shape == out.shape:
            print("  ✓ Reverse rearrangement shape is correct")
        else:
            print("  ✗ Reverse rearrangement shape is incorrect")
        
    except Exception as e:
        print(f"  ✗ Reverse rearrangement failed: {e}")
        return
    
    # Test 5: Round-trip consistency
    print("\nTest 5: Round-trip Consistency")  
    print("-" * 30)
    
    try:
        # Start with original tensors
        q_orig = q.clone()
        k_orig = k.clone()
        v_orig = v.clone()
        
        # Rearrange
        q_rearranged, k_rearranged, v_rearranged = rearranger.rearrange(q_orig, k_orig, v_orig)
        
        # Create a simple "attention output" by just using q_rearranged as output
        fake_output = q_rearranged.clone()
        
        # Reverse rearrangement
        output_recovered = rearranger.reversed_rearrange(fake_output)
        
        # Check if we can recover the original
        # Note: The text and video parts are swapped in the process, so we need to account for that
        video_part_orig = q_orig[..., text_length:, :]
        text_part_orig = q_orig[..., :text_length, :]
        expected_output = torch.cat((text_part_orig, video_part_orig), dim=-2)
        
        if torch.allclose(output_recovered, expected_output, atol=1e-6):
            print("  ✓ Round-trip consistency test passed")
        else:
            max_diff = (output_recovered - expected_output).abs().max().item()
            print(f"  ✗ Round-trip consistency failed, max diff: {max_diff}")
        
    except Exception as e:
        print(f"  ✗ Round-trip test failed: {e}")
    
    # Test 6: Coordinate mapping verification
    print("\nTest 6: Coordinate Mapping Verification")
    print("-" * 30)
    
    def index_to_coord(idx, width, height):
        z = idx // (width * height)
        remainder = idx % (width * height)
        y = remainder // width
        x = remainder % width
        return x, y, z
    
    # Check first few mappings
    print("  First 10 coordinate mappings:")
    for i in range(min(10, total_video_tokens)):
        gilbert_idx = rearranger.original_order2gilbert_order[i].item()
        orig_coord = index_to_coord(i, width, height)
        print(f"    Original {i} {orig_coord} -> Gilbert position {gilbert_idx}")
    
    # Verify that all indices are used exactly once
    used_gilbert_indices = set(rearranger.original_order2gilbert_order.tolist())
    used_orig_indices = set(rearranger.gilbert_order2original_order.tolist())
    expected_indices = set(range(total_video_tokens))
    
    if used_gilbert_indices == expected_indices and used_orig_indices == expected_indices:
        print("  ✓ All indices are used exactly once")
    else:
        print("  ✗ Some indices are missing or duplicated")
        if used_gilbert_indices != expected_indices:
            print(f"    Gilbert indices missing: {expected_indices - used_gilbert_indices}")
            print(f"    Gilbert indices extra: {used_gilbert_indices - expected_indices}")
        if used_orig_indices != expected_indices:
            print(f"    Original indices missing: {expected_indices - used_orig_indices}")
            print(f"    Original indices extra: {used_orig_indices - expected_indices}")
    
    print("\n" + "=" * 50)
    print("GilbertRearranger test completed!")

def test_edge_cases():
    """Test edge cases and potential issues"""
    print("\nTesting Edge Cases")
    print("=" * 50)
    
    # Test with minimal dimensions
    print("\nTest: Minimal dimensions (2x2x2)")
    try:
        rearranger = GilbertRearranger(2, 2, 2, text_length=2)
        print("  ✓ Minimal dimensions work")
    except Exception as e:
        print(f"  ✗ Minimal dimensions failed: {e}")
    
    # Test with unequal dimensions
    print("\nTest: Unequal dimensions (3x5x7)")
    try:
        rearranger = GilbertRearranger(3, 5, 7, text_length=5)
        total_elements = 3 * 5 * 7
        if len(rearranger.original_order2gilbert_order) == total_elements:
            print("  ✓ Unequal dimensions work")
        else:
            print("  ✗ Unequal dimensions have wrong mapping size")
    except Exception as e:
        print(f"  ✗ Unequal dimensions failed: {e}")
    
    # Test with large text_length
    print("\nTest: Large text length")
    try:
        width, height, depth = 4, 4, 4
        video_tokens = width * height * depth
        large_text_length = video_tokens * 2  # Larger than video tokens
        
        q = torch.randn(1, 1, video_tokens + large_text_length, 32)
        k = torch.randn(1, 1, video_tokens + large_text_length, 32)
        v = torch.randn(1, 1, video_tokens + large_text_length, 32)
        
        rearranger = GilbertRearranger(width, height, depth, text_length=large_text_length)
        q_r, k_r, v_r = rearranger.rearrange(q, k, v)
        
        if q_r.shape == q.shape:
            print("  ✓ Large text length works")
        else:
            print("  ✗ Large text length produces wrong shapes")
            
    except Exception as e:
        print(f"  ✗ Large text length failed: {e}")

if __name__ == "__main__":
    test_gilbert_rearranger()
    test_edge_cases()