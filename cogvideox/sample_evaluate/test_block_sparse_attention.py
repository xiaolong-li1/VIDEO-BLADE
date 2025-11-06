#!/usr/bin/env python3
"""
Test script for block sparse attention kernel with backward pass
Tests both correctness and performance compared to torch SDPA baseline
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import math
from typing import Tuple, Optional

# Import the block sparse attention implementation
# from .Triton.kernels.block_sparse_attn_kernel_with_backward import sparse_attention_factory
# Note: The file is named with version suffix _9_10
from Triton.kernels.block_sparse_attn_kernel_with_backward_9_10 import sparse_attention_factory 
from Triton.kernels.block_sparse_attn_kernel_with_backward_10_10 import sparse_attention_factory as sparse_attention_factory_new
# Default block size  
BLOCK_M = 128
BLOCK_N = 128

# Create attention function with correct block size
block_sparse_triton_fn = sparse_attention_factory(BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
block_sparse_triton_fn2= sparse_attention_factory_new(BLOCK_M=BLOCK_M, BLOCK_N=32)

def create_block_mask(seq_len: int, block_size: int, sparsity_pattern: str = "random") -> torch.Tensor:
    """
    Create a block sparse mask for attention.
    
    Args:
        seq_len: Sequence length
        block_size: Size of each block
        sparsity_pattern: "random", "diagonal", "local" or "causal"
    
    Returns:
        Block mask tensor of shape [1, 1, num_blocks_m, num_blocks_n]
        Values: 0=skip, 1=full attention, 2=2x pooled, 4=4x pooled, 8=8x pooled
    """
    num_blocks = seq_len // block_size
    if seq_len % block_size != 0:
        num_blocks += 1
    
    mask = torch.zeros((1, 1, num_blocks, num_blocks), dtype=torch.int32)
    
    if sparsity_pattern == "random":
        # Random sparse pattern with different pooling levels
        rand = torch.rand((num_blocks, num_blocks))
        mask[0, 0] = torch.where(rand < 0.3, 0,  # 30% skip
                                torch.where(rand < 0.5, 1,  # 20% full attention
                                torch.where(rand < 0.7, 2,  # 20% 2x pooled
                                torch.where(rand < 0.9, 4, 8))))  # 20% 4x pooled, 10% 8x pooled
        
    elif sparsity_pattern == "diagonal":
        # Diagonal pattern - full attention on diagonal, pooled attention nearby
        for i in range(num_blocks):
            for j in range(num_blocks):
                dist = abs(i - j)
                if dist == 0:
                    mask[0, 0, i, j] = 1  # Full attention on diagonal
                elif dist == 1:
                    mask[0, 0, i, j] = 2  # 2x pooled for adjacent blocks
                elif dist == 2:
                    mask[0, 0, i, j] = 4  # 4x pooled for nearby blocks
                elif dist <= 4:
                    mask[0, 0, i, j] = 8  # 8x pooled for distant blocks
                # else: 0 (skip for very distant blocks)
                    
    elif sparsity_pattern == "local":
        # Local attention pattern
        window_size = 3  # Attend to 3 blocks on each side
        for i in range(num_blocks):
            start = max(0, i - window_size)
            end = min(num_blocks, i + window_size + 1)
            for j in range(start, end):
                dist = abs(i - j)
                if dist <= 1:
                    mask[0, 0, i, j] = 1  # Full attention for close blocks
                else:
                    mask[0, 0, i, j] = 2  # 2x pooled for nearby blocks
                    
    elif sparsity_pattern == "causal":
        # Causal (lower triangular) with different pooling levels
        for i in range(num_blocks):
            for j in range(i + 1):  # Only attend to previous blocks
                if j == i:
                    mask[0, 0, i, j] = 1  # Full attention for current block
                elif i - j <= 2:
                    mask[0, 0, i, j] = 2  # 2x pooled for recent blocks
                elif i - j <= 5:
                    mask[0, 0, i, j] = 4  # 4x pooled for older blocks
                else:
                    mask[0, 0, i, j] = 8  # 8x pooled for very old blocks
    
    return mask


def torch_sdpa_baseline(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Baseline implementation using torch.nn.functional.scaled_dot_product_attention
    """
    return F.scaled_dot_product_attention(q, k, v, is_causal=False)


def test_baseline_accuracy():
    """Test accuracy against torch SDPA baseline with all masks = 1"""
    print("Testing baseline accuracy (mask all = 1 vs torch SDPA)...")
    
    # Test parameters
    batch_size = 1
    num_heads = 1
    seq_len = 5120
    head_dim = 64
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    
    # Create test tensors
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    
    # Create block mask with all 1s (full attention everywhere)
    num_blocks_m = (seq_len + BLOCK_M - 1) // BLOCK_M
    num_blocks_n = (seq_len + BLOCK_N - 1) // BLOCK_N
    
    block_mask = torch.ones(1, 1, num_blocks_m, num_blocks_n, dtype=torch.int32, device=device)
    block_mask[0,0,:5]=8
    block_mask[0,0,5:10]=2
    
    print(f"  Config: B={batch_size}, H={num_heads}, N={seq_len}, D={head_dim}")
    print(f"  Block size M: {BLOCK_M}, Block size N: {BLOCK_N}, Num blocks m: {num_blocks_m}, Num blocks n: {num_blocks_n}")

    try:
        # Torch SDPA baseline
        with torch.no_grad():
            out_baseline = block_sparse_triton_fn(q, k, v, block_mask)
        
        # Our sparse implementation (should be identical when mask=1 everywhere)
        with torch.no_grad():
            out_sparse = block_sparse_triton_fn2(q, k, v, block_mask, None)
        
        # Calculate absolute error
        abs_error = torch.abs(out_baseline - out_sparse)
        mean_abs_error = abs_error.mean().item()
        max_abs_error = abs_error.max().item()
        
        print(f"  Baseline output mean: {out_baseline.mean().item():.6f}, std: {out_baseline.std().item():.6f}")
        print(f"  Sparse output mean: {out_sparse.mean().item():.6f}, std: {out_sparse.std().item():.6f}")
        print(f"  Mean absolute error: {mean_abs_error:.8f}")
        print(f"  Max absolute error: {max_abs_error:.8f}")
        
        # Check if error is within acceptable range
        if mean_abs_error < 1e-2:  # Adjusted for fp16
            print("  ✓ Accuracy test passed!")
            return True
        else:
            print("  ✗ Accuracy test failed - error too large!")
            return False
            
    except Exception as e:
        print(f"  ✗ Error during accuracy test: {e}")
        return False


def test_gradient_correctness():
    """Test gradient correctness by comparing with torch SDPA"""
    print("\nTesting gradient correctness...")
    
    batch_size = 1
    num_heads = 4
    seq_len = 17776
    head_dim = 64
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16  # Use float32 for better numerical precision
    
    print(f"  Config: B={batch_size}, H={num_heads}, N={seq_len}, D={head_dim}")
    
    torch.manual_seed(123)
    
    # Create input tensors
    q_baseline = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
    k_baseline = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
    v_baseline = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
    
    # Clone for sparse attention (same inputs)
    q_sparse = q_baseline.clone().detach().requires_grad_(True)
    k_sparse = k_baseline.clone().detach().requires_grad_(True) 
    v_sparse = v_baseline.clone().detach().requires_grad_(True)
    
    # Create block mask - all 1s for dense attention comparison
    # Calculate blocks for Q (rows) and K (columns) separately
    num_blocks_m = (seq_len + BLOCK_M - 1) // BLOCK_M  # Q blocks (rows)
    num_blocks_n = (seq_len + BLOCK_N - 1) // BLOCK_N  # K blocks (columns)
    # Ensure mask matches Q,K dimensions: [batch_size, num_heads, num_blocks_q, num_blocks_k]
    block_mask = torch.ones(batch_size, num_heads, num_blocks_m, num_blocks_n, dtype=torch.int32, device=device)
    # block_mask[0,0,:5]=8
    # block_mask[0,0,5:10]=2
    print(f"  Number of blocks M (Q): {num_blocks_m}, N (K): {num_blocks_n}")
    print(f"  Block mask shape: {block_mask.shape}")
    
    try:
        print("  Computing torch SDPA baseline...")
        # Baseline: torch SDPA
        out_baseline = torch.nn.functional.scaled_dot_product_attention(q_baseline, k_baseline, v_baseline)
        
        # Create random gradient for backward pass
        grad_output = torch.randn_like(out_baseline)
        
        # Backward pass for baseline
        out_baseline.backward(grad_output)
        
        print("  Computing sparse attention...")
        # Our sparse implementation
        out_sparse = block_sparse_triton_fn2(q_sparse, k_sparse, v_sparse, block_mask, None)
        # Same gradient for fair comparison
        grad_output_sparse = grad_output.clone()
        
        # # Backward pass for sparse
        out_sparse.backward(grad_output_sparse)
        
        print("  Comparing forward outputs...")
        # Compare forward outputs
        forward_abs_error = torch.abs(out_baseline - out_sparse).mean().item()
        forward_max_error = torch.abs(out_baseline - out_sparse).max().item()
        print(f"    Forward mean absolute value: {out_baseline.abs().mean().item():.8f}")
        print(f"    Forward mean absolute error: {forward_abs_error:.8f}")
        print(f"    Forward max absolute error: {forward_max_error:.8f}")
        
        print("  Comparing gradients...")
        # Compare gradients
        if q_baseline.grad is not None and q_sparse.grad is not None:
            q_grad_error = torch.abs(q_baseline.grad - q_sparse.grad).mean().item()
            q_grad_max_error = torch.abs(q_baseline.grad - q_sparse.grad).max().item()
            print(f"    Q grad mean absolute value: {q_baseline.grad.abs().mean().item():.8f}")
            print(f"    Q grad mean absolute error: {q_grad_error:.8f}")
            print(f"    Q grad max absolute error: {q_grad_max_error:.8f}")
            print(f"    Q gradients[1]:{q_baseline.grad.abs()[0][1].mean().item():.8f}, {q_sparse.grad.abs()[0][1].mean().item():.8f}")
        else:
            print("    ✗ Q gradients missing")
            
        if k_baseline.grad is not None and k_sparse.grad is not None:
            k_grad_error = torch.abs(k_baseline.grad - k_sparse.grad).mean().item()
            k_grad_max_error = torch.abs(k_baseline.grad - k_sparse.grad).max().item()
            print(f"    K grad mean absolute value: {k_baseline.grad.abs().mean().item():.8f}")
            print(f"    K grad mean absolute error: {k_grad_error:.8f}")
            print(f"    K grad max absolute error: {k_grad_max_error:.8f}")
        else:
            print("    ✗ K gradients missing")
            
        if v_baseline.grad is not None and v_sparse.grad is not None:
            v_grad_error = torch.abs(v_baseline.grad - v_sparse.grad).mean().item()
            v_grad_max_error = torch.abs(v_baseline.grad - v_sparse.grad).max().item()
            print(f"    V grad mean absolute value: {v_baseline.grad.abs().mean().item():.8f}")
            print(f"    V grad mean absolute error: {v_grad_error:.8f}")
            print(f"    V grad max absolute error: {v_grad_max_error:.8f}")
        else:
            print("    ✗ V gradients missing")
        
        # Check if errors are within acceptable range
        forward_ok = forward_abs_error < 1e-2
        grad_ok = True
        
        if q_baseline.grad is not None and q_sparse.grad is not None:
            grad_ok = grad_ok and (q_grad_error < 1e-2)
        if k_baseline.grad is not None and k_sparse.grad is not None:
            grad_ok = grad_ok and (k_grad_error < 1e-2)
        if v_baseline.grad is not None and v_sparse.grad is not None:
            grad_ok = grad_ok and (v_grad_error < 1e-2)
        
        if forward_ok and grad_ok:
            print("  ✓ Gradient correctness test passed!")
            return True
        else:
            print("  ✗ Gradient correctness test failed - errors too large!")
            return False
            
    except Exception as e:
        print(f"  ✗ Gradient test failed: {e}")
        return False


def test_sparsity_patterns():
    """Test different sparsity patterns and measure speedup"""
    print("\nTesting different sparsity patterns...")
    
    # Test parameters
    batch_size = 2
    num_heads = 8
    seq_len = 1024
    head_dim = 64
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    num_runs = 10
    
    # Create test tensors
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    
    num_blocks_m = (seq_len + BLOCK_M - 1) // BLOCK_M
    num_blocks_n = (seq_len + BLOCK_N - 1) // BLOCK_N
    print(f"  Config: B={batch_size}, H={num_heads}, N={seq_len}, D={head_dim}")
    print(f"  Block size M: {BLOCK_M}, N: {BLOCK_N}, Num blocks M: {num_blocks_m}, N: {num_blocks_n}")
    
    # Baseline: torch SDPA (dense attention)
    print("\n  Baseline: Torch SDPA (dense attention)")
    baseline_times = []
    try:
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                out_baseline = torch_sdpa_baseline(q, k, v)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.time()
            baseline_times.append(end_time - start_time)
        
        baseline_mean = np.mean(baseline_times) * 1000
        baseline_std = np.std(baseline_times) * 1000
        print(f"    Time: {baseline_mean:.2f} ± {baseline_std:.2f} ms")
        
    except Exception as e:
        print(f"    Failed: {e}")
        return
    
    # Test different sparsity patterns
    patterns = [
        ("All 1s (dense)", lambda: torch.ones(1, 1, num_blocks_m, num_blocks_n, dtype=torch.int32, device=device)),
        ("50% zeros", lambda: create_sparse_pattern(num_blocks_m, num_blocks_n, {"0": 0.5, "1": 0.5}, device)),
        ("Mixed sparse", lambda: create_sparse_pattern(num_blocks_m, num_blocks_n, {"0": 0.3, "1": 0.2, "2": 0.2, "4": 0.2, "8": 0.1}, device)),
        ("Heavy pooling", lambda: create_sparse_pattern(num_blocks_m, num_blocks_n, {"0": 0.2, "1": 0.1, "2": 0.1, "4": 0.3, "8": 0.3}, device)),
        ("Local attention", lambda: create_local_pattern(num_blocks_m, num_blocks_n, window=3, device=device)),
    ]
    
    for pattern_name, pattern_func in patterns:
        print(f"\n  Testing: {pattern_name}")
        
        # Create pattern
        block_mask = pattern_func()
        
        # Count pattern statistics
        mask_flat = block_mask.flatten()
        pattern_stats = {}
        for val in [0, 1, 2, 4, 8]:
            count = (mask_flat == val).sum().item()
            if count > 0:
                pattern_stats[val] = count / len(mask_flat) * 100
        
        stats_str = ", ".join([f"{k}:{v:.1f}%" for k, v in pattern_stats.items()])
        print(f"    Pattern: {stats_str}")
        
        # Benchmark
        sparse_times = []
        try:
            for _ in range(num_runs):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start_time = time.time()
                
                with torch.no_grad():
                    out_sparse = block_sparse_triton_fn(q, k, v, block_mask, None)
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end_time = time.time()
                sparse_times.append(end_time - start_time)
            
            sparse_mean = np.mean(sparse_times) * 1000
            sparse_std = np.std(sparse_times) * 1000
            speedup = baseline_mean / sparse_mean if sparse_mean > 0 else 0
            
            print(f"    Time: {sparse_mean:.2f} ± {sparse_std:.2f} ms")
            print(f"    Speedup: {speedup:.2f}x")
            
            # Calculate accuracy vs baseline for patterns with some 1s
            if 1 in pattern_stats:
                try:
                    abs_error = torch.abs(out_baseline - out_sparse).mean().item()
                    print(f"    Mean abs error vs baseline: {abs_error:.8f}")
                except:
                    pass
            
        except Exception as e:
            print(f"    Failed: {e}")


def create_sparse_pattern(num_blocks_m: int, num_blocks_n: int, distribution: dict, device) -> torch.Tensor:
    """Create a sparse pattern with given distribution of values"""
    mask = torch.zeros(1, 1, num_blocks_m, num_blocks_n, dtype=torch.int32, device=device)
    
    total_elements = num_blocks_m * num_blocks_n
    cumulative_prob = 0.0
    rand_vals = torch.rand(num_blocks_m, num_blocks_n, device=device)
    
    for value, prob in distribution.items():
        lower = cumulative_prob
        upper = cumulative_prob + prob
        mask_condition = (rand_vals >= lower) & (rand_vals < upper)
        mask[0, 0][mask_condition] = int(value)
        cumulative_prob = upper
    
    return mask


def create_local_pattern(num_blocks_m: int, num_blocks_n: int, window: int, device) -> torch.Tensor:
    """Create a local attention pattern"""
    mask = torch.zeros(1, 1, num_blocks_m, num_blocks_n, dtype=torch.int32, device=device)
    
    for i in range(num_blocks_m):
        for j in range(num_blocks_n):
            dist = abs(i - j)
            if dist <= window:
                if dist == 0:
                    mask[0, 0, i, j] = 1  # Full attention on diagonal
                elif dist == 1:
                    mask[0, 0, i, j] = 2  # 2x pooled for adjacent
                else:
                    mask[0, 0, i, j] = 4  # 4x pooled for nearby
    
    return mask


def test_different_sequence_lengths():
    """Test with different sequence lengths to verify padding/truncation"""
    print("\nTesting different sequence lengths...")
    
    batch_size = 1
    num_heads = 4
    head_dim = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16
    
    # Test various sequence lengths that are not multiples of BLOCK_SIZE
    seq_lengths = [100, 127, 129, 200, 300, 500, 1000]
    
    for seq_len in seq_lengths:
        print(f"  Testing sequence length: {seq_len}")
        
        torch.manual_seed(42)
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
        
        # Create appropriate block mask
        num_blocks_m = (seq_len + BLOCK_M - 1) // BLOCK_M
        num_blocks_n = (seq_len + BLOCK_N - 1) // BLOCK_N
        block_mask = create_local_pattern(num_blocks_m, num_blocks_n, window=2, device=device)
        
        try:
            out = block_sparse_triton_fn(q, k, v, block_mask, None)
            
            # Test backward pass
            grad_output = torch.randn_like(out)
            out.backward(grad_output)
            
            assert out.shape == q.shape, f"Output shape mismatch: {out.shape} vs {q.shape}"
            assert q.grad is not None and q.grad.shape == q.shape
            assert k.grad is not None and k.grad.shape == k.shape  
            assert v.grad is not None and v.grad.shape == v.shape
            
            print(f"    ✓ Success - Output shape: {out.shape}, Blocks M: {num_blocks_m}, N: {num_blocks_n}")
            
        except Exception as e:
            print(f"    ✗ Failed: {e}")


def main():
    """Run all tests"""
    print("Block Sparse Attention Test Suite")
    print("=" * 50)
    print(f"Block size M: {BLOCK_M}, Block size N: {BLOCK_N}")
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, tests will run on CPU (may be slow)")
    
    # Test baseline accuracy with mask=1 vs torch SDPA
    print("\n" + "=" * 50)
    success = test_baseline_accuracy()

    if not success:
        print("Baseline accuracy test failed, but continuing with other tests...")
    
    # Test gradient correctness
    print("\n" + "=" * 50)
    test_gradient_correctness()
    exit(-1)
    # Test different sequence lengths  
    print("\n" + "=" * 50)
    test_different_sequence_lengths()
    
    # Test sparsity patterns and performance
    print("\n" + "=" * 50)
    test_sparsity_patterns()
    
    print("\n" + "=" * 50)
    print("Test suite completed!")


if __name__ == "__main__":
    main()