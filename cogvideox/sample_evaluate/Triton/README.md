# Triton Block Sparse Attention Implementation

This directory contains the implementation of block sparse attention mechanisms optimized for CogVideo, including Gilbert curve-based 3D spatial rearrangement for improved memory locality.

## Directory Structure

```
Triton/
├── README.md                           # This file
├── cogvideo_newattn.py                # Main CogVideo attention implementation with Gilbert rearrangement
├── test_block_sparse_attention.py     # Comprehensive test suite for block sparse attention
├── kernels/                           # Triton kernel implementations
│   ├── attn_pooling_kernel.py         # Attention pooling kernel
│   ├── block_sparse_attn_kernel.py    # Basic block sparse attention kernel
│   └── block_sparse_attn_kernel_with_backward.py  # Block sparse attention with backward pass
├── utils/                             # Utility functions and algorithms
│   ├── gilbert3d.py                   # Gilbert 3D space-filling curve implementation
│   ├── tools.py                      # General utility functions
│   ├── 2307.08691v1.pdf              # Reference paper
│   └── new_kernel.pdf                # Kernel documentation
└── tests/                            # Additional test files
    └── test_gilbert_rearranger.py    # Tests for Gilbert rearrangement functionality

```

## Key Components

### 1. Main Implementation (`cogvideo_newattn.py`)
- **GilbertRearranger**: 3D space-filling curve rearrangement for video tokens
- **AdaptiveBlockSparseAttnTrain**: Main training-ready attention module
- **Sparse attention mechanisms**: Energy-based mask generation and pooling
- **Integration functions**: For CogVideo model integration

### 2. Kernels (`kernels/`)
- Optimized Triton kernels for block sparse attention
- Support for different pooling levels (1x, 2x, 4x, 8x)
- Forward and backward pass implementations

### 3. Utilities (`utils/`)
- Gilbert 3D curve implementation for spatial locality
- Performance timing utilities
- Reference documentation

### 4. Tests (`tests/`)
- Comprehensive test suites for correctness and performance
- Edge case testing
- Gradient verification

## Key Features

### Gilbert Curve Rearrangement
- **Purpose**: Improve spatial locality for 3D video tokens
- **Implementation**: `GilbertRearranger` class in `cogvideo_newattn.py`
- **Benefits**: 1.86x better spatial locality, 100% perfect adjacency
- **Usage**: Automatically handles text + video token sequences

### Block Sparse Attention
- **Adaptive sparsity**: Energy-based mask generation
- **Multi-level pooling**: 1x, 2x, 4x, 8x pooling support  
- **Gradient support**: Full backward pass implementation
- **Performance**: Significant speedup on long sequences

### Integration with CogVideo
- **Text + Video tokens**: Seamless handling of mixed sequences
- **Memory efficient**: Reduced memory usage through sparsity
- **Training ready**: Gradient-enabled for end-to-end training

## Usage Example

```python
from cogvideo_newattn import AdaptiveBlockSparseAttnTrain

# Initialize attention module
attention = AdaptiveBlockSparseAttnTrain()

# Process attention (B, H, L, D) tensors
output = attention(q, k, v)
```

## Testing

Run the comprehensive test suite:
```bash
python test_block_sparse_attention.py
python tests/test_gilbert_rearranger.py
```

## Performance

- **Spatial locality**: 1.86x improvement with Gilbert rearrangement  
- **Memory efficiency**: Significant reduction through adaptive sparsity
- **Speed**: Up to 2-3x speedup on long sequences depending on sparsity pattern

## References

- Gilbert 3D space-filling curves for spatial locality
- Block sparse attention for efficient long sequence processing
- CogVideo integration for text-to-video generation