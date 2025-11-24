# v28e Climate CNN/U-Net - Design Document

**Goal**: Build a proper CNN/U-Net architecture for weather prediction that exploits the spatial structure of climate data, running entirely on consumer hardware.

## Project Vision

Demonstrate that state-of-the-art weather prediction architectures (inspired by Pangu-Weather, GraphCast) can run on consumer hardware using our streaming infrastructure. This is a "no-compromise" approach - handle 81GB datasets as easily as toy datasets.

## Why CNN/U-Net for Weather Data?

### The Problem with FC Networks

The v28d baseline uses fully-connected layers that:
- Treat each grid cell independently (no spatial awareness)
- Cannot learn local patterns (fronts, pressure gradients)
- Require massive parameters (174,240 x 256 = 44M params just for one layer)
- Don't generalize well to unseen patterns

### What Weather Data Looks Like

```
Input shape: (6, 240, 121) = 6 channels x 240 lat x 121 lon

Channel 0 (z500):     Channel 1 (t850):     ...
+---------------+     +---------------+
|  Pressure     |     |  Temperature  |
|  patterns     |     |  gradients    |
|  ~150km grid  |     |               |
+---------------+     +---------------+
```

Weather phenomena are **spatial**:
- A cold front spans multiple grid cells
- Pressure systems have smooth gradients
- Jet streams are elongated features

### How CNNs Help

Convolutional layers:
- Share weights across spatial locations (efficient)
- Learn local patterns (3x3 or 5x5 neighborhoods)
- Build hierarchical features (edges -> textures -> objects)
- Much fewer parameters than FC for same expressivity

## Architecture Options

### Option 1: Simple CNN (Starting Point)

```
Input (6, 240, 121)
    |
Conv2D(6 -> 32, 3x3, pad=1) + ReLU
Conv2D(32 -> 64, 3x3, pad=1) + ReLU
MaxPool2D(2x2) -> (64, 120, 60)
    |
Conv2D(64 -> 128, 3x3, pad=1) + ReLU
MaxPool2D(2x2) -> (128, 60, 30)
    |
Conv2D(128 -> 64, 3x3, pad=1) + ReLU
Upsample(2x2) -> (64, 120, 60)
    |
Conv2D(64 -> 32, 3x3, pad=1) + ReLU
Upsample(2x2) -> (32, 240, 120)
    |
Conv2D(32 -> 6, 3x3, pad=1)
    |
Output (6, 240, 121)  # Pad last dim
```

**Parameters**: ~200K (vs 89M for 2-layer FC)
**Pros**: Simple, fast, tests infrastructure
**Cons**: Information loss through pooling, no skip connections

### Option 2: U-Net (Recommended)

```
Input (6, 240, 121)
    |
[Encoder Block 1] Conv(6->32) + Conv(32->32) ----+
    | MaxPool                                     |
[Encoder Block 2] Conv(32->64) + Conv(64->64) ---+|
    | MaxPool                                    ||
[Encoder Block 3] Conv(64->128) + Conv(128->128)+||
    | MaxPool                                   |||
                                                |||
[Bottleneck] Conv(128->256) + Conv(256->128)   |||
                                                |||
    | Upsample                                  |||
[Decoder Block 3] Concat + Conv(256->64) -------+||
    | Upsample                                   ||
[Decoder Block 2] Concat + Conv(128->32) --------+|
    | Upsample                                    |
[Decoder Block 1] Concat + Conv(64->6) -----------+
    |
Output (6, 240, 121)
```

**Parameters**: ~2M
**Pros**: Skip connections preserve detail, proven architecture for dense prediction
**Cons**: More complex implementation

### Option 3: Vision Transformer (Future)

Self-attention over spatial patches. Would require significant new infrastructure.

## Implementation Plan

### Phase 1: Conv2D Module (Foundation)

Build a reusable Conv2D module using cuDNN:

```fortran
module conv2d_cudnn
    use cudafor
    use cudnn
    
    type :: conv2d_layer
        type(cudnnTensorDescriptor) :: input_desc, output_desc
        type(cudnnFilterDescriptor) :: filter_desc
        type(cudnnConvolutionDescriptor) :: conv_desc
        real(4), device, allocatable :: weights(:,:,:,:)
        real(4), device, allocatable :: bias(:)
        real(4), device, allocatable :: grad_weights(:,:,:,:)
        real(4), device, allocatable :: grad_bias(:)
        ! Adam state...
    end type
    
contains
    subroutine conv2d_init(layer, in_channels, out_channels, kernel_size, padding)
    subroutine conv2d_forward(layer, input, output)
    subroutine conv2d_backward(layer, grad_output, grad_input)
    subroutine conv2d_update(layer, lr, timestep)
end module
```

**Test**: Verify output shapes, gradient flow

### Phase 2: Pooling and Upsampling Modules

```fortran
module pooling_cudnn
    subroutine maxpool2d_forward(input, output, indices)  ! Save indices for backward
    subroutine maxpool2d_backward(grad_output, indices, grad_input)
    
    subroutine upsample2d_forward(input, output, scale)  ! Bilinear or nearest
    subroutine upsample2d_backward(grad_output, grad_input, scale)
end module
```

**Test**: Verify dimensions, gradient correctness

### Phase 3: Simple CNN Integration

Combine modules into working CNN:
1. Forward pass through all layers
2. MSE loss (reuse from v28d)
3. Backward pass through all layers
4. Adam update (reuse from v28d)

**Test**: Train on small subset, verify loss decreases

### Phase 4: U-Net Architecture

Add skip connections:
1. Store encoder outputs during forward
2. Concatenate with decoder inputs
3. Handle dimension matching

**Test**: Full training run, compare with FC baseline

### Phase 5: Optimization

1. **cuDNN workspace tuning** - find optimal algorithms
2. **Mixed precision** - FP16 convolutions with FP32 accumulation
3. **Tensor cores** - if architecture supports NHWC layout
4. **Memory optimization** - gradient checkpointing if needed

## Technical Challenges

### 1. Dimension Handling

The grid is 240x121 (not powers of 2). Options:
- Pad to 256x128 for cleaner pooling
- Use stride-1 convolutions at boundaries
- Asymmetric pooling

**Recommendation**: Pad to 256x128 at input, crop at output

### 2. cuDNN Descriptor Management

Each layer needs:
- Input tensor descriptor
- Output tensor descriptor
- Filter descriptor
- Convolution descriptor
- Algorithm selection

**Recommendation**: Create a `layer_context` type that bundles all descriptors

### 3. Memory for Skip Connections

U-Net stores encoder outputs for skip connections. With batch=32:
- Encoder 1 output: 32 x 32 x 256 x 128 = 32MB
- Encoder 2 output: 32 x 64 x 128 x 64 = 16MB
- Encoder 3 output: 32 x 128 x 64 x 32 = 8MB

Total ~60MB per batch - manageable.

### 4. Gradient Checkpointing (if needed)

If memory is tight, recompute forward activations during backward pass instead of storing them. Trade compute for memory.

## File Structure

```
v28e_climate_cnn/
+-- common/
|   +-- conv2d_cudnn.cuf       # Conv2D layer with cuDNN
|   +-- pooling_cudnn.cuf      # MaxPool, Upsample
|   +-- batch_norm.cuf         # Batch normalization (optional)
|   +-- unet_blocks.cuf        # Encoder/decoder blocks
+-- tests/
|   +-- test_conv2d.cuf        # Test convolution layer
|   +-- test_pooling.cuf       # Test pooling layers
|   +-- test_unet_block.cuf    # Test U-Net blocks
|   +-- test_simple_cnn.cuf    # Test simple CNN end-to-end
+-- climate_unet.cuf           # Main U-Net model
+-- climate_train_cnn.cuf      # Training program
+-- compile.sh                 # Compilation script
+-- DESIGN_DOCUMENT.md         # This file
+-- README.md                  # Usage documentation
```

## Dependencies

Reuse from v28d:
- `streaming_regression_loader.cuf` - Data loading
- `mse_loss.cuf` - Loss computation
- `adam_optimizer.cuf` - Weight updates
- `warp_shuffle.cuf` - Reduction primitives

New:
- cuDNN convolution APIs
- Potentially cuBLAS for 1x1 convolutions

## Success Metrics

| Metric | FC Baseline | CNN Target | U-Net Target |
|--------|-------------|------------|--------------|
| Parameters | 89M | ~200K | ~2M |
| Train Loss (epoch 3) | 0.66 | <0.5 | <0.3 |
| Time per epoch | ~15 min* | <5 min | <10 min |
| Memory | 1.7 GB | <2 GB | <3 GB |

*Extrapolated from batch timing

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| cuDNN API complexity | High | Medium | Start with simple conv, build up |
| Dimension mismatches | Medium | High | Comprehensive unit tests |
| Memory overflow | Low | High | Monitor usage, implement checkpointing |
| Slow convergence | Medium | Medium | Tune learning rate, add batch norm |

## Timeline (Effort-based, not calendar)

| Phase | Effort | Deliverable |
|-------|--------|-------------|
| 1. Conv2D module | Medium | Working cuDNN convolution |
| 2. Pooling modules | Low | MaxPool, Upsample |
| 3. Simple CNN | Medium | End-to-end training |
| 4. U-Net | High | Full architecture |
| 5. Optimization | Medium | Production performance |

## References

### Weather Prediction
- Pangu-Weather: https://arxiv.org/abs/2211.02556
- GraphCast: https://arxiv.org/abs/2212.12794
- FourCastNet: https://arxiv.org/abs/2202.11214

### U-Net Architecture
- Original U-Net: https://arxiv.org/abs/1505.04597
- U-Net for weather: https://arxiv.org/abs/2103.09564

### cuDNN Documentation
- cuDNN Developer Guide: https://docs.nvidia.com/deeplearning/cudnn/developer-guide/
- cuDNN API Reference: https://docs.nvidia.com/deeplearning/cudnn/api/

---

**Status**: Planning Complete
**Next Step**: Implement Phase 1 (Conv2D module)
**Created**: 2025-11-22
