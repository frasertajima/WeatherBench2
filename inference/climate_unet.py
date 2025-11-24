"""
Climate U-Net Model - PyTorch Implementation
=============================================
Matches the Fortran CUDA implementation for verification.

Architecture:
    Encoder: 6->32->64->128 channels with 2x downsampling
    Bottleneck: 128->256->256 at lowest resolution
    Decoder: 256->128->64->32 with skip connections and 2x upsampling
    Output: 32->6 with 1x1 conv

Input:  (batch, 6, 240, 121) - padded internally to (batch, 6, 256, 128)
Output: (batch, 6, 240, 121)

Author: v28e Climate CNN Team
Date: 2025-11-23
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    """Encoder block: 2 convolutions + max pooling"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Conv1 + ReLU
        skip = F.relu(self.conv1(x))
        # Conv2 + ReLU
        skip = F.relu(self.conv2(skip))
        # Pool
        out = self.pool(skip)
        return out, skip


class DecoderBlock(nn.Module):
    """Decoder block: upsample + concat skip + 2 convolutions"""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels + skip_channels, out_channels, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, skip):
        # Upsample 2x (nearest neighbor)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        # Conv1 + ReLU
        x = F.relu(self.conv1(x))
        # Conv2 + ReLU
        x = F.relu(self.conv2(x))
        return x


class ClimateUNet(nn.Module):
    """
    U-Net for climate prediction.

    Matches Fortran implementation exactly for verification.
    """

    # Climate data dimensions
    ORIGINAL_HEIGHT = 240
    ORIGINAL_WIDTH = 121
    PADDED_HEIGHT = 256
    PADDED_WIDTH = 128
    INPUT_CHANNELS = 6

    def __init__(self):
        super().__init__()

        # Encoder blocks
        self.enc1 = EncoderBlock(6, 32)  # 256x128 -> 128x64
        self.enc2 = EncoderBlock(32, 64)  # 128x64 -> 64x32
        self.enc3 = EncoderBlock(64, 128)  # 64x32 -> 32x16

        # Bottleneck
        self.bottleneck1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bottleneck2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Decoder blocks
        self.dec3 = DecoderBlock(256, 128, 128)  # 32x16 -> 64x32
        self.dec2 = DecoderBlock(128, 64, 64)  # 64x32 -> 128x64
        self.dec1 = DecoderBlock(64, 32, 32)  # 128x64 -> 256x128

        # Final 1x1 conv (no ReLU)
        self.final_conv = nn.Conv2d(32, 6, kernel_size=1)

    def pad_input(self, x):
        """Pad from 240x121 to 256x128"""
        # x: (batch, 6, 240, 121)
        # Pad right and bottom with zeros
        pad_h = self.PADDED_HEIGHT - self.ORIGINAL_HEIGHT  # 16
        pad_w = self.PADDED_WIDTH - self.ORIGINAL_WIDTH  # 7
        # F.pad format: (left, right, top, bottom)
        return F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)

    def crop_output(self, x):
        """Crop from 256x128 to 240x121"""
        return x[:, :, : self.ORIGINAL_HEIGHT, : self.ORIGINAL_WIDTH]

    def forward(self, x):
        # Pad input
        x = self.pad_input(x)

        # Encoder path
        x, skip1 = self.enc1(x)  # skip1: 256x128, x: 128x64
        x, skip2 = self.enc2(x)  # skip2: 128x64, x: 64x32
        x, skip3 = self.enc3(x)  # skip3: 64x32, x: 32x16

        # Bottleneck
        x = F.relu(self.bottleneck1(x))
        x = F.relu(self.bottleneck2(x))

        # Decoder path
        x = self.dec3(x, skip3)  # 32x16 -> 64x32
        x = self.dec2(x, skip2)  # 64x32 -> 128x64
        x = self.dec1(x, skip1)  # 128x64 -> 256x128

        # Final conv (no activation for regression)
        x = self.final_conv(x)

        # Crop output
        x = self.crop_output(x)

        return x

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_fortran_weights(model, weights_dir):
    """
    Load weights exported from Fortran into PyTorch model.

    Fortran exports weights in standard F-order (column-major).
    Conv weights allocated in Fortran as: weights(out_ch, in_ch, kH, kW)

    Standard F-order loading:
    - np.fromfile().reshape((out_ch, in_ch, kH, kW), order='F')
    - This correctly interprets the column-major memory layout

    Args:
        model: ClimateUNet instance
        weights_dir: Path to directory with .bin files
    """
    weights_dir = Path(weights_dir)

    def load_conv_weights(layer, prefix):
        """Load weights and bias for a conv layer"""
        # Load weights
        w_file = weights_dir / f"{prefix}_weights.bin"
        w = np.fromfile(w_file, dtype=np.float32)

        # Get expected shape from layer
        expected_shape = layer.weight.shape  # (out_ch, in_ch, kH, kW)
        out_ch, in_ch, kH, kW = expected_shape

        # Fortran allocates weights(out_ch, in_ch, kH, kW) and writes in F-order
        # CIFAR-10 proven method:
        # Step 1: Reshape with REVERSED dims using F-order
        w = w.reshape((kH, kW, in_ch, out_ch), order="F")
        # Step 2: Transpose (3,2,1,0) to get (out_ch, in_ch, kW, kH)
        w = w.transpose(3, 2, 1, 0)
        # Step 3: Flip spatial dims to fix rotation
        w = np.flip(w, axis=(2, 3)).copy()

        layer.weight.data = torch.from_numpy(w)

        # Load bias
        b_file = weights_dir / f"{prefix}_bias.bin"
        b = np.fromfile(b_file, dtype=np.float32)
        layer.bias.data = torch.from_numpy(b.copy())

        print(f"  Loaded {prefix}: weight {expected_shape}, bias {b.shape}")

    print(f"Loading weights from {weights_dir}")

    # Encoder blocks
    load_conv_weights(model.enc1.conv1, "enc1_conv1")
    load_conv_weights(model.enc1.conv2, "enc1_conv2")
    load_conv_weights(model.enc2.conv1, "enc2_conv1")
    load_conv_weights(model.enc2.conv2, "enc2_conv2")
    load_conv_weights(model.enc3.conv1, "enc3_conv1")
    load_conv_weights(model.enc3.conv2, "enc3_conv2")

    # Bottleneck
    load_conv_weights(model.bottleneck1, "bottleneck1")
    load_conv_weights(model.bottleneck2, "bottleneck2")

    # Decoder blocks
    load_conv_weights(model.dec3.conv1, "dec3_conv1")
    load_conv_weights(model.dec3.conv2, "dec3_conv2")
    load_conv_weights(model.dec2.conv1, "dec2_conv1")
    load_conv_weights(model.dec2.conv2, "dec2_conv2")
    load_conv_weights(model.dec1.conv1, "dec1_conv1")
    load_conv_weights(model.dec1.conv2, "dec1_conv2")

    # Final conv
    load_conv_weights(model.final_conv, "final")

    print("Weights loaded successfully!")
    return model


def load_sample_io(weights_dir, sample_id=0):
    """
    Load sample input/output exported from Fortran for verification.

    Fortran tensor layout: (W, H, C, N) = (121, 240, 6, 1)
    This (W,H,C,N) layout ensures F-order storage matches cuDNN's C-order expectation.
    Stored in column-major (F-order): W varies fastest, then H, then C, then N

    To load in Python:
    1. Reshape with F-order using Fortran's dimensions (W, H, C, N)
    2. Transpose to PyTorch's (N, C, H, W) format

    Returns:
        input_tensor: (1, 6, 240, 121) in PyTorch NCHW format
        output_tensor: (1, 6, 240, 121) in PyTorch NCHW format
    """
    weights_dir = Path(weights_dir)

    # Load input
    # Fortran array is (W=121, H=240, C=6, N=1) stored in column-major order
    input_file = weights_dir / f"sample_{sample_id:04d}_input.bin"
    inp = np.fromfile(input_file, dtype=np.float32)
    inp = inp.reshape((121, 240, 6, 1), order="F")  # (W, H, C, N)
    inp = inp.transpose(3, 2, 1, 0)  # -> (N, C, H, W) = (1, 6, 240, 121)

    # Load output (same layout)
    output_file = weights_dir / f"sample_{sample_id:04d}_output.bin"
    out = np.fromfile(output_file, dtype=np.float32)
    out = out.reshape((121, 240, 6, 1), order="F")  # (W, H, C, N)
    out = out.transpose(3, 2, 1, 0)  # -> (N, C, H, W) = (1, 6, 240, 121)

    return torch.from_numpy(inp.copy()), torch.from_numpy(out.copy())


if __name__ == "__main__":
    # Quick test
    print("=" * 60)
    print("Climate U-Net - PyTorch Implementation")
    print("=" * 60)

    model = ClimateUNet()
    print(f"Total parameters: {model.count_parameters():,}")

    # Test forward pass
    x = torch.randn(1, 6, 240, 121)
    y = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")

    # Verify shapes match
    assert y.shape == x.shape, "Output shape mismatch!"
    print("Shape verification passed!")
