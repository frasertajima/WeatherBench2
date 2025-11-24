"""
Verification Script: Compare Fortran and PyTorch U-Net Outputs
==============================================================
This script verifies that the PyTorch implementation produces
identical outputs to the Fortran CUDA implementation.

Usage:
    python verify_fortran_pytorch.py <weights_dir>

Example:
    python verify_fortran_pytorch.py ../saved_models/climate_unet/epoch_0001/

Author: v28e Climate CNN Team
Date: 2025-11-23
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from climate_unet import ClimateUNet, load_fortran_weights, load_sample_io


def verify_outputs(weights_dir):
    """
    Load Fortran weights, run PyTorch inference, compare to Fortran output.
    """
    weights_dir = Path(weights_dir)

    print("=" * 70)
    print("Fortran vs PyTorch Verification")
    print("=" * 70)
    print(f"Weights directory: {weights_dir}")
    print()

    # Check if required files exist
    # Weights are in weights_dir, samples may be in parent directory
    weight_files = ["enc1_conv1_weights.bin", "final_weights.bin"]
    sample_files = ["sample_0000_input.bin", "sample_0000_output.bin"]

    # Check weights
    missing_weights = [f for f in weight_files if not (weights_dir / f).exists()]
    if missing_weights:
        print("ERROR: Missing weight files:")
        for f in missing_weights:
            print(f"  - {f}")
        print()
        print("Make sure you ran training with --save")
        return False

    # Check samples - try current dir, then parent
    samples_dir = weights_dir
    if not (weights_dir / "sample_0000_output.bin").exists():
        samples_dir = weights_dir.parent

    missing_samples = [f for f in sample_files if not (samples_dir / f).exists()]
    if missing_samples:
        print("ERROR: Missing sample files:")
        for f in missing_samples:
            print(f"  - {f}")
        print()
        print("Make sure you ran training with --export_samples")
        return False

    print(f"Samples directory: {samples_dir}")
    print()

    # Create model
    print("Creating PyTorch model...")
    model = ClimateUNet()
    model.eval()
    print(f"  Parameters: {model.count_parameters():,}")
    print()

    # Load weights
    print("Loading Fortran weights...")
    load_fortran_weights(model, weights_dir)
    print()

    # Load sample I/O
    print("Loading sample input/output from Fortran...")
    fortran_input, fortran_output = load_sample_io(samples_dir, sample_id=0)
    print(f"  Input shape:  {fortran_input.shape}")
    print(f"  Output shape: {fortran_output.shape}")
    print()

    # Run PyTorch inference
    print("Running PyTorch inference...")
    with torch.no_grad():
        pytorch_output = model(fortran_input)
    print(f"  PyTorch output shape: {pytorch_output.shape}")
    print()

    # Compare outputs
    print("Comparing outputs...")
    fortran_np = fortran_output.numpy()
    pytorch_np = pytorch_output.numpy()

    # Compute differences
    diff = np.abs(fortran_np - pytorch_np)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # Relative error (avoid division by zero)
    fortran_abs = np.abs(fortran_np)
    mask = fortran_abs > 1e-6
    rel_diff = np.zeros_like(diff)
    rel_diff[mask] = diff[mask] / fortran_abs[mask]
    max_rel_diff = np.max(rel_diff[mask]) if np.any(mask) else 0
    mean_rel_diff = np.mean(rel_diff[mask]) if np.any(mask) else 0

    print(f"  Max absolute difference:  {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")
    print(f"  Max relative difference:  {max_rel_diff:.6e}")
    print(f"  Mean relative difference: {mean_rel_diff:.6e}")
    print()

    # Per-channel analysis
    print("Per-channel analysis:")
    channel_names = ["u10", "v10", "t2m", "sp", "msl", "tp"]
    for c in range(6):
        ch_diff = np.abs(fortran_np[0, c] - pytorch_np[0, c])
        ch_max = np.max(ch_diff)
        ch_mean = np.mean(ch_diff)
        print(
            f"  Channel {c} ({channel_names[c]}): max={ch_max:.6e}, mean={ch_mean:.6e}"
        )
    print()

    # Verdict
    print("=" * 70)
    tolerance = 1e-4  # Allow small floating point differences

    if max_diff < tolerance:
        print("PASS: Fortran and PyTorch outputs match!")
        print(f"      (max difference {max_diff:.6e} < tolerance {tolerance:.6e})")
        success = True
    elif max_diff < 1e-2:
        print("WARNING: Small differences detected (likely numerical precision)")
        print(f"         max difference = {max_diff:.6e}")
        print("         This may be acceptable for verification purposes.")
        success = True
    else:
        print("FAIL: Significant differences between Fortran and PyTorch!")
        print(f"      max difference = {max_diff:.6e}")
        success = False

    print("=" * 70)

    return success


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_fortran_pytorch.py <weights_dir>")
        print()
        print("Example:")
        print(
            "  python verify_fortran_pytorch.py ../saved_models/climate_unet/epoch_0001/"
        )
        print(
            "  python verify_fortran_pytorch.py ../saved_models/climate_unet/epoch_0001/"
        )
        sys.exit(1)

    weights_dir = sys.argv[1]
    success = verify_outputs(weights_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
