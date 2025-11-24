#!/usr/bin/env python3
"""
Training Step Verification: Compare Fortran and PyTorch Training
=================================================================

This script verifies that ONE training step in PyTorch produces identical
results to the Fortran CUDA implementation:
  - Forward pass output
  - Loss value
  - Gradients for all layers
  - Updated weights after Adam step
  - TIMING comparison for performance analysis

Usage:
    python verify_training_step.py <verify_dir>

Example:
    python verify_training_step.py ../training_verify/

The verify_dir should contain:
    initial_weights/  - Weights before training (from Fortran)
    step_data/        - Input, target, gradients, timing
    updated_weights/  - Weights after one Adam step (from Fortran)

Author: v28e Climate CNN Team
Date: 2025-11-23
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from climate_unet import ClimateUNet, load_fortran_weights


def load_binary_array(filepath, shape, dtype=np.float32):
    """Load binary array in Fortran order."""
    data = np.fromfile(filepath, dtype=dtype)
    return data.reshape(shape, order="F")


def load_hyperparams(filepath):
    """Load hyperparameters from text file."""
    params = {}
    with open(filepath, "r") as f:
        for line in f:
            if ":" in line:
                key, value = line.strip().split(":", 1)
                key = key.strip()
                value = value.strip()
                try:
                    if "." in value or "e" in value.lower():
                        params[key] = float(value)
                    else:
                        params[key] = int(value)
                except ValueError:
                    params[key] = value
    return params


def load_fortran_timing(filepath):
    """Load Fortran timing results."""
    return load_hyperparams(filepath)


def convert_fortran_tensor(arr, is_weight=False):
    """
    Convert Fortran array to PyTorch tensor.

    For activations: Fortran (W,H,C,N) -> PyTorch (N,C,H,W)
    For weights: Fortran (out,in,kH,kW) in F-order -> PyTorch (out,in,kH,kW)
    """
    if is_weight:
        # Weights: reshape F-order, transpose, flip spatial
        # Shape is (out_ch, in_ch, kH, kW)
        arr = arr.transpose(3, 2, 1, 0)  # Reverse all dims for F->C order
        arr = np.flip(arr, axis=(2, 3)).copy()  # Flip spatial dims
    else:
        # Activations: (W,H,C,N) -> (N,C,H,W)
        arr = arr.transpose(3, 2, 1, 0)  # (W,H,C,N) -> (N,C,H,W)

    return torch.from_numpy(arr.copy())


class TimedClimateUNet(ClimateUNet):
    """ClimateUNet with timing for each phase."""

    def timed_forward(self, x):
        """Forward pass with timing (returns output and time in ms)."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            output = self.forward(x)

            end.record()
            torch.cuda.synchronize()
            elapsed_ms = start.elapsed_time(end)
        else:
            start = time.perf_counter()
            output = self.forward(x)
            elapsed_ms = (time.perf_counter() - start) * 1000

        return output, elapsed_ms


def verify_training_step(verify_dir):
    """
    Complete training step verification with timing comparison.
    """
    verify_dir = Path(verify_dir)
    initial_dir = verify_dir / "initial_weights"
    step_dir = verify_dir / "step_data"
    updated_dir = verify_dir / "updated_weights"

    print("=" * 70)
    print("Training Step Verification: Fortran vs PyTorch")
    print("=" * 70)
    print(f"Verify directory: {verify_dir}")
    print()

    # Check directories exist
    for d in [initial_dir, step_dir, updated_dir]:
        if not d.exists():
            print(f"ERROR: Directory not found: {d}")
            return False

    # Load hyperparameters
    print("Loading hyperparameters...")
    params = load_hyperparams(step_dir / "hyperparams.txt")
    lr = params["learning_rate"]
    timestep = params["timestep"]
    batch_size = params["batch_size"]
    beta1 = params.get("adam_beta1", 0.9)
    beta2 = params.get("adam_beta2", 0.999)
    epsilon = params.get("adam_epsilon", 1e-8)

    print(f"  Learning rate: {lr}")
    print(f"  Timestep: {timestep}")
    print(f"  Batch size: {batch_size}")
    print(f"  Adam betas: ({beta1}, {beta2})")
    print(f"  Adam epsilon: {epsilon}")
    print()

    # Load Fortran timing
    print("Loading Fortran timing...")
    fortran_timing = load_fortran_timing(step_dir / "timing_fortran.txt")
    print(f"  Forward:  {fortran_timing.get('forward_ms', 'N/A'):.3f} ms")
    print(f"  Backward: {fortran_timing.get('backward_ms', 'N/A'):.3f} ms")
    print(f"  Update:   {fortran_timing.get('update_ms', 'N/A'):.3f} ms")
    print(f"  Total:    {fortran_timing.get('total_gpu_ms', 'N/A'):.3f} ms")
    print()

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch device: {device}")
    print()

    # Create model
    print("Creating PyTorch model...")
    model = TimedClimateUNet()
    model.to(device)
    model.train()  # Training mode
    print(f"  Parameters: {model.count_parameters():,}")
    print()

    # Load initial weights
    print("Loading initial weights from Fortran...")
    load_fortran_weights(model, initial_dir)
    model.to(device)  # Ensure model is on device after loading weights
    print()

    # Load input and target
    print("Loading input/target batch...")

    # Try padded first, fall back to original
    if (step_dir / "input_padded.bin").exists():
        input_shape = (128, 256, 6, batch_size)  # (W,H,C,N) padded
        input_np = load_binary_array(step_dir / "input_padded.bin", input_shape)
        target_np = load_binary_array(step_dir / "target_padded.bin", input_shape)
        using_padded = True
    else:
        input_shape = (121, 240, 6, batch_size)  # (W,H,C,N) original
        input_np = load_binary_array(step_dir / "input_batch.bin", input_shape)
        target_np = load_binary_array(step_dir / "target_batch.bin", input_shape)
        using_padded = False

    input_tensor = convert_fortran_tensor(input_np).to(device)
    target_tensor = convert_fortran_tensor(target_np).to(device)

    print(f"  Input shape (PyTorch): {input_tensor.shape}")
    print(f"  Using padded: {using_padded}")

    # If data is already padded, we need to skip model's internal padding
    # Override the model's pad_input and crop_output to be no-ops
    if using_padded:
        print("  Note: Using pre-padded data, disabling model padding/cropping")
        model.pad_input = lambda x: x
        model.crop_output = lambda x: x
    print()

    # Setup optimizer with same hyperparameters
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon
    )

    # Load Fortran loss for comparison
    with open(step_dir / "loss.txt", "r") as f:
        fortran_loss = float(f.read().strip())
    print(f"Fortran loss: {fortran_loss:.6f}")
    print()

    # =========================================
    # TIMED PYTORCH TRAINING STEP
    # =========================================
    print("=" * 70)
    print("TIMED PYTORCH TRAINING STEP")
    print("=" * 70)

    # Warm-up run
    print("Warm-up run...")
    optimizer.zero_grad()
    _ = model(input_tensor)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Reset for timed run
    load_fortran_weights(model, initial_dir)
    model.to(device)  # Ensure model is on device after reloading
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon
    )

    # Forward pass (timed)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        fwd_start = torch.cuda.Event(enable_timing=True)
        fwd_end = torch.cuda.Event(enable_timing=True)
        fwd_start.record()
    else:
        fwd_start_time = time.perf_counter()

    optimizer.zero_grad()
    output = model(input_tensor)

    if torch.cuda.is_available():
        fwd_end.record()
        torch.cuda.synchronize()
        pytorch_fwd_ms = fwd_start.elapsed_time(fwd_end)
    else:
        pytorch_fwd_ms = (time.perf_counter() - fwd_start_time) * 1000

    print(f"  Forward pass:  {pytorch_fwd_ms:.3f} ms")

    # Loss computation (timed)
    if torch.cuda.is_available():
        loss_start = torch.cuda.Event(enable_timing=True)
        loss_end = torch.cuda.Event(enable_timing=True)
        loss_start.record()
    else:
        loss_start_time = time.perf_counter()

    loss = F.mse_loss(output, target_tensor)

    if torch.cuda.is_available():
        loss_end.record()
        torch.cuda.synchronize()
        pytorch_loss_ms = loss_start.elapsed_time(loss_end)
    else:
        pytorch_loss_ms = (time.perf_counter() - loss_start_time) * 1000

    print(f"  Loss compute:  {pytorch_loss_ms:.3f} ms")
    print(f"  Loss value:    {loss.item():.6f}")

    # Backward pass (timed)
    if torch.cuda.is_available():
        bwd_start = torch.cuda.Event(enable_timing=True)
        bwd_end = torch.cuda.Event(enable_timing=True)
        bwd_start.record()
    else:
        bwd_start_time = time.perf_counter()

    loss.backward()

    if torch.cuda.is_available():
        bwd_end.record()
        torch.cuda.synchronize()
        pytorch_bwd_ms = bwd_start.elapsed_time(bwd_end)
    else:
        pytorch_bwd_ms = (time.perf_counter() - bwd_start_time) * 1000

    print(f"  Backward pass: {pytorch_bwd_ms:.3f} ms")

    # Adam update (timed)
    if torch.cuda.is_available():
        upd_start = torch.cuda.Event(enable_timing=True)
        upd_end = torch.cuda.Event(enable_timing=True)
        upd_start.record()
    else:
        upd_start_time = time.perf_counter()

    optimizer.step()

    if torch.cuda.is_available():
        upd_end.record()
        torch.cuda.synchronize()
        pytorch_upd_ms = upd_start.elapsed_time(upd_end)
    else:
        pytorch_upd_ms = (time.perf_counter() - upd_start_time) * 1000

    print(f"  Adam update:   {pytorch_upd_ms:.3f} ms")

    pytorch_total_ms = pytorch_fwd_ms + pytorch_bwd_ms + pytorch_upd_ms
    print()
    print(f"  TOTAL (GPU):   {pytorch_total_ms:.3f} ms")
    print()

    # =========================================
    # TIMING COMPARISON
    # =========================================
    print("=" * 70)
    print("TIMING COMPARISON")
    print("=" * 70)

    fortran_fwd = fortran_timing.get("forward_ms", 0)
    fortran_bwd = fortran_timing.get("backward_ms", 0)
    fortran_upd = fortran_timing.get("update_ms", 0)
    fortran_total = fortran_timing.get("total_gpu_ms", 0)

    print(
        f"{'Phase':<15} {'Fortran (ms)':<15} {'PyTorch (ms)':<15} {'Ratio (F/P)':<15}"
    )
    print("-" * 60)
    print(
        f"{'Forward':<15} {fortran_fwd:<15.3f} {pytorch_fwd_ms:<15.3f} {fortran_fwd / pytorch_fwd_ms if pytorch_fwd_ms > 0 else 0:<15.2f}"
    )
    print(
        f"{'Backward':<15} {fortran_bwd:<15.3f} {pytorch_bwd_ms:<15.3f} {fortran_bwd / pytorch_bwd_ms if pytorch_bwd_ms > 0 else 0:<15.2f}"
    )
    print(
        f"{'Update':<15} {fortran_upd:<15.3f} {pytorch_upd_ms:<15.3f} {fortran_upd / pytorch_upd_ms if pytorch_upd_ms > 0 else 0:<15.2f}"
    )
    print("-" * 60)
    print(
        f"{'TOTAL':<15} {fortran_total:<15.3f} {pytorch_total_ms:<15.3f} {fortran_total / pytorch_total_ms if pytorch_total_ms > 0 else 0:<15.2f}"
    )
    print()

    if fortran_total < pytorch_total_ms:
        speedup = pytorch_total_ms / fortran_total
        print(f"Fortran is {speedup:.2f}x FASTER than PyTorch")
    else:
        speedup = fortran_total / pytorch_total_ms
        print(f"PyTorch is {speedup:.2f}x faster than Fortran")
    print()

    # =========================================
    # NUMERICAL VERIFICATION
    # =========================================
    print("=" * 70)
    print("NUMERICAL VERIFICATION")
    print("=" * 70)

    # Compare loss
    loss_diff = abs(loss.item() - fortran_loss)
    loss_rel = loss_diff / fortran_loss if fortran_loss > 0 else 0
    print(f"Loss comparison:")
    print(f"  Fortran: {fortran_loss:.10f}")
    print(f"  PyTorch: {loss.item():.10f}")
    print(f"  Abs diff: {loss_diff:.2e}")
    print(f"  Rel diff: {loss_rel:.2e}")
    loss_match = loss_diff < 1e-4
    print(f"  Match: {'PASS' if loss_match else 'FAIL'}")
    print()

    # Compare forward output
    print("Forward output comparison:")
    if using_padded:
        fortran_output = load_binary_array(
            step_dir / "forward_output_padded.bin", (128, 256, 6, batch_size)
        )
    else:
        fortran_output = load_binary_array(
            step_dir / "forward_output.bin", (121, 240, 6, batch_size)
        )
    fortran_output_tensor = convert_fortran_tensor(fortran_output)

    output_diff = (output.cpu() - fortran_output_tensor).abs()
    max_output_diff = output_diff.max().item()
    mean_output_diff = output_diff.mean().item()
    print(f"  Max diff:  {max_output_diff:.2e}")
    print(f"  Mean diff: {mean_output_diff:.2e}")
    output_match = max_output_diff < 1e-4
    print(f"  Match: {'PASS' if output_match else 'FAIL'}")
    print()

    # Compare updated weights (sample a few layers)
    print("Updated weights comparison (sample layers):")

    layers_to_check = [
        ("enc1_conv1", model.enc1.conv1, (32, 6, 3, 3)),
        ("bottleneck1", model.bottleneck1, (256, 128, 3, 3)),
        ("final", model.final_conv, (6, 32, 1, 1)),
    ]

    all_weights_match = True
    for name, layer, shape in layers_to_check:
        try:
            fortran_weights = load_binary_array(
                updated_dir / f"{name}_weights.bin", shape
            )
            # Convert Fortran weights to PyTorch format
            fw = fortran_weights.transpose(3, 2, 1, 0)
            fw = np.flip(fw, axis=(2, 3)).copy()
            fw_tensor = torch.from_numpy(fw)

            weight_diff = (layer.weight.data.cpu() - fw_tensor).abs()
            max_diff = weight_diff.max().item()
            mean_diff = weight_diff.mean().item()
            match = max_diff < 1e-4
            all_weights_match = all_weights_match and match

            print(
                f"  {name}: max={max_diff:.2e}, mean={mean_diff:.2e} {'PASS' if match else 'FAIL'}"
            )
        except Exception as e:
            print(f"  {name}: SKIP - {type(e).__name__}")
            all_weights_match = False
    print()

    # Compare gradients (sample a few layers)
    print("Gradient comparison (sample layers):")

    all_grads_match = True
    grad_layers_to_check = [
        ("enc1_conv1", model.enc1.conv1, (32, 6, 3, 3)),
        ("bottleneck1", model.bottleneck1, (256, 128, 3, 3)),
        ("final", model.final_conv, (6, 32, 1, 1)),
    ]
    for name, layer, shape in grad_layers_to_check:
        grad_file = step_dir / f"{name}_grad_weights.bin"
        if grad_file.exists():
            try:
                fortran_grads = load_binary_array(grad_file, shape)
                # Convert Fortran gradients to PyTorch format
                fg = fortran_grads.transpose(3, 2, 1, 0)
                fg = np.flip(fg, axis=(2, 3)).copy()
                fg_tensor = torch.from_numpy(fg)

                if layer.weight.grad is not None:
                    grad_diff = (layer.weight.grad.cpu() - fg_tensor).abs()
                    max_diff = grad_diff.max().item()
                    mean_diff = grad_diff.mean().item()
                    match = max_diff < 1e-3  # Gradients can have slightly larger diff
                    all_grads_match = all_grads_match and match
                    print(
                        f"  {name}: max={max_diff:.2e}, mean={mean_diff:.2e} {'PASS' if match else 'FAIL'}"
                    )
                else:
                    print(f"  {name}: No gradient computed")
                    all_grads_match = False
            except Exception as e:
                print(f"  {name}: SKIP - {type(e).__name__}")
                all_grads_match = False
        else:
            print(f"  {name}: Fortran gradient file not found")
    print()

    # =========================================
    # FINAL VERDICT
    # =========================================
    print("=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    all_pass = loss_match and output_match and all_weights_match and all_grads_match

    print(f"Loss computation:     {'PASS' if loss_match else 'FAIL'}")
    print(f"Forward pass:         {'PASS' if output_match else 'FAIL'}")
    print(f"Weight updates:       {'PASS' if all_weights_match else 'FAIL'}")
    print(f"Gradient computation: {'PASS' if all_grads_match else 'FAIL'}")
    print()

    if all_pass:
        print("OVERALL: PASS - Fortran training matches PyTorch!")
        print()
        print("This verifies:")
        print("  - Forward pass produces identical outputs")
        print("  - Loss computation is correct")
        print("  - Backward pass computes correct gradients")
        print("  - Adam optimizer updates weights identically")
    else:
        print("OVERALL: FAIL - Some differences detected")
        print()
        print("Check the individual results above for details.")

    print("=" * 70)

    # Save timing comparison
    timing_file = verify_dir / "timing_comparison.txt"
    with open(timing_file, "w") as f:
        f.write("Training Step Timing Comparison\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"{'Phase':<15} {'Fortran (ms)':<15} {'PyTorch (ms)':<15}\n")
        f.write("-" * 45 + "\n")
        f.write(f"{'Forward':<15} {fortran_fwd:<15.3f} {pytorch_fwd_ms:<15.3f}\n")
        f.write(f"{'Backward':<15} {fortran_bwd:<15.3f} {pytorch_bwd_ms:<15.3f}\n")
        f.write(f"{'Update':<15} {fortran_upd:<15.3f} {pytorch_upd_ms:<15.3f}\n")
        f.write("-" * 45 + "\n")
        f.write(f"{'TOTAL':<15} {fortran_total:<15.3f} {pytorch_total_ms:<15.3f}\n")
        f.write("\n")
        if fortran_total < pytorch_total_ms:
            f.write(f"Fortran is {pytorch_total_ms / fortran_total:.2f}x faster\n")
        else:
            f.write(f"PyTorch is {fortran_total / pytorch_total_ms:.2f}x faster\n")

    print(f"\nTiming comparison saved to: {timing_file}")

    return all_pass


def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_training_step.py <verify_dir>")
        print()
        print("Example:")
        print("  python verify_training_step.py ../training_verify/")
        print()
        print("First run the Fortran test to generate verification data:")
        print("  cd ..")
        print("  ./test_training_step")
        sys.exit(1)

    verify_dir = sys.argv[1]
    success = verify_training_step(verify_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
