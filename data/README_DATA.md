# WeatherBench2 ERA5 Data Setup

This document describes how to download and prepare the WeatherBench2 ERA5 dataset for training the Climate U-Net model.

## Dataset Overview

- **Source**: ERA5 reanalysis data via WeatherBench2
- **Variables**: 6 atmospheric/surface variables
  - z500: Geopotential at 500 hPa
  - t850: Temperature at 850 hPa
  - u850: U-wind component at 850 hPa
  - v850: V-wind component at 850 hPa
  - t2m: 2-meter temperature
  - msl: Mean sea level pressure
- **Resolution**: 1.5° grid (240 lat × 121 lon)
- **Training period**: 1979-2016 (55,519 samples)
- **Test period**: 2017-2020 (5,843 samples)
- **Task**: Predict weather state at t+6h given state at t
- **Total size**: ~72GB (streaming binary format)

## Required Directory Structure

After setup, you should have:

```
data/
├── climate_config.cuf              # Dataset configuration (included)
├── climate_data_streaming/         # Binary data files (you create)
│   ├── train_input.bin             # Training inputs (~18GB)
│   ├── train_target.bin            # Training targets (~18GB)
│   ├── test_input.bin              # Test inputs (~2GB)
│   └── test_target.bin             # Test targets (~2GB)
└── README_DATA.md                  # This file
```

## Data Preparation

### Option 1: Download Pre-Converted Data (Recommended)

If you have access to pre-converted streaming binary files, copy them to `data/climate_data_streaming/`:

```bash
# Create data directory
mkdir -p data/climate_data_streaming

# Copy pre-converted files
cp /path/to/train_input.bin data/climate_data_streaming/
cp /path/to/train_target.bin data/climate_data_streaming/
cp /path/to/test_input.bin data/climate_data_streaming/
cp /path/to/test_target.bin data/climate_data_streaming/
```

### Option 2: Convert from WeatherBench2 NetCDF (Full Process)

If starting from scratch, you'll need to:

1. **Download ERA5 data from WeatherBench2**

   ```bash
   # Install WeatherBench2 tools (requires Python)
   pip install weatherbench2
   
   # Download ERA5 data (this will take a while and require ~100GB storage)
   # See: https://github.com/google-research/weatherbench2
   ```

2. **Convert NetCDF to streaming binary format**

   The streaming format is optimized for sequential reading during training:
   
   ```python
   import numpy as np
   import xarray as xr
   
   def convert_to_streaming_format(input_nc, output_bin, variables):
       """
       Convert WeatherBench2 NetCDF to streaming binary format.
       
       Streaming format:
       - Sample-major ordering (all channels for sample 0, then sample 1, ...)
       - Single precision (float32)
       - No metadata (raw binary data)
       - Shape: (num_samples, 6, 240, 121)
       """
       ds = xr.open_dataset(input_nc)
       
       # Extract variables and stack into shape (time, vars, lat, lon)
       data = []
       for var in variables:
           data.append(ds[var].values)
       data = np.stack(data, axis=1)  # (time, 6, lat, lon)
       
       # Ensure correct shape (time, 6, 240, 121)
       assert data.shape[1] == 6
       assert data.shape[2] == 240
       assert data.shape[3] == 121
       
       # Convert to float32 and write
       data = data.astype(np.float32)
       data.tofile(output_bin)
       
       print(f"Wrote {data.shape[0]} samples to {output_bin}")
       print(f"File size: {data.nbytes / 1024**3:.2f} GB")
   
   # Variables to extract
   variables = ['z500', 't850', 'u850', 'v850', 't2m', 'msl']
   
   # Convert training data (1979-2016)
   convert_to_streaming_format(
       'era5_1979-2016.nc',
       'data/climate_data_streaming/train_input.bin',
       variables
   )
   
   # Convert training targets (same data, shifted by 6 hours)
   # Note: Target creation requires temporal shifting
   # See create_6h_forecast_targets() function
   
   # Convert test data (2017-2020)
   convert_to_streaming_format(
       'era5_2017-2020.nc',
       'data/climate_data_streaming/test_input.bin',
       variables
   )
   ```

3. **Create forecast targets (6-hour lead time)**

   ```python
   def create_6h_forecast_targets(input_bin, output_bin, num_samples):
       """
       Create 6-hour forecast targets by shifting input data.
       
       target[t] = input[t+1]  (assuming 6h timesteps)
       """
       # Read all data
       data = np.fromfile(input_bin, dtype=np.float32)
       data = data.reshape(num_samples, 6, 240, 121)
       
       # Shift by 1 timestep (6 hours)
       targets = data[1:, :, :, :]  # Skip first sample
       inputs = data[:-1, :, :, :]  # Skip last sample
       
       # Write targets
       targets.tofile(output_bin)
       
       # Update input file to match target count
       inputs.tofile(input_bin)
       
       print(f"Created {targets.shape[0]} target samples")
   
   # Create targets for train and test
   create_6h_forecast_targets(
       'data/climate_data_streaming/train_input.bin',
       'data/climate_data_streaming/train_target.bin',
       num_samples=55520  # One more than final due to shifting
   )
   
   create_6h_forecast_targets(
       'data/climate_data_streaming/test_input.bin',
       'data/climate_data_streaming/test_target.bin',
       num_samples=5844
   )
   ```

## Verifying Data Setup

After preparing the data, verify the file sizes:

```bash
cd data/climate_data_streaming
ls -lh

# Expected output (approximate):
# train_input.bin   ~18GB  (55519 samples × 6 vars × 240 × 121 × 4 bytes)
# train_target.bin  ~18GB  (55519 samples × 6 vars × 240 × 121 × 4 bytes)
# test_input.bin    ~2GB   (5843 samples × 6 vars × 240 × 121 × 4 bytes)
# test_target.bin   ~2GB   (5843 samples × 6 vars × 240 × 121 × 4 bytes)
```

You can also verify with Python:

```python
import numpy as np

def verify_binary_file(filepath, expected_samples):
    """Verify streaming binary file format."""
    data = np.fromfile(filepath, dtype=np.float32)
    
    # Expected shape: (samples, 6, 240, 121)
    expected_size = expected_samples * 6 * 240 * 121
    
    if data.size != expected_size:
        print(f"ERROR: {filepath}")
        print(f"  Expected {expected_size} values, got {data.size}")
        return False
    
    # Reshape to verify
    data = data.reshape(expected_samples, 6, 240, 121)
    
    print(f"✓ {filepath}")
    print(f"  Shape: {data.shape}")
    print(f"  Min: {data.min():.2f}, Max: {data.max():.2f}")
    print(f"  Mean: {data.mean():.2f}, Std: {data.std():.2f}")
    
    return True

# Verify all files
verify_binary_file('train_input.bin', 55519)
verify_binary_file('train_target.bin', 55519)
verify_binary_file('test_input.bin', 5843)
verify_binary_file('test_target.bin', 5843)
```

## Configuration

The dataset configuration is in `climate_config.cuf`. Key settings:

```fortran
! Data paths (relative to training program execution directory)
data_dir = "data/climate_data_streaming"

! Dataset dimensions
num_train = 55519       ! Training samples (1979-2016)
num_test = 5843         ! Test samples (2017-2020)
input_channels = 6      ! z500, t850, u850, v850, t2m, msl
output_channels = 6     ! Same variables (6h forecast)
height = 240            ! Latitude points
width = 121             ! Longitude points

! File names
train_input_file = "train_input.bin"
train_target_file = "train_target.bin"
test_input_file = "test_input.bin"
test_target_file = "test_target.bin"
```

If you change the data location, update the `--data` flag when training:

```bash
./climate_train_unet --stream --data /path/to/your/data
```

## Troubleshooting

### "File not found" error during training

**Problem**: Training program can't find data files.

**Solution**: Check that:
1. Data files exist in `data/climate_data_streaming/`
2. You're running from the repository root directory
3. Or use `--data` flag to specify custom location

### "Invalid file size" or dimension mismatch

**Problem**: Binary files don't match expected dimensions.

**Solution**: 
1. Verify file sizes (see "Verifying Data Setup" above)
2. Check that streaming format is correct (sample-major, float32)
3. Ensure NetCDF → binary conversion used correct variable ordering

### Out of disk space

**Problem**: Not enough space for 72GB dataset.

**Solution**:
1. Use smaller subset of years for testing
2. Convert data on external drive
3. Use symbolic links to store data elsewhere:
   ```bash
   ln -s /external/drive/climate_data_streaming data/climate_data_streaming
   ```

## Data Format Specification

### Streaming Binary Format

Files are raw binary (no headers, no compression):

```
Layout: Sample-major ordering
Type: float32 (4 bytes per value)
Endianness: Native (little-endian on x86-64)

Structure:
  sample_0:
    channel_0: [height × width] = [240 × 121] values
    channel_1: [height × width] = [240 × 121] values
    ...
    channel_5: [height × width] = [240 × 121] values
  sample_1:
    channel_0: [height × width] = [240 × 121] values
    ...
  
Total per file: num_samples × 6 × 240 × 121 × 4 bytes
```

### Variable Ordering

Channels are ordered as defined in `climate_config.cuf`:

```
Channel 0: z500  - Geopotential at 500 hPa
Channel 1: t850  - Temperature at 850 hPa  
Channel 2: u850  - U-wind component at 850 hPa
Channel 3: v850  - V-wind component at 850 hPa
Channel 4: t2m   - 2-meter temperature
Channel 5: msl   - Mean sea level pressure
```

This order must match the WeatherBench2 NetCDF variable extraction.

## References

- [WeatherBench2 GitHub](https://github.com/google-research/weatherbench2)
- [ERA5 Documentation](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5)
- [WeatherBench2 Paper](https://arxiv.org/abs/2308.15560)

## Need Help?

If you're unable to prepare the data or need pre-converted files:

1. Check the WeatherBench2 repository for official data sources
2. Open an issue on the GitHub repository
3. Contact the author for assistance with large file transfers
