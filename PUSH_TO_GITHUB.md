# Instructions: Push to WeatherBench2 Repository

This document provides step-by-step instructions for pushing the standalone v28e_climate_cnn to the WeatherBench2 GitHub repository.

## Prerequisites

1. You have created a GitHub repository at: https://github.com/frasertajima/WeatherBench2
2. You have SSH keys configured for GitHub or can use HTTPS with credentials
3. Git is installed on your system

## Option 1: Using Git Subtree (Recommended)

This method creates a clean repository containing only v28e_climate_cnn files.

### Step 1: Create a temporary working directory

```bash
cd /var/home/fraser/Downloads/CIFAR-10
mkdir -p /tmp/weatherbench2_push
```

### Step 2: Copy v28e_climate_cnn to temporary directory

```bash
# Copy all source files
rsync -av --exclude='.git' \
  --exclude='*.o' \
  --exclude='*.mod' \
  --exclude='test_*' \
  --exclude='climate_train_unet' \
  --exclude='__pycache__' \
  --exclude='saved_models' \
  --exclude='training_verify' \
  --exclude='data/climate_data_streaming' \
  --exclude='*.log' \
  --exclude='notebooks/*.png' \
  v28e_climate_cnn/ /tmp/weatherbench2_push/
```

### Step 3: Initialize git repository

```bash
cd /tmp/weatherbench2_push

# Initialize new git repository
git init

# Add .gitignore
git add .gitignore

# Add all source files
git add common/ data/ inference/ notebooks/ tests/
git add climate_train_unet.cuf compile.sh
git add README.md LICENSE
git add data/README_DATA.md

# Optional: Add design document if desired
# git add DESIGN_DOCUMENT.md

# Commit
git commit -m "Initial commit: Climate U-Net for WeatherBench2

- CUDA Fortran U-Net implementation for weather prediction
- Trains 72GB dataset on 8GB GPU using streaming
- cuDNN-accelerated with PyTorch verification
- 1.3x faster training than PyTorch
- Includes tests, inference verification, and Jupyter notebooks
"
```

### Step 4: Push to GitHub

```bash
# Add remote (use SSH)
git remote add origin git@github.com:frasertajima/WeatherBench2.git

# Or use HTTPS
# git remote add origin https://github.com/frasertajima/WeatherBench2.git

# Create main branch and push
git branch -M main
git push -u origin main
```

### Step 5: Clean up

```bash
cd /var/home/fraser/Downloads/CIFAR-10
rm -rf /tmp/weatherbench2_push
```

## Option 2: Manual GitHub Upload (Simpler, Less Git-Clean)

If you prefer a simpler approach without command-line git:

### Step 1: Create a clean directory

```bash
cd /var/home/fraser/Downloads/CIFAR-10
mkdir -p ~/weatherbench2_upload
```

### Step 2: Copy files manually

```bash
# Copy everything except build artifacts and data
rsync -av --exclude='.git' \
  --exclude='*.o' \
  --exclude='*.mod' \
  --exclude='test_*' \
  --exclude='climate_train_unet' \
  --exclude='__pycache__' \
  --exclude='saved_models' \
  --exclude='training_verify' \
  --exclude='data/climate_data_streaming' \
  --exclude='*.log' \
  --exclude='notebooks/*.png' \
  v28e_climate_cnn/ ~/weatherbench2_upload/
```

### Step 3: Upload via GitHub web interface

1. Go to https://github.com/frasertajima/WeatherBench2
2. Click "uploading an existing file"
3. Drag and drop all folders from ~/weatherbench2_upload/
4. Commit with message: "Initial commit: Climate U-Net for WeatherBench2"

## Option 3: Clone and Replace (For Existing Repo)

If you've already initialized the WeatherBench2 repository:

```bash
# Clone your repository
cd /var/home/fraser/Downloads/
git clone git@github.com:frasertajima/WeatherBench2.git
cd WeatherBench2

# Copy v28e_climate_cnn files
rsync -av --exclude='.git' \
  --exclude='*.o' \
  --exclude='*.mod' \
  --exclude='test_*' \
  --exclude='climate_train_unet' \
  --exclude='__pycache__' \
  --exclude='saved_models' \
  --exclude='training_verify' \
  --exclude='data/climate_data_streaming' \
  --exclude='*.log' \
  --exclude='notebooks/*.png' \
  ../CIFAR-10/v28e_climate_cnn/ ./

# Add and commit
git add .
git commit -m "Initial commit: Climate U-Net for WeatherBench2

- CUDA Fortran U-Net implementation for weather prediction
- Trains 72GB dataset on 8GB GPU using streaming
- cuDNN-accelerated with PyTorch verification
- 1.3x faster training than PyTorch
- Includes tests, inference verification, and Jupyter notebooks
"

# Push
git push origin main
```

## Verification

After pushing, verify the repository at:
https://github.com/frasertajima/WeatherBench2

Check that:
- ✓ README.md displays correctly on the homepage
- ✓ Source code files are present in common/, data/, inference/, notebooks/, tests/
- ✓ .gitignore is working (no .o, .mod, or data files)
- ✓ LICENSE file is present
- ✓ Compilation instructions work for fresh users

## Files to Include

### Required (must be present):
```
common/
  ├── cmdline_args.cuf
  ├── streaming_regression_loader.cuf
  ├── conv2d_cudnn.cuf
  ├── pooling_cudnn.cuf
  ├── unet_blocks.cuf
  ├── climate_unet.cuf
  ├── training_export.cuf
  └── unet_export.cuf

data/
  ├── climate_config.cuf
  └── README_DATA.md

inference/
  ├── climate_unet.py
  ├── verify_fortran_pytorch.py
  └── verify_training_step.py

notebooks/
  ├── climate_unet_analysis.ipynb
  └── climate_unet_evaluation.ipynb

tests/
  ├── test_conv2d.cuf
  ├── test_pooling.cuf
  ├── test_unet_blocks.cuf
  ├── test_climate_unet.cuf
  └── test_training_step.cuf

climate_train_unet.cuf
compile.sh
README.md
LICENSE
.gitignore
```

### Optional (can include):
```
DESIGN_DOCUMENT.md  # Technical design details
```

### Excluded (never include):
```
*.o, *.mod          # Compiled files
test_*, climate_train_unet  # Executables
__pycache__/        # Python cache
saved_models/       # Training checkpoints (too large)
data/climate_data_streaming/  # User provides their own data
*.log               # Training logs
notebooks/*.png     # Generated plots
```

## Post-Push: Update Repository Settings

On GitHub, consider:

1. **Add description**: "CUDA Fortran U-Net for weather prediction - trains 72GB WeatherBench2 on 8GB GPU"

2. **Add topics**: 
   - cuda-fortran
   - deep-learning
   - weather-prediction
   - weatherbench2
   - u-net
   - cudnn
   - high-performance-computing

3. **Update README**: If you want to add badges:
   ```markdown
   [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
   ```

4. **Create releases**: Tag the initial version as v1.0.0

## Troubleshooting

### Permission denied (publickey)

If you get SSH errors:
```bash
# Switch to HTTPS instead
git remote set-url origin https://github.com/frasertajima/WeatherBench2.git
```

### Repository already has content

If the repository isn't empty:
```bash
# Force push (WARNING: overwrites remote)
git push -f origin main

# Or better: pull and merge first
git pull origin main --allow-unrelated-histories
git push origin main
```

### Large files rejected

If you accidentally included data files:
```bash
# Remove from git history
git rm --cached -r data/climate_data_streaming
git commit -m "Remove data files"
git push origin main
```

## Next Steps After Pushing

1. **Test from fresh clone**:
   ```bash
   cd /tmp
   git clone https://github.com/frasertajima/WeatherBench2.git
   cd WeatherBench2
   ./compile.sh
   ```

2. **Update blog post** with new repository link

3. **Share**: Post on relevant forums, social media, etc.

4. **Monitor**: Watch for issues, pull requests, or questions from users
