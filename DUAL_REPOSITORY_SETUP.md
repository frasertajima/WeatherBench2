# Dual Repository Setup - v28e Climate CNN

## Overview

This project needs to be pushed to **two repositories**:
1. **Private CIFAR-10 repository** - Your complete ML research collection
2. **Public WeatherBench2 repository** - Standalone climate CNN for the community

## Current Status

- ✅ Currently in: CIFAR-10 repository (private)
- 🎯 Need to push to: WeatherBench2 repository (public)

## Strategy: Subtree Push

The best approach is to push **only the v28e_climate_cnn directory** to a separate public repository.

## Setup Instructions

### Option 1: Git Subtree Split (Recommended)

This creates a clean public repository with only v28e_climate_cnn and its history.

```bash
# 1. Commit current changes to CIFAR-10 (private)
cd /var/home/fraser/Downloads/CIFAR-10
git add v28e_climate_cnn/
git commit -F COMMIT_MESSAGE_v28e.txt
git push origin main

# 2. Create/clone your WeatherBench2 public repository
cd ~/Downloads
git clone git@github.com:frasertajima/WeatherBench2.git
# (Or create new: mkdir WeatherBench2 && cd WeatherBench2 && git init)

# 3. Copy v28e_climate_cnn to WeatherBench2 (preserving git history)
cd /var/home/fraser/Downloads/CIFAR-10
git subtree split --prefix=v28e_climate_cnn -b v28e-export

# 4. Push to WeatherBench2
cd ~/Downloads/WeatherBench2
git pull /var/home/fraser/Downloads/CIFAR-10 v28e-export
git push origin main

# 5. Clean up temporary branch
cd /var/home/fraser/Downloads/CIFAR-10
git branch -D v28e-export
```

### Option 2: Simple Copy (Easier, No History)

If you don't need git history in the public repo, just copy the files:

```bash
# 1. Commit to CIFAR-10 (private)
cd /var/home/fraser/Downloads/CIFAR-10
git add v28e_climate_cnn/
git commit -F COMMIT_MESSAGE_v28e.txt
git push origin main

# 2. Clone/create WeatherBench2 repository
cd ~/Downloads
git clone git@github.com:frasertajima/WeatherBench2.git
# (Or: mkdir WeatherBench2 && cd WeatherBench2 && git init)

# 3. Copy files (excluding private data)
cd ~/Downloads/WeatherBench2
cp -r /var/home/fraser/Downloads/CIFAR-10/v28e_climate_cnn/* .

# 4. Remove large data files (don't push to public GitHub!)
rm -rf climate_data_streaming/  # 72GB - don't push!
rm -rf saved_models/            # Large model files - don't push!
rm -rf data/                    # User should download their own

# 5. Create .gitignore for public repo
cat > .gitignore << 'EOF'
# Large data files - users should download themselves
climate_data_streaming/
saved_models/
data/

# Build artifacts
*.o
*.mod
*.a
climate_train_unet
test_*
!tests/test_*.cuf

# Python
__pycache__/
*.pyc
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
EOF

# 6. Commit and push
git add -A
git commit -m "Add v28e Climate U-Net with critical bug fix (19x improvement)"
git push origin main
```

### Option 3: Git Remote with Dual Push (Advanced)

Set up CIFAR-10 to push v28e to both repos simultaneously:

```bash
# Add WeatherBench2 as a second remote
cd /var/home/fraser/Downloads/CIFAR-10
git remote add weatherbench2 git@github.com:frasertajima/WeatherBench2.git

# Push only v28e subdirectory to weatherbench2
git subtree push --prefix=v28e_climate_cnn weatherbench2 main

# In future, use:
git push origin main                                    # Push all to CIFAR-10
git subtree push --prefix=v28e_climate_cnn weatherbench2 main  # Push v28e to public
```

## What to Include in Public Repository

### ✅ Include (Code & Documentation)
- All `.cuf` source files (common/, tests/, *.cuf)
- All Python scripts (inference/, notebooks/)
- Documentation (README.md, *.md files)
- Build scripts (compile.sh, verify_deployment.sh)
- LICENSE file
- .gitignore

### ❌ Exclude (Large Data)
- `climate_data_streaming/` (72GB - users download separately)
- `saved_models/` (large checkpoints - users train their own)
- `data/` directory (if it exists)
- Build artifacts (*.o, *.mod, executables)

## Public Repository README Updates

Before pushing to public, update README.md to:

1. **Add data download instructions**:
   ```markdown
   ## Data Setup
   
   Download WeatherBench2 ERA5 data:
   1. Visit https://weatherbench2.readthedocs.io/
   2. Download 6-hour data (1979-2020)
   3. Convert to streaming format (see scripts/)
   4. Place in climate_data_streaming/
   ```

2. **Update repository URL**:
   Change git clone URL to weatherbench2 repository

3. **Add citation**:
   ```markdown
   ## Citation
   
   If you use this code, please cite:
   
   @software{tajima2025climate,
     author = {Tajima, Fraser},
     title = {Climate U-Net: CUDA Fortran CNN for WeatherBench2},
     year = {2025},
     url = {https://github.com/frasertajima/WeatherBench2}
   }
   ```

## Recommended Workflow

**For ongoing development:**

1. Work in CIFAR-10 repository (private)
2. Test and verify changes
3. Commit to CIFAR-10: `git commit && git push origin main`
4. When ready for public release:
   ```bash
   git subtree push --prefix=v28e_climate_cnn weatherbench2 main
   ```

## Current Files to Push

Based on git status:
```
Modified:
  v28e_climate_cnn/common/conv2d_cudnn.cuf        # ✅ Critical bug fix
  v28e_climate_cnn/common/pooling_cudnn.cuf       # ✅ Critical bug fix
  v28e_climate_cnn/notebooks/*.ipynb              # ✅ Updated paths
  
New files:
  v28e_climate_cnn/CRITICAL_BUG_FIX_SUCCESS.md    # ✅ Documentation
  v28e_climate_cnn/DEPLOYMENT_SUMMARY.md          # ✅ Documentation
  v28e_climate_cnn/READY_FOR_DEPLOYMENT.md        # ✅ Documentation
  v28e_climate_cnn/verify_deployment.sh           # ✅ Verification script
  v28e_climate_cnn/common/cmdline_args.cuf        # ✅ New module
  v28e_climate_cnn/common/streaming_regression_loader.cuf  # ✅ New module
```

## Next Steps

1. ✅ Commit to CIFAR-10 (private) - preserves all your work
2. 🎯 Set up WeatherBench2 repository (public)
3. 🎯 Push v28e_climate_cnn to WeatherBench2 (choose option above)
4. 🎯 Update WeatherBench2 README with data download instructions

---

**Note**: The ACC=0.9789 result is incredible! This is publication-worthy. Consider writing a brief technical report or blog post about the 19x improvement from the bug fix.
