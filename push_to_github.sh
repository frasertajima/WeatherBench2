#!/bin/bash
#
# Script to push v28e_climate_cnn to WeatherBench2 repository
# Usage: ./push_to_github.sh
#

set -e  # Exit on error

echo "========================================================================"
echo "  Push v28e_climate_cnn to WeatherBench2 Repository"
echo "========================================================================"
echo ""

# Configuration
REPO_URL="git@github.com:frasertajima/WeatherBench2.git"
TEMP_DIR="/tmp/weatherbench2_push_$$"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Source directory: $SOURCE_DIR"
echo "Temporary directory: $TEMP_DIR"
echo "Repository: $REPO_URL"
echo ""

# Ask for confirmation
read -p "Proceed with push? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Step 1: Create temporary directory
echo ""
echo "Step 1: Creating temporary directory..."
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR"

# Step 2: Copy files (excluding build artifacts and data)
echo ""
echo "Step 2: Copying source files..."
rsync -av --exclude='.git' \
  --exclude='*.o' \
  --exclude='*.mod' \
  --exclude='test_conv2d' \
  --exclude='test_pooling' \
  --exclude='test_unet_blocks' \
  --exclude='test_climate_unet' \
  --exclude='test_training_step' \
  --exclude='climate_train_unet' \
  --exclude='__pycache__' \
  --exclude='.ipynb_checkpoints' \
  --exclude='saved_models' \
  --exclude='training_verify' \
  --exclude='data/climate_data_streaming' \
  --exclude='climate_data_streaming' \
  --exclude='*.log' \
  --exclude='notebooks/*.png' \
  --exclude='push_to_github.sh' \
  --exclude='PUSH_TO_GITHUB.md' \
  "$SOURCE_DIR/" ./

# Step 3: Initialize git repository
echo ""
echo "Step 3: Initializing git repository..."
git init
git branch -M main

# Step 4: Add files
echo ""
echo "Step 4: Adding files to git..."
git add .gitignore
git add common/ data/ inference/ notebooks/ tests/
git add climate_train_unet.cuf compile.sh
git add README.md LICENSE

# Optional: Add design document
if [ -f "DESIGN_DOCUMENT.md" ]; then
    echo "Found DESIGN_DOCUMENT.md - adding to repository"
    git add DESIGN_DOCUMENT.md
fi

# Step 5: Commit
echo ""
echo "Step 5: Creating commit..."
git commit -m "Initial commit: Climate U-Net for WeatherBench2

- CUDA Fortran U-Net implementation for weather prediction
- Trains 72GB dataset on 8GB GPU using streaming
- cuDNN-accelerated with PyTorch verification
- 1.3x faster training than PyTorch
- Includes tests, inference verification, and Jupyter notebooks
"

# Step 6: Show summary
echo ""
echo "========================================================================"
echo "  Repository Summary"
echo "========================================================================"
git log --oneline
echo ""
echo "Files to be pushed:"
git ls-files | head -20
echo "..."
echo "Total files: $(git ls-files | wc -l)"
echo ""

# Step 7: Push to GitHub
echo "========================================================================"
echo "  Ready to Push"
echo "========================================================================"
echo ""
read -p "Push to $REPO_URL? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Step 7: Adding remote and pushing..."
    git remote add origin "$REPO_URL"

    # Try to push
    if git push -u origin main; then
        echo ""
        echo "========================================================================"
        echo "  SUCCESS!"
        echo "========================================================================"
        echo ""
        echo "Repository pushed to: https://github.com/frasertajima/WeatherBench2"
        echo ""
        echo "Next steps:"
        echo "  1. Visit the repository and verify files"
        echo "  2. Update repository description and topics"
        echo "  3. Test fresh clone: git clone $REPO_URL"
        echo ""
    else
        echo ""
        echo "========================================================================"
        echo "  Push Failed"
        echo "========================================================================"
        echo ""
        echo "This might be because:"
        echo "  1. Repository already has content (use --force if intentional)"
        echo "  2. SSH keys not configured (try HTTPS instead)"
        echo "  3. No write permissions"
        echo ""
        echo "To retry with force (WARNING: overwrites remote):"
        echo "  cd $TEMP_DIR && git push -f origin main"
        echo ""
        echo "To use HTTPS instead:"
        echo "  cd $TEMP_DIR"
        echo "  git remote set-url origin https://github.com/frasertajima/WeatherBench2.git"
        echo "  git push -u origin main"
        echo ""
        echo "Temporary directory preserved at: $TEMP_DIR"
        exit 1
    fi
else
    echo ""
    echo "Push cancelled."
    echo "Temporary directory preserved at: $TEMP_DIR"
    echo ""
    echo "To push manually:"
    echo "  cd $TEMP_DIR"
    echo "  git remote add origin $REPO_URL"
    echo "  git push -u origin main"
    echo ""
    exit 0
fi

# Step 8: Clean up
echo ""
read -p "Clean up temporary directory? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd /
    rm -rf "$TEMP_DIR"
    echo "Temporary directory removed."
else
    echo "Temporary directory preserved at: $TEMP_DIR"
fi

echo ""
echo "Done!"
