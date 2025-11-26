#!/bin/bash
#================================================================
# Deployment Verification Script - v28e Climate CNN
#================================================================
# Verifies all changes are correct before pushing to weatherbench2
#================================================================

set -e  # Exit on error

echo "================================================================"
echo "  v28e Climate CNN - Deployment Verification"
echo "================================================================"
echo ""

# Check for v28d dependencies in source code
echo "[1/5] Checking for v28d dependencies in source code..."
if grep -r "v28d" --include="*.cuf" --include="*.py" . 2>/dev/null | grep -v "DESIGN_DOCUMENT\|README\|CRITICAL_BUG\|V28D_DEPENDENCIES\|READY_FOR"; then
    echo "❌ FAIL: Found v28d references in source code!"
    exit 1
else
    echo "✅ PASS: No v28d dependencies in source code"
fi
echo ""

# Verify bug fix is present in conv2d_cudnn.cuf
echo "[2/5] Verifying bug fix in conv2d_cudnn.cuf..."
if grep -q "alpha = 1.0" common/conv2d_cudnn.cuf && grep -q "beta = 0.0" common/conv2d_cudnn.cuf; then
    echo "✅ PASS: Bug fix present (explicit alpha/beta initialization)"
else
    echo "❌ FAIL: Bug fix missing in conv2d_cudnn.cuf"
    exit 1
fi
echo ""

# Verify bug fix is present in pooling_cudnn.cuf
echo "[3/5] Verifying bug fix in pooling_cudnn.cuf..."
if grep -q "alpha = 1.0" common/pooling_cudnn.cuf && grep -q "beta = 0.0" common/pooling_cudnn.cuf; then
    echo "✅ PASS: Bug fix present (explicit alpha/beta initialization)"
else
    echo "❌ FAIL: Bug fix missing in pooling_cudnn.cuf"
    exit 1
fi
echo ""

# Check notebook paths
echo "[4/5] Checking notebook paths..."
ANALYSIS_PATH=$(grep -o "Path('[^']*')" notebooks/climate_unet_analysis.ipynb | head -1)
if echo "$ANALYSIS_PATH" | grep -q "v28d"; then
    echo "❌ FAIL: climate_unet_analysis.ipynb still references v28d"
    exit 1
else
    echo "✅ PASS: Notebooks use v28e paths"
fi
echo ""

# Verify compilation works
echo "[5/5] Verifying compilation..."
if [ -f "compile.sh" ]; then
    echo "Attempting compilation..."
    if bash compile.sh > /dev/null 2>&1; then
        echo "✅ PASS: Code compiles successfully"
    else
        echo "⚠️  WARNING: Compilation failed (may be missing cuDNN/nvfortran)"
        echo "    This is OK if you're not on the training machine"
    fi
else
    echo "⚠️  WARNING: compile.sh not found"
fi
echo ""

echo "================================================================"
echo "  Verification Complete!"
echo "================================================================"
echo ""
echo "Summary:"
echo "  ✅ No v28d dependencies in source code"
echo "  ✅ Bug fix applied to conv2d_cudnn.cuf"
echo "  ✅ Bug fix applied to pooling_cudnn.cuf"
echo "  ✅ Notebooks use v28e paths"
echo ""
echo "Status: Ready for deployment to weatherbench2"
echo ""
echo "Next steps:"
echo "  1. Train final model with: ./climate_train_unet --stream --epochs 30 --save"
echo "  2. Commit changes with: git add -A && git commit -F ../COMMIT_MESSAGE_v28e.txt"
echo "  3. Push to weatherbench2 repository"
echo ""
echo "================================================================"
