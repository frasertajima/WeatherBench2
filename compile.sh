#!/bin/bash
#================================================================
# v28e Climate CNN - Compilation Script
#================================================================

set -e

echo "=================================================="
echo "  v28e Climate CNN - Compilation"
echo "=================================================="

# Compiler settings
FC="nvfortran"
FFLAGS="-cuda -O3"
LIBS="-lcudnn -lcublas -lcudart"

# Directories
COMMON_DIR="common"
TEST_DIR="tests"
DATA_DIR="data"

# Build common modules first
echo ""
echo "Compiling common modules..."

echo "  conv2d_cudnn.cuf"
$FC $FFLAGS -c $COMMON_DIR/conv2d_cudnn.cuf -o conv2d_cudnn.o

# Build tests
echo ""
echo "Compiling tests..."

echo "  test_conv2d.cuf"
$FC $FFLAGS -o test_conv2d conv2d_cudnn.o $TEST_DIR/test_conv2d.cuf $LIBS

echo "  pooling_cudnn.cuf"
$FC $FFLAGS -c $COMMON_DIR/pooling_cudnn.cuf -o pooling_cudnn.o

echo "  unet_blocks.cuf"
$FC $FFLAGS -c $COMMON_DIR/unet_blocks.cuf -o unet_blocks.o

echo "  test_pooling.cuf"
$FC $FFLAGS -o test_pooling pooling_cudnn.o $TEST_DIR/test_pooling.cuf $LIBS

echo "  test_unet_blocks.cuf"
$FC $FFLAGS -o test_unet_blocks conv2d_cudnn.o pooling_cudnn.o unet_blocks.o $TEST_DIR/test_unet_blocks.cuf $LIBS

echo "  climate_unet.cuf"
$FC $FFLAGS -c $COMMON_DIR/climate_unet.cuf -o climate_unet.o

echo "  unet_export.cuf"
$FC $FFLAGS -c $COMMON_DIR/unet_export.cuf -o unet_export.o

echo "  test_climate_unet.cuf"
$FC $FFLAGS -o test_climate_unet conv2d_cudnn.o pooling_cudnn.o unet_blocks.o climate_unet.o $TEST_DIR/test_climate_unet.cuf $LIBS

echo "  training_export.cuf"
$FC $FFLAGS -c $COMMON_DIR/training_export.cuf -o training_export.o

echo "  test_training_step.cuf"
$FC $FFLAGS -o test_training_step conv2d_cudnn.o pooling_cudnn.o unet_blocks.o climate_unet.o unet_export.o training_export.o $TEST_DIR/test_training_step.cuf $LIBS

echo ""
echo "=================================================="
echo "  Compiling data/config modules..."
echo "=================================================="

# Compile common modules needed for streaming
echo "  cmdline_args.cuf"
$FC $FFLAGS -c $COMMON_DIR/cmdline_args.cuf -o cmdline_args.o

echo "  streaming_regression_loader.cuf"
$FC $FFLAGS -c $COMMON_DIR/streaming_regression_loader.cuf -o streaming_regression_loader.o

echo "  climate_config.cuf"
$FC $FFLAGS -c $DATA_DIR/climate_config.cuf -o climate_config.o

echo ""
echo "=================================================="
echo "  Compiling training program..."
echo "=================================================="

echo "  climate_train_unet.cuf"
$FC $FFLAGS -o climate_train_unet \
    conv2d_cudnn.o pooling_cudnn.o unet_blocks.o climate_unet.o unet_export.o \
    cmdline_args.o streaming_regression_loader.o climate_config.o \
    climate_train_unet.cuf $LIBS

echo ""
echo "=================================================="
echo "  Compilation complete!"
echo "=================================================="
echo ""
echo "Run tests:"
echo "  ./test_conv2d"
echo "  ./test_pooling"
echo "  ./test_unet_blocks"
echo "  ./test_climate_unet"
echo ""
echo "Run training (after setting up data - see README):"
echo "  ./climate_train_unet --stream --data data/climate_data_streaming"
echo ""
echo "Run training verification:"
echo "  ./test_training_step"
echo "  cd inference && python verify_training_step.py ../training_verify/"
echo ""
