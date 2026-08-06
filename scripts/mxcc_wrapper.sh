#!/bin/bash
# MXMACA CUDA Compiler Wrapper
# Compiles CUDA code for MetaX C500 GPU using clang++-15

MACA_PATH="${MXMACA_PATH:-/opt/maca-3.2.1}"
CLANG="${MACA_PATH}/mxgpu_llvm/bin/clang++-15"

if [ ! -f "$CLANG" ]; then
    CLANG="clang++-15"
fi

TARGET="--target=mxg-metax-macahca -mcpu=xcore2000"
INCLUDES="-I${MACA_PATH}/include -I${MACA_PATH}/include/mcblas -I${MACA_PATH}/mxgpu_llvm/lib/clang/12.0.0/include"
DEFS="-D__MACACC__ -D__MACA_ARCH__ -D__HIP_PLATFORM_METAX__"
FLAGS="-O3 -fPIC -fno-strict-aliasing -fvisibility=protected"

# Determine if this is a compilation or linking step
if [[ "$*" == *"-c"* ]]; then
    # Compilation step - use -nocudalib and -nocudainc to avoid CUDA dependency
    exec $CLANG -x cuda -std=c++17 -nocudalib -nocudainc $TARGET $FLAGS $DEFS $INCLUDES "$@"
else
    # Linking step
    exec $CLANG $TARGET $FLAGS "$@" -L${MACA_PATH}/lib -lmc_runtime -lmcblas
fi
