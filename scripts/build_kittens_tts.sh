#!/usr/bin/env bash
# Build src/kittens-tts.c against vendors/llama.cpp's already-built static libs.
# Produces tmp/kittens-tts-cpu and tmp/kittens-tts-metal.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA="$ROOT/vendors/llama.cpp"
SRC="$ROOT/Sources/KittenApp/KittensGGML/kittens-tts.c"
OUT="$ROOT/tmp"
mkdir -p "$OUT"

# Common flags
CFLAGS="-O3 -std=c11 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -DKT_BUILD_CLI"
INCLUDES="-I$LLAMA/ggml/include"

# CPU build: ggml + ggml-cpu + ggml-base + Accelerate (BLAS).
build_cpu() {
    local BUILD="$LLAMA/build-cpu"
    local LIBS=(
        "$BUILD/ggml/src/libggml.a"
        "$BUILD/ggml/src/libggml-cpu.a"
        "$BUILD/ggml/src/libggml-base.a"
        "$BUILD/ggml/src/ggml-blas/libggml-blas.a"
    )
    cc $CFLAGS $INCLUDES "$SRC" -o "$OUT/kittens-tts-cpu" \
        "${LIBS[@]}" \
        -framework Accelerate -framework Foundation \
        -lc++ -lpthread
    echo "built: $OUT/kittens-tts-cpu"
}

# Metal build: adds ggml-metal (with the embedded default.metallib) and Metal frameworks.
build_metal() {
    local BUILD="$LLAMA/build-metal"
    local LIBS=(
        "$BUILD/ggml/src/libggml.a"
        "$BUILD/ggml/src/libggml-cpu.a"
        "$BUILD/ggml/src/libggml-base.a"
        "$BUILD/ggml/src/ggml-blas/libggml-blas.a"
        "$BUILD/ggml/src/ggml-metal/libggml-metal.a"
    )
    cc $CFLAGS -DKT_HAVE_METAL $INCLUDES "$SRC" -o "$OUT/kittens-tts-metal" \
        "${LIBS[@]}" \
        -framework Accelerate -framework Foundation \
        -framework Metal -framework MetalKit -framework MetalPerformanceShaders \
        -framework CoreGraphics \
        -lc++ -lpthread
    echo "built: $OUT/kittens-tts-metal"
}

build_lstm_test() {
    local TGT_BACKEND="$1"  # cpu or metal
    local BUILD="$LLAMA/build-${TGT_BACKEND}"
    local LIBS=(
        "$BUILD/ggml/src/libggml.a"
        "$BUILD/ggml/src/libggml-cpu.a"
        "$BUILD/ggml/src/libggml-base.a"
        "$BUILD/ggml/src/ggml-blas/libggml-blas.a"
    )
    local DEFS=""
    local FRAMEWORKS=""
    if [ "$TGT_BACKEND" = "metal" ]; then
        LIBS+=("$BUILD/ggml/src/ggml-metal/libggml-metal.a")
        DEFS="-DKT_HAVE_METAL"
        FRAMEWORKS="-framework Metal -framework MetalKit -framework MetalPerformanceShaders -framework CoreGraphics"
    fi
    cc $CFLAGS $DEFS $INCLUDES "$ROOT/src/test_lstm.c" -o "$OUT/test-lstm-${TGT_BACKEND}" \
        "${LIBS[@]}" \
        -framework Accelerate -framework Foundation $FRAMEWORKS \
        -lc++ -lpthread
    echo "built: $OUT/test-lstm-${TGT_BACKEND}"
}

case "${1:-both}" in
    cpu)         build_cpu ;;
    metal)       build_metal ;;
    both)        build_cpu; build_metal ;;
    lstm-cpu)    build_lstm_test cpu ;;
    lstm-metal)  build_lstm_test metal ;;
    lstm)        build_lstm_test cpu; build_lstm_test metal ;;
    all)         build_cpu; build_metal; build_lstm_test cpu; build_lstm_test metal ;;
    *)           echo "usage: $0 [cpu|metal|both|lstm-cpu|lstm-metal|lstm|all]"; exit 1 ;;
esac
