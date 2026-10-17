#!/usr/bin/env bash
# Build and run the Omegahee Metal kernel pack.
#
#   bash metal/run.sh
#
# Steps: generate parity vectors with the numpy reference, compile the Metal
# kernels, compile the Swift harness, run it, and print metal/build/report.json.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
METAL_DIR="$ROOT/metal"
BUILD="$METAL_DIR/build"
PYTHON="${OMEGAHEE_PYTHON:-$ROOT/.venv/bin/python}"

if [ ! -x "$PYTHON" ]; then
  PYTHON="$(command -v python3)"
fi

mkdir -p "$BUILD/vectors"

echo "[1/4] generating parity vectors (numpy reference)"
"$PYTHON" "$METAL_DIR/tools/gen_vectors.py" > /dev/null

echo "[2/4] compiling Metal kernels"
xcrun -sdk macosx metal -O2 -c "$METAL_DIR/kernels/omegahee.metal" -o "$BUILD/omegahee.air"
xcrun -sdk macosx metallib "$BUILD/omegahee.air" -o "$BUILD/omegahee.metallib"

echo "[3/4] compiling Swift harness"
xcrun -sdk macosx swiftc -O "$METAL_DIR/harness/main.swift" -o "$BUILD/omegahee-harness" -framework Metal -framework Foundation

echo "[4/4] running parity and bandwidth checks"
"$BUILD/omegahee-harness"
