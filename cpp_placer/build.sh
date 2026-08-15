#!/usr/bin/env bash
# Builds the C++20 placer into a shared library loaded by placement.py via ctypes.
# No dependencies beyond a C++20 compiler.
set -euo pipefail
cd "$(dirname "$0")"

CXX="${CXX:-c++}"
case "$(uname -s)" in
  Darwin) EXT=dylib ;;
  *)      EXT=so ;;
esac

mkdir -p build
"$CXX" -std=c++20 -O3 -DNDEBUG -ffast-math -fno-finite-math-only \
       -shared -fPIC src/partcl_place.cpp -o "build/libpartcl_place.$EXT"
echo "built build/libpartcl_place.$EXT"
