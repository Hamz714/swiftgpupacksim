#!/bin/bash

# Build all global-var executables for the GPU buffer layout comparison:
# buffer layout (aos, soa) x loop split (5) x particle layout (10).
#
# Executables are named
#   swiftgpupack_<node>_<layout>_global-var_<split>_<buffer>
#
# Since we compile with -march=native, run this on the node type the
# executables are meant for (i.e. inside the Slurm job, see
# examples/job_buffer_layouts_*.slm).
#
# Usage:
#   ./make_buffer_layout_variants.sh <node> <outdir> [options]
# Options:
#   -b, --buffers "<..>"  buffer layouts to build (default: "aos soa")
#   -c, --cuda <path>     CUDA installation passed to --with-cuda
#                         (default: let configure find nvcc)
#   -j <n>                parallel make jobs (default: 16)
#   -f, --force           rebuild executables that already exist
#   -d, --debug           add --enable-debug --enable-debugging-checks
# Environment:
#   MYCFLAGS              CFLAGS (default: "-march=native")
#   EXTRA_CONFFLAGS       extra configure flags
#   SPLITS, LAYOUTS       restrict the loop splits / particle layouts built
#                         (default: all)
#
# The build happens in-tree in the directory containing this script, so don't
# run two instances on the same source tree at the same time.

set -e

if [[ $# -lt 2 ]]; then
  head -30 "$0" | grep '^#' | grep -v '/bin/bash'
  exit 1
fi

NODE="$1"
OUTDIR="$2"
shift 2

BUFFERS="aos soa"
CUDA_FLAG=""
NJOBS=16
FORCE=0
MYCFLAGS="${MYCFLAGS:--march=native}"
EXTRA_CONFFLAGS="${EXTRA_CONFFLAGS:-} --config-cache"

while [[ $# -gt 0 ]]; do
  case "$1" in
  -b | --buffers) BUFFERS="$2"; shift ;;
  -c | --cuda) CUDA_FLAG="--with-cuda=$2"; shift ;;
  -j) NJOBS="$2"; shift ;;
  -f | --force) FORCE=1 ;;
  -d | --debug) EXTRA_CONFFLAGS="$EXTRA_CONFFLAGS --enable-debug --enable-debugging-checks" ;;
  *) echo "unknown option '$1'"; exit 1 ;;
  esac
  shift
done

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"

if [[ ! -f ./configure || configure.ac -nt configure ]]; then
  ./autogen.sh
fi

rm -f config.cache

part_access="global-var"
SPLITS="${SPLITS:-none by-struct by-element by-type by-struct-and-type}"
LAYOUTS="${LAYOUTS:-aos soa upstream pack-gradient pack-force pack-shared pack-force-type random-order pack-gradient-type pack-shared-type}"

for buffer in $BUFFERS; do
  for loop in $SPLITS; do
    for layout in $LAYOUTS; do

      exe="$OUTDIR"/swiftgpupack_"$NODE"_"$layout"_"$part_access"_"$loop"_"$buffer"

      if [[ -f "$exe" && $FORCE -eq 0 ]]; then
        echo "found $exe, not rebuilding"
        continue
      fi

      CFLAGS="$MYCFLAGS" ./configure \
        --with-particle-memory-layout="$layout" \
        --with-particle-access="$part_access" \
        --with-loop-split="$loop" \
        --with-buffer-layout="$buffer" \
        $CUDA_FLAG $EXTRA_CONFFLAGS
      make clean
      make -j "$NJOBS"
      mv ./swiftgpupack "$exe"
      echo
      echo COMPILED "$exe"
      echo

    done
  done
done
