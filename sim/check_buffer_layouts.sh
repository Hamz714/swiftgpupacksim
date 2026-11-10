#!/bin/bash

# Correctness check for --with-buffer-layout.
#
# For every particle memory layout and loop splitting variant (global-var
# particle access), build an aos-buffer and a soa-buffer binary with
# --enable-buffer-dump, run both on a small test log, and compare the dumps
# (buffer contents after each pack/unpack plus the final particle state).
#
# The dumps are written in a canonical order that does not depend on the buffer
# layout, so the aos-buffer and soa-buffer dumps must be bitwise identical.
# This (and the final particle state) is what counts as pass/fail.
#
# For information, each dump is also compared against the first loop split
# checked for the same particle layout ("vs-first-split"). These can differ
# without a buffer problem, because some loop splitting variants do not pack
# exactly the same values as split "none" for every particle layout (e.g.
# by-struct with the pack-gradient/pack-force/pack-shared layouts). Dumps of
# different particle layouts are not compared, since the aos and upstream
# layouts store some density and force quantities in a union.
#
# Usage:
#   ./check_buffer_layouts.sh [options]
# Options:
#   -d, --data <dir>      test log directory
#                         (default: ../data/IntelXeonGold5218_Gresho64_4threads)
#   -w, --workdir <dir>   where to build and run (default: ./buffer_check)
#   -s, --steps <n>       number of steps to replay (default: 1)
#   -l, --layouts "<..>"  particle layouts to check (default: all)
#   -p, --splits "<..>"   loop splits to check (default: all)
#   -g, --debug           also configure with --enable-debugging-checks
#   -c, --cuda <path>     CUDA installation (default: $CUDA_HOME, or the COSMA
#                         installation)
#   -m, --mallochost      use cudaMallocHost (needs a GPU; default: plain
#                         aligned malloc, --enable-no-cuda-mallochost)
#   -k, --keep            keep the dump files
#
# Results are summarised in <workdir>/summary.txt. Exits with status 1 if any
# check failed.

set -u

SIMDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA="$SIMDIR/../data/IntelXeonGold5218_Gresho64_4threads"
WORKDIR="$SIMDIR/buffer_check"
STEPS=1
LAYOUTS="aos soa upstream pack-gradient pack-gradient-type pack-force pack-force-type pack-shared pack-shared-type random-order"
SPLITS="none by-struct by-element by-type by-struct-and-type"
EXTRA_CONFFLAGS="--enable-no-cuda-mallochost"
CUDA="${CUDA_HOME:-/cosma/local/Nvidia-hpc-sdk/2024_245/Linux_x86_64/24.5/cuda}"
KEEP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
  -d | --data) DATA="$2"; shift ;;
  -w | --workdir) WORKDIR="$2"; shift ;;
  -s | --steps) STEPS="$2"; shift ;;
  -l | --layouts) LAYOUTS="$2"; shift ;;
  -p | --splits) SPLITS="$2"; shift ;;
  -g | --debug) EXTRA_CONFFLAGS="$EXTRA_CONFFLAGS --enable-debugging-checks" ;;
  -c | --cuda) CUDA="$2"; shift ;;
  -m | --mallochost) EXTRA_CONFFLAGS="${EXTRA_CONFFLAGS/--enable-no-cuda-mallochost/}" ;;
  -k | --keep) KEEP=1 ;;
  *) echo "unknown option '$1'"; exit 2 ;;
  esac
  shift
done

DATA="$(cd "$DATA" && pwd)" || { echo "data dir not found"; exit 2; }
mkdir -p "$WORKDIR"
WORKDIR="$(cd "$WORKDIR" && pwd)"
SUMMARY="$WORKDIR/summary.txt"

# Build in a copy of the source tree: builds are in-tree, and this way the
# check doesn't touch the configuration of $SIMDIR.
BUILDDIR="$WORKDIR/sim"
rsync -a --delete --exclude '/swiftgpupack' --exclude '/swiftgpupack_*' \
  --exclude '/buffer_check' "$SIMDIR/" "$BUILDDIR/"
cd "$BUILDDIR"
./autogen.sh > "$WORKDIR/autogen.log" 2>&1 || { echo "autogen failed"; exit 1; }

{
  echo "# buffer layout correctness check"
  echo "# date: $(date)"
  echo "# git: $(git -C "$SIMDIR" rev-parse --short HEAD 2>/dev/null)"
  echo "# data: $DATA, steps: $STEPS"
  echo "# extra configure flags: $EXTRA_CONFFLAGS"
  echo "# layout split aos-vs-soa vs-first-split(info) sha256(aos) final-state-ok"
} > "$SUMMARY"

nfail=0

build() {
  # build <layout> <split> <buffer> -> prints binary path
  local layout=$1 split=$2 buffer=$3
  local log="$WORKDIR/build_${layout}_${split}_${buffer}.log"
  CFLAGS="-march=native" ./configure \
    --with-particle-memory-layout="$layout" --with-particle-access=global-var \
    --with-loop-split="$split" --with-buffer-layout="$buffer" \
    --with-cuda="$CUDA" --enable-buffer-dump $EXTRA_CONFFLAGS > "$log" 2>&1 &&
    make clean >> "$log" 2>&1 &&
    make -j 8 >> "$log" 2>&1 ||
    return 1
  local bin="$WORKDIR/swiftgpupack_${layout}_${split}_${buffer}"
  mv swiftgpupack "$bin"
  echo "$bin"
}

run() {
  # run <binary> <rundir> -> dump in <rundir>/gpu_buffer_dump.bin
  local bin=$1 rundir=$2
  mkdir -p "$rundir"
  (cd "$rundir" && "$bin" -s "$STEPS" --noflush "$DATA" > run.log 2>&1)
}

for layout in $LAYOUTS; do
  ref_hash=""
  for split in $SPLITS; do
    echo "--- $layout $split"
    status="OK"
    declare -A dump=()
    for buffer in aos soa; do
      bin=$(build "$layout" "$split" "$buffer") || {
        echo "BUILD FAILED: $layout $split $buffer"
        status="BUILD_FAIL"
        break
      }
      rundir="$WORKDIR/run_${layout}_${split}_${buffer}"
      if ! run "$bin" "$rundir"; then
        echo "RUN FAILED: $layout $split $buffer (see $rundir/run.log)"
        status="RUN_FAIL"
        break
      fi
      dump[$buffer]="$rundir/gpu_buffer_dump.bin"
    done

    if [[ "$status" != "OK" ]]; then
      echo "$layout $split $status - - -" >> "$SUMMARY"
      nfail=$((nfail + 1))
      continue
    fi

    # aos buffer vs soa buffer
    if cmp -s "${dump[aos]}" "${dump[soa]}"; then
      same="SAME"
    else
      same="DIFFER"
      nfail=$((nfail + 1))
      cmp "${dump[aos]}" "${dump[soa]}" | head -1
    fi

    # against the first split of this particle layout (informational only)
    hash=$(sha256sum "${dump[aos]}" | cut -d' ' -f1)
    if [[ -z "$ref_hash" ]]; then
      ref_hash=$hash
      vsref="REFERENCE"
    elif [[ "$hash" == "$ref_hash" ]]; then
      vsref="SAME"
    else
      vsref="DIFFER"
    fi

    # the final particle state is the last record; compare it on its own
    # too, so a mismatch can be attributed (pack/unpack vs particle data)
    final_state=$(python3 - "${dump[aos]}" "${dump[soa]}" << 'EOF'
import struct, sys
def last_record(fn):
    with open(fn, "rb") as fp:
        data = fp.read()
    # walk the records: header int32 x4 then payload
    pos, last = 0, None
    while pos < len(data):
        tag, sub, ind, count = struct.unpack_from("4i", data, pos)
        start = pos
        pos += 16
        if tag == 3:
            last = data[start:]
            break
        nbytes = {(1, 1): 10, (1, 2): 17, (1, 3): 20,
                  (2, 1): 8, (2, 2): 3, (2, 3): 6}[(tag, sub)] * 4 * count
        pos += nbytes
    return last
a, b = (last_record(f) for f in sys.argv[1:3])
print("OK" if a is not None and a == b else "DIFFER")
EOF
    )
    [[ "$final_state" == "OK" ]] || nfail=$((nfail + 1))

    echo "$layout $split $same $vsref $hash $final_state" >> "$SUMMARY"
    echo "$layout $split aos-vs-soa:$same vs-ref:$vsref final-state:$final_state"

    if [[ $KEEP -eq 0 ]]; then
      rm -f "${dump[aos]}" "${dump[soa]}"
    fi
    unset dump
  done
done

echo "# failures: $nfail" >> "$SUMMARY"
echo
cat "$SUMMARY"
[[ $nfail -eq 0 ]]
