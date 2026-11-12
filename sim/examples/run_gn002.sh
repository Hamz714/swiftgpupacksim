#!/usr/bin/bash
#
# Run the GPU transfer buffer layout comparison on gn002 (Grace Hopper,
# aarch64), which is not in Slurm. Runs the same steps as the gn003 chain of
# job_buffer_layouts_gn003.slm, one after the other, with NODE=gn002:
#
#   0. toolchain check (MODE=env): prints the chosen nvcc, compiler, likwid-pin
#   1. build BUFFER=aos      2. build BUFFER=soa
#   3-10. runs for aos and soa x EAGLE25_72threads, Gresho256_72threads,
#         EAGLE25_36threads, Gresho256_36threads
#
# Resumable: existing executables and results are skipped, so after an
# interruption simply start it again.
#
# Usage (on gn002, ideally inside tmux):
#   bash run_gn002.sh [--dry-run] > <RUNDIR>/gn002_driver.log 2>&1
#
# Environment:
#   REPO     repository root (default: two levels above this script)
#   RUNDIR   where binaries, builds and results go
#            (default: /cosma/apps/do015/dc-ibra4/swiftgpupacksim_runs)
#
# Output in RUNDIR:
#   bin_gn002/, build_gn002_<buffer>/, work_gn002_*/
#   gn002_5steps_buffer-<buffer>/...          results
#   gn002_5steps_buffer-<buffer>/load.log     load before/after every run
#   gn002_driver.status                       current step, start time,
#                                             last finished run
#
# Stop it with Ctrl-c in the tmux session (or `tmux kill-session -t packsim-gn002`).

set -e

DRYRUN=0
[[ "${1:-}" == "--dry-run" ]] && DRYRUN=1

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
RUNDIR="${RUNDIR:-/cosma/apps/do015/dc-ibra4/swiftgpupacksim_runs}"
JOB="$HERE/job_buffer_layouts_gn003.slm"
NODE=gn002
BUFFERS="aos soa"
DIRS="EAGLE25_72threads Gresho256_72threads EAGLE25_36threads Gresho256_36threads"
STATUS_FILE="$RUNDIR/${NODE}_driver.status"

# steps: "<mode> <buffer> [<dir>]"
STEPS=("env aos")
for b in $BUFFERS; do STEPS+=("build $b"); done
for b in $BUFFERS; do for d in $DIRS; do STEPS+=("run $b $d"); done; done

host="$(hostname -s)"
arch="$(uname -m)"
host_ok=0
[[ "$host" == "$NODE" && "$arch" == "aarch64" ]] && host_ok=1

if [[ $DRYRUN -eq 1 ]]; then
  echo "dry run on $host ($arch): host check $([[ $host_ok -eq 1 ]] && echo passes || echo 'would REFUSE to start')"
  echo "repo:   $REPO @ $(git -C "$REPO" rev-parse --short HEAD)"
  echo "rundir: $RUNDIR"
  echo "status: $STATUS_FILE"
  i=0
  for st in "${STEPS[@]}"; do
    i=$((i + 1))
    read -r mode buf dir <<< "$st"
    case $mode in
    env) what="toolchain check" ;;
    build)
      n=$(ls "$RUNDIR/bin_${NODE}"/swiftgpupack_${NODE}_*_global-var_*_"$buf" 2> /dev/null | wc -l)
      what="build $buf ($n/50 executables present)" ;;
    run)
      n=$(ls "$RUNDIR/${NODE}_5steps_buffer-$buf/${dir}_global-var_"*_noflush/results_*.csv 2> /dev/null | wc -l)
      what="run $buf $dir ($n/50 results present)" ;;
    esac
    echo "  step $i/${#STEPS[@]}: $what"
    echo "      NODE=$NODE MODE=$mode BUFFER=$buf${dir:+ DIR=$dir} REPO=$REPO" \
      "LOAD_LOG=$RUNDIR/${NODE}_5steps_buffer-$buf/load.log STATUS_FILE=$STATUS_FILE" \
      "bash -l $JOB   (in $RUNDIR)"
  done
  exit 0
fi

if [[ $host_ok -ne 1 ]]; then
  echo "ERROR: this must run on $NODE (aarch64); this is $host ($arch)"
  exit 2
fi

started="$(date -Is)"
set_status() { # step text
  local last=""
  if [[ -f "$STATUS_FILE" ]]; then
    last="$(grep '^last_finished_run=' "$STATUS_FILE" || true)"
  fi
  {
    echo "driver_start=$started"
    echo "driver_pid=$$"
    echo "step=$1"
    echo "step_start=$(date -Is)"
    if [[ -n "$last" ]]; then echo "$last"; fi
  } > "$STATUS_FILE"
  return 0
}
trap 'set_status "STOPPED (exit $?) during: ${cur:-?}"' EXIT

echo "driver start $started on $host, repo $REPO @ $(git -C "$REPO" rev-parse --short HEAD)"
mkdir -p "$RUNDIR"
cd "$RUNDIR"

i=0
for st in "${STEPS[@]}"; do
  i=$((i + 1))
  read -r mode buf dir <<< "$st"
  cur="$i/${#STEPS[@]} $st"
  set_status "$cur"
  echo
  echo "##### $(date -Is) step $cur"
  mkdir -p "$RUNDIR/${NODE}_5steps_buffer-$buf"
  env NODE="$NODE" MODE="$mode" BUFFER="$buf" ${dir:+DIR="$dir"} REPO="$REPO" \
    LOAD_LOG="$RUNDIR/${NODE}_5steps_buffer-$buf/load.log" STATUS_FILE="$STATUS_FILE" \
    bash -l "$JOB"
done

cur="all done"
trap - EXIT
set_status "DONE ($(date -Is))"
echo "##### $(date -Is) all done"
