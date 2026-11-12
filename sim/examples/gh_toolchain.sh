# Toolchain selection for the Grace Hopper (aarch64) jobs, sourced by
# job_buffer_layouts_gn003.slm. Only defines functions and variables.
#
# The login environment on COSMA loads x86_64 modules (e.g.
# nvhpc-byo-compiler/24.5 puts an x86_64 nvcc on PATH), and Slurm exports that
# environment to the job. So a candidate nvcc / likwid-pin is accepted only if
#   - its path does not contain a foreign architecture (x86_64 on aarch64),
#   - `file` does not report a foreign ELF architecture, and
#   - it actually runs here.
#
# For testing on another machine, GH_ARCH overrides `uname -m` and
# GH_CUDA_SEARCH_ROOT (default /usr/local) the directory searched for cuda*.

gh_arch() { echo "${GH_ARCH:-$(uname -m)}"; }

# the architecture string `file` prints for ELF binaries of this machine
gh_file_arch() {
  case "$(gh_arch)" in
  aarch64) echo "aarch64" ;;
  x86_64) echo "x86-64" ;;
  *) gh_arch ;;
  esac
}

# 0 if $1 is (or resolves to) an ELF binary of another architecture, or its
# path names another architecture
gh_is_foreign() {
  local p="$1" real info
  real="$(readlink -f "$p" 2> /dev/null || echo "$p")"
  if [[ "$(gh_arch)" != "x86_64" && ("$p" == *x86_64* || "$real" == *x86_64*) ]]; then
    return 0
  fi
  if [[ "$(gh_arch)" != "aarch64" && ("$p" == *aarch64* || "$real" == *aarch64*) ]]; then
    return 0
  fi
  info="$(file -L "$p" 2> /dev/null)"
  if [[ "$info" == *ELF* && "$info" != *"$(gh_file_arch)"* ]]; then
    return 0
  fi
  return 1
}

# usable CUDA root: native nvcc that runs, runtime header and library present
gh_cuda_ok() {
  local r="$1"
  [[ -n "$r" && -x "$r/bin/nvcc" && -f "$r/include/cuda_runtime.h" ]] || return 1
  ls "$r"/lib64/libcudart.so* > /dev/null 2>&1 || return 1
  gh_is_foreign "$r/bin/nvcc" && return 1
  "$r/bin/nvcc" --version > /dev/null 2>&1 || return 1
  return 0
}

# CUDA root of the nvcc currently on PATH (nvhpc: <sdk>/compilers/bin/nvcc is a
# wrapper, the toolkit is $NVHPC_ROOT/cuda)
gh_cuda_root_of_path_nvcc() {
  local n r
  n="$(command -v nvcc)" || return 1
  r="$(dirname "$(dirname "$(readlink -f "$n")")")"
  if ! gh_cuda_ok "$r" && [[ -n "${NVHPC_ROOT:-}" ]] && gh_cuda_ok "$NVHPC_ROOT/cuda"; then
    r="$(readlink -f "$NVHPC_ROOT/cuda")"
  fi
  echo "$r"
}

# Sets GH_CUDA (resolved root), GH_CUDA_MODULE (if a module was needed) and
# CUDA_SOURCE. Order:
#   1. GH_CUDA (must be usable, otherwise this fails)
#   2. <root>/cuda-13.0
#   3. <root>/cuda (resolved)
#   4. newest usable <root>/cuda-*
#   5. cuda/<v>, nvhpc/<v> modules, newest first, whose nvcc is usable
# Returns 1 if nothing usable was found. Rejected candidates are printed.
gh_find_cuda() {
  local root="${GH_CUDA_SEARCH_ROOT:-/usr/local}" c m r
  CUDA_SOURCE=""
  if [[ -n "${GH_CUDA:-}" ]]; then
    if [[ -n "${GH_CUDA_MODULE:-}" ]]; then module load "$GH_CUDA_MODULE"; fi
    if gh_cuda_ok "$GH_CUDA"; then
      CUDA_SOURCE="GH_CUDA"
      return 0
    fi
    echo "rejected GH_CUDA=$GH_CUDA: not a usable CUDA for $(gh_arch)"
    GH_CUDA=""
    return 1
  fi
  GH_CUDA_MODULE=""
  for c in "$root/cuda-13.0" "$root/cuda" \
    $(ls -d "$root"/cuda-* 2> /dev/null | sort -V -r); do
    [[ -e "$c" ]] || continue
    r="$(readlink -f "$c")"
    if gh_cuda_ok "$r"; then
      GH_CUDA="$r"
      CUDA_SOURCE="$c"
      return 0
    fi
    echo "rejected CUDA candidate $c ($r)"
  done
  if type module > /dev/null 2>&1; then
    for m in $(module -t avail 2>&1 | grep -E '^(cuda|nvhpc)/[0-9]' | sed 's/(.*//' |
      awk -F/ '{print $2" "$0}' | sort -V -r | awk '{print $2}'); do
      if module load "$m" > /dev/null 2>&1 && command -v nvcc > /dev/null &&
        r="$(gh_cuda_root_of_path_nvcc)" && gh_cuda_ok "$r"; then
        GH_CUDA="$r"
        GH_CUDA_MODULE="$m"
        CUDA_SOURCE="module $m"
        return 0
      fi
      echo "rejected module $m"
      module unload "$m" > /dev/null 2>&1 || true
    done
  fi
  GH_CUDA=""
  return 1
}

# likwid-pin is a lua script; its interpreter likwid-lua (same directory) must
# be native, and `likwid-pin -v` must run.
gh_likwid_ok() {
  local p="$1" lua
  [[ -n "$p" && -x "$p" ]] || return 1
  gh_is_foreign "$p" && return 1
  lua="$(dirname "$(readlink -f "$p")")/likwid-lua"
  if [[ -e "$lua" ]] && gh_is_foreign "$lua"; then return 1; fi
  "$p" -v > /dev/null 2>&1 || return 1
  return 0
}

# Sets LIKWID_PIN. Order: LIKWID_PIN (must be usable), the ARM builds in
# /cosma/local/likwid (newest version first), likwid-pin on PATH.
gh_find_likwid_pin() {
  local c
  if [[ -n "${LIKWID_PIN:-}" ]]; then
    gh_likwid_ok "$LIKWID_PIN" && return 0
    echo "rejected LIKWID_PIN=$LIKWID_PIN: not usable on $(gh_arch)"
    LIKWID_PIN=""
    return 1
  fi
  for c in $(ls -d /cosma/local/likwid/*arm*/*/bin/likwid-pin 2> /dev/null |
    awk -F/ '{print $(NF-2)" "$0}' | sort -V -r | awk '{print $2}') \
    $(command -v likwid-pin 2> /dev/null); do
    if gh_likwid_ok "$c"; then
      LIKWID_PIN="$c"
      return 0
    fi
    echo "rejected likwid-pin candidate $c"
  done
  LIKWID_PIN=""
  return 1
}

# On aarch64, drop x86_64 directories that the exported login environment may
# still have in PATH / LD_LIBRARY_PATH after `module purge`.
gh_strip_foreign_paths() {
  [[ "$(gh_arch)" == "x86_64" ]] && return 0
  local v
  for v in PATH LD_LIBRARY_PATH; do
    if [[ -n "${!v:-}" ]]; then
      export "$v=$(echo "${!v}" | tr ':' '\n' | grep -v x86_64 | paste -sd: -)"
    fi
  done
}
