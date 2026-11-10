#!/usr/bin/env python3
"""
Generate gpu_part_buffer_access.h: inline getters/setters for the host-side GPU
transfer buffers, with an AoS implementation (existing float4/float3/int2/int4
structs in gpu_part_structs.h) and an SoA implementation (one array per scalar
quantity), selected by GPU_BUFFER_AOS / GPU_BUFFER_SOA.

Usage:
    python3 generate_gpu_part_buffer_access.py

writes gpu_part_buffer_access.h next to this script and formats it with
clang-format (version 18, like format.sh; override with CLANG_FORMAT_CMD) if
that is available.

The tables below are the single source of truth for which scalar quantity lives
where in the AoS structs. If gpu_part_structs.h changes, update them and
regenerate.
"""

import os
import shutil
import subprocess

# Layout conditions under which the AoS structs group quantities differently.
# Must match the #if conditions in gpu_part_structs.h.
def _any_defined(macros):
    """Return a preprocessor condition 'defined(A) || defined(B) ...'."""
    return " || \\\n    ".join(f"defined({m})" for m in macros)


COND_SEND_G_ALT = _any_defined(
    [
        "SPHENIX_PACK_FORCE_PARTICLE",
        "SPHENIX_PACK_SHARED_PARTICLE",
        "SPHENIX_PACK_SHARED_TYPE_PARTICLE",
        "SPHENIX_PACK_FORCE_TYPE_PARTICLE",
    ]
)
COND_SEND_F_ALT = _any_defined(
    [
        "SPHENIX_PACK_GRADIENT_PARTICLE",
        "SPHENIX_PACK_GRADIENT_TYPE_PARTICLE",
        "SPHENIX_PACK_SHARED_PARTICLE",
        "SPHENIX_PACK_SHARED_TYPE_PARTICLE",
    ]
)

# Each buffer: (short name, AoS struct, SoA struct, alt-grouping condition,
# fields). Each field: (name, C type, AoS member, AoS member under the
# alternative grouping or None if it does not depend on the layout).
# The field order is the canonical order, also used by the buffer dumps.
BUFFERS = [
    (
        "send_d",
        "gpu_part_send_d",
        "gpu_part_send_d_soa",
        None,
        [
            ("x", "float", "x_h.x", None),
            ("y", "float", "x_h.y", None),
            ("z", "float", "x_h.z", None),
            ("h", "float", "x_h.w", None),
            ("vx", "float", "vx_m.x", None),
            ("vy", "float", "vx_m.y", None),
            ("vz", "float", "vx_m.z", None),
            ("m", "float", "vx_m.w", None),
            ("pjs", "int", "pjs_pje.x", None),
            ("pje", "int", "pjs_pje.y", None),
        ],
    ),
    (
        "recv_d",
        "gpu_part_recv_d",
        "gpu_part_recv_d_soa",
        None,
        [
            ("rho", "float", "rho_rhodh_wcount_wcount_dh.x", None),
            ("rho_dh", "float", "rho_rhodh_wcount_wcount_dh.y", None),
            ("wcount", "float", "rho_rhodh_wcount_wcount_dh.z", None),
            ("wcount_dh", "float", "rho_rhodh_wcount_wcount_dh.w", None),
            ("rot_vx", "float", "rot_vx_div_v.x", None),
            ("rot_vy", "float", "rot_vx_div_v.y", None),
            ("rot_vz", "float", "rot_vx_div_v.z", None),
            ("div_v", "float", "rot_vx_div_v.w", None),
        ],
    ),
    (
        "send_g",
        "gpu_part_send_g",
        "gpu_part_send_g_soa",
        COND_SEND_G_ALT,
        [
            ("x", "float", "x_h.x", None),
            ("y", "float", "x_h.y", None),
            ("z", "float", "x_h.z", None),
            ("h", "float", "x_h.w", None),
            ("vx", "float", "vx_m.x", None),
            ("vy", "float", "vx_m.y", None),
            ("vz", "float", "vx_m.z", None),
            ("m", "float", "vx_m.w", None),
            ("u", "float", "u_rho_c_aviscmax.x", "u_rho_c_avisc.x"),
            ("rho", "float", "u_rho_c_aviscmax.y", "u_rho_c_avisc.y"),
            ("c", "float", "u_rho_c_aviscmax.z", "u_rho_c_avisc.z"),
            ("alpha_visc", "float", "avisc_vsig_lapu.x", "u_rho_c_avisc.w"),
            (
                "alpha_visc_max_ngb",
                "float",
                "u_rho_c_aviscmax.w",
                "aviscmax_vsig_lapu.x",
            ),
            ("v_sig", "float", "avisc_vsig_lapu.y", "aviscmax_vsig_lapu.y"),
            ("laplace_u", "float", "avisc_vsig_lapu.z", "aviscmax_vsig_lapu.z"),
            ("pjs", "int", "pjs_pje.x", None),
            ("pje", "int", "pjs_pje.y", None),
        ],
    ),
    (
        "recv_g",
        "gpu_part_recv_g",
        "gpu_part_recv_g_soa",
        None,
        [
            ("alpha_visc_max_ngb", "float", "aviscmax_vsig_lapu.x", None),
            ("v_sig", "float", "aviscmax_vsig_lapu.y", None),
            ("laplace_u", "float", "aviscmax_vsig_lapu.z", None),
        ],
    ),
    (
        "send_f",
        "gpu_part_send_f",
        "gpu_part_send_f_soa",
        COND_SEND_F_ALT,
        [
            ("x", "float", "x_h.x", None),
            ("y", "float", "x_h.y", None),
            ("z", "float", "x_h.z", None),
            ("h", "float", "x_h.w", None),
            ("vx", "float", "vx_m.x", None),
            ("vy", "float", "vx_m.y", None),
            ("vz", "float", "vx_m.z", None),
            ("m", "float", "vx_m.w", None),
            ("u", "float", "u_rho_f_p.x", "u_rho_c_avisc.x"),
            ("rho", "float", "u_rho_f_p.y", "u_rho_c_avisc.y"),
            ("f_gradh", "float", "u_rho_f_p.z", "f_p_balsara_adiff.x"),
            ("pressure", "float", "u_rho_f_p.w", "f_p_balsara_adiff.y"),
            ("balsara", "float", "bals_c_avisc_adiff.x", "f_p_balsara_adiff.z"),
            ("c", "float", "bals_c_avisc_adiff.y", "u_rho_c_avisc.z"),
            ("alpha_visc", "float", "bals_c_avisc_adiff.z", "u_rho_c_avisc.w"),
            ("alpha_diff", "float", "bals_c_avisc_adiff.w", "f_p_balsara_adiff.w"),
            ("time_bin", "int", "timebin_minngbtimebin_pjs_pje.x", None),
            ("min_ngb_time_bin", "int", "timebin_minngbtimebin_pjs_pje.y", None),
            ("pjs", "int", "timebin_minngbtimebin_pjs_pje.z", None),
            ("pje", "int", "timebin_minngbtimebin_pjs_pje.w", None),
        ],
    ),
    (
        "recv_f",
        "gpu_part_recv_f",
        "gpu_part_recv_f_soa",
        None,
        [
            ("a_x", "float", "a_hydro.x", None),
            ("a_y", "float", "a_hydro.y", None),
            ("a_z", "float", "a_hydro.z", None),
            ("u_dt", "float", "udt_hdt_minngbtb.x", None),
            ("h_dt", "float", "udt_hdt_minngbtb.y", None),
            ("min_ngb_time_bin", "float", "udt_hdt_minngbtb.z", None),
        ],
    ),
]

INL = "__attribute__((always_inline)) INLINE static"

HEADER = """\
/*******************************************************************************
 * This file is part of SWIFT.
 * Copyright (c) 2025 Abouzied M. A. Nasar (abouzied.nasar@manchester.ac.uk)
 *                    Mladen Ivkovic (mladen.ivkovic@durham.ac.uk)
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/

/* -------------------------------------------------------------------------
 * GENERATED FILE -- DO NOT EDIT.
 * Generated by generate_gpu_part_buffer_access.py in this directory.
 * ------------------------------------------------------------------------- */

#ifndef CUDA_GPU_PART_BUFFER_ACCESS_H
#define CUDA_GPU_PART_BUFFER_ACCESS_H

/**
 * @file cuda/gpu_part_buffer_access.h
 * @brief Getters and setters for the host-side GPU transfer buffers.
 *
 * Each buffer (send/recv for density, gradient, force) is handled through a
 * handle type gpu_<buf>_buf:
 *  - AoS (GPU_BUFFER_AOS, default): a pointer to the gpu_part_<buf> structs.
 *  - SoA (GPU_BUFFER_SOA): a gpu_part_<buf>_soa struct of scalar arrays,
 *    passed by value.
 *
 * gpu_<buf>_offset(b, ind) returns a handle whose element 0 is element ind of
 * b. gpu_<buf>_get_<field>(b, i) and gpu_<buf>_set_<field>(b, i, val) access
 * scalar quantity <field> of element i.
 */

#include "../../config.h"
#include "../inline.h"
#include "gpu_part_structs.h"

#include <stddef.h>

#if !defined(GPU_BUFFER_AOS) && !defined(GPU_BUFFER_SOA)
#error "Unknown GPU buffer layout"
#endif

#if defined(GPU_BUFFER_SOA) && !defined(SWIFT_PARTICLE_ACCESS_GLOBAL_VAR)
#error "SoA GPU buffers are only implemented for global-var particle access"
#endif
"""


def handle_typedefs():
    out = ["", "#if defined(GPU_BUFFER_SOA)", ""]
    for short, aos, soa, _, _ in BUFFERS:
        out.append(f"typedef struct {soa} gpu_{short}_buf;")
    out += ["", "#else", ""]
    for short, aos, soa, _, _ in BUFFERS:
        out.append(f"typedef struct {aos}* restrict gpu_{short}_buf;")
    out += ["", "#endif /* GPU_BUFFER_SOA */", ""]
    return out


def soa_section(short, fields):
    t = f"gpu_{short}_buf"
    out = [
        "/**",
        f" * @brief Handle to element ind of a {short} buffer",
        " */",
        f"{INL} {t}",
        f"gpu_{short}_offset({t} b, const ptrdiff_t ind) {{",
    ]
    for name, _, _, _ in fields:
        out.append(f"  b.{name} += ind;")
    out += ["  return b;", "}", ""]
    for name, ctype, _, _ in fields:
        out += [
            f"{INL} {ctype}",
            f"gpu_{short}_get_{name}(const {t} b, const ptrdiff_t i) {{",
            f"  return b.{name}[i];",
            "}",
            "",
            f"{INL} void gpu_{short}_set_{name}(",
            f"    const {t} b, const ptrdiff_t i, const {ctype} val) {{",
            f"  b.{name}[i] = val;",
            "}",
            "",
        ]
    return out


def aos_accessors(short, aos, ctype, name, member):
    # Plain (non-restrict) pointers here on purpose: after inlining, a restrict
    # parameter would only create a noalias scope for this tiny function. The
    # aliasing guarantee comes from the restrict buffer parameter of the
    # calling pack/unpack function, as before the accessors were introduced.
    return [
        f"{INL} {ctype}",
        f"gpu_{short}_get_{name}(const struct {aos}* b, const ptrdiff_t i) {{",
        f"  return b[i].{member};",
        "}",
        "",
        f"{INL} void gpu_{short}_set_{name}(",
        f"    struct {aos}* b, const ptrdiff_t i, const {ctype} val) {{",
        f"  b[i].{member} = val;",
        "}",
        "",
    ]


def aos_section(short, aos, cond, fields):
    t = f"gpu_{short}_buf"
    out = [
        "/**",
        f" * @brief Handle to element ind of a {short} buffer",
        " */",
        f"{INL} struct {aos}*",
        f"gpu_{short}_offset(struct {aos}* b, const ptrdiff_t ind) {{",
        "  return &b[ind];",
        "}",
        "",
    ]
    fixed = [f for f in fields if f[3] is None]
    varying = [f for f in fields if f[3] is not None]
    for name, ctype, member, _ in fixed:
        out += aos_accessors(short, aos, ctype, name, member)
    if varying:
        out += [
            "/* Grouping of these quantities depends on the particle layout, "
            "see",
            " * gpu_part_structs.h */",
            f"#if {cond}",
            "",
        ]
        for name, ctype, _, alt in varying:
            out += aos_accessors(short, aos, ctype, name, alt)
        out += ["#else", ""]
        for name, ctype, member, _ in varying:
            out += aos_accessors(short, aos, ctype, name, member)
        out += ["#endif", ""]
    return out


def main():
    out = [HEADER]
    out += handle_typedefs()

    out += ["#if defined(GPU_BUFFER_SOA)", ""]
    for short, _, _, _, fields in BUFFERS:
        out += [f"/* --- {short}: SoA --- */", ""]
        out += soa_section(short, fields)
    out += ["#else /* GPU_BUFFER_AOS */", ""]
    for short, aos, _, cond, fields in BUFFERS:
        out += [f"/* --- {short}: AoS --- */", ""]
        out += aos_section(short, aos, cond, fields)
    out += ["#endif /* GPU_BUFFER_SOA */", ""]

    # Field lists in canonical order, used e.g. by the buffer dumps. Each entry
    # of GPU_<BUF>_FIELDS(X) expands to X(<buf>, <field>, <ctype>).
    out += [
        "/* Field lists in canonical order. X(buf, field, ctype) is called for",
        " * each field, e.g. to dump all fields of a buffer. */",
    ]
    for short, _, _, _, fields in BUFFERS:
        out.append(f"#define GPU_{short.upper()}_FIELDS(X) \\")
        for k, (name, ctype, _, _) in enumerate(fields):
            end = " \\" if k < len(fields) - 1 else ""
            out.append(f"  X({short}, {name}, {ctype}){end}")
        out.append("")

    out += ["#endif /* CUDA_GPU_PART_BUFFER_ACCESS_H */"]

    here = os.path.dirname(os.path.abspath(__file__))
    outfile = os.path.join(here, "gpu_part_buffer_access.h")
    with open(outfile, "w") as fp:
        fp.write("\n".join(out) + "\n")
    print("written", outfile)

    clang = os.environ.get("CLANG_FORMAT_CMD")
    if clang is None:
        clang = shutil.which("clang-format-18") or shutil.which("clang-format")
    if clang is None:
        print("WARNING: clang-format not found, output not formatted")
        return
    subprocess.run([clang, "-style=file", "-i", outfile], check=True, cwd=here)
    print("formatted with", clang)


if __name__ == "__main__":
    main()
