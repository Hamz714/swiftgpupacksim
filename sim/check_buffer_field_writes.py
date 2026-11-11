#!/usr/bin/env python3
"""
Static check of the global-var pack/unpack functions.

For every particle layout and loop split, the transfer buffer fields that each
pack function writes (gpu_send_*_set_<field>) and each unpack function reads
(gpu_recv_*_get_<field>) must be the same as for the `none` split, and every
field must be accessed exactly once. The particle quantities read by the pack
functions (part_get_*) and written by the unpack functions (part_set_*) are
checked the same way.

Each pack function file is run through the C preprocessor with the layout
macro defined (and #include lines removed), so only the code that is
compiled for that layout is counted.

Usage: ./check_buffer_field_writes.py [-v]
Exit code 0 if no differences were found.
"""

import collections
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PACKDIR = os.environ.get(
    "PACKDIR", os.path.join(HERE, "src", "gpu_part_pack_functions")
)
FILE = "gpu_part_pack_functions_global_var_access_loop_split_{}.h"

SPLITS = ["none", "by-struct", "by-element", "by-type", "by-struct-and-type"]
LAYOUTS = {
    "aos": "SPHENIX_AOS_PARTICLE",
    "soa": "SPHENIX_SOA_PARTICLE",
    "upstream": "SPHENIX_UPSTREAM_PARTICLE",
    "pack-gradient": "SPHENIX_PACK_GRADIENT_PARTICLE",
    "pack-gradient-type": "SPHENIX_PACK_GRADIENT_TYPE_PARTICLE",
    "pack-force": "SPHENIX_PACK_FORCE_PARTICLE",
    "pack-force-type": "SPHENIX_PACK_FORCE_TYPE_PARTICLE",
    "pack-shared": "SPHENIX_PACK_SHARED_PARTICLE",
    "pack-shared-type": "SPHENIX_PACK_SHARED_TYPE_PARTICLE",
    "random-order": "SPHENIX_RANDOM_ORDER_PARTICLE",
}
FUNCS = [
    "gpu_pack_part_density",
    "gpu_pack_part_gradient",
    "gpu_pack_part_force",
    "gpu_unpack_part_density",
    "gpu_unpack_part_gradient",
    "gpu_unpack_part_force",
]

FUNC_RE = re.compile(r"\b(" + "|".join(FUNCS) + r")\s*\(")
BUF_SET_RE = re.compile(r"\bgpu_send_[dgf]_set_(\w+)\s*\(")
BUF_GET_RE = re.compile(r"\bgpu_recv_[dgf]_get_(\w+)\s*\(")
PART_GET_RE = re.compile(r"\bpart_get_(?:const_)?(\w+?)_global\s*\(")
PART_SET_RE = re.compile(r"\bpart_set_(\w+?)_global\s*\(")


def cpp(src, defines):
    out = subprocess.run(
        ["cpp", "-P", "-undef", "-nostdinc"] + ["-D" + d for d in defines] + ["-"],
        input=src,
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        raise RuntimeError("cpp {} failed:\n{}".format(defines, out.stderr))
    return out.stdout


def effective_split(split, macro):
    """
    gpu_part_pack_functions.h maps the configured loop split to the pack
    function file actually used (e.g. by-struct with aos particles uses the
    `none` file). Evaluate its helper macros for this layout and split.
    """
    with open(os.path.join(HERE, "src", "gpu_part_pack_functions.h")) as f:
        src = f.read()
    src = src[: src.index("#if defined(SWIFT_PARTICLE_ACCESS_PART_STRUCT)")]
    src = "".join(l + "\n" for l in src.splitlines() if "#include" not in l)
    names = [s.upper().replace("-", "_") for s in SPLITS]
    for n in names:
        src += "#ifdef USE_LOOP_SPLIT_{0}\nUSE {0}\n#endif\n".format(n)
    swift_split = "SWIFT_LOOP_SPLIT_" + split.upper().replace("-", "_")
    used = [l.split()[1] for l in cpp(src, [macro, swift_split]).splitlines()
            if l.startswith("USE ")]
    if len(used) != 1:
        raise RuntimeError("{} {}: selected splits {}".format(macro, split, used))
    return used[0].lower().replace("_", "-")


def preprocess(split, macro):
    path = os.path.join(PACKDIR, FILE.format(split.replace("-", "_")))
    with open(path) as f:
        src = "".join(l for l in f if not l.lstrip().startswith("#include"))
    return cpp(src, [macro, "USE_LOOP_SPLIT_" + split.upper().replace("-", "_")])


def functions(code):
    """split preprocessed code into {function name: body text}"""
    starts = [(m.start(), m.group(1)) for m in FUNC_RE.finditer(code)]
    # keep only definitions (followed by a '{' before the next ';')
    defs = []
    for pos, name in starts:
        rest = code[pos:]
        if rest.find("{") != -1 and rest.find("{") < rest.find(";"):
            defs.append((pos, name))
    bodies = {}
    for k, (pos, name) in enumerate(defs):
        end = defs[k + 1][0] if k + 1 < len(defs) else len(code)
        if name in bodies:
            raise RuntimeError("function {} defined twice".format(name))
        bodies[name] = code[pos:end]
    return bodies


def accesses(name, body):
    """multisets of (buffer fields, particle quantities) accessed"""
    if name.startswith("gpu_pack"):
        buf = BUF_SET_RE.findall(body)
        part = PART_GET_RE.findall(body)
    else:
        buf = BUF_GET_RE.findall(body)
        part = PART_SET_RE.findall(body)
    return collections.Counter(buf), collections.Counter(part)


def buffer_field_lists():
    """{buffer: set of fields} from the GPU_<BUF>_FIELDS(X) lists"""
    path = os.path.join(HERE, "src", "swift_placeholders", "cuda",
                        "gpu_part_buffer_access.h")
    with open(path) as f:
        src = f.read()
    fields = collections.defaultdict(set)
    for buf, field in re.findall(r"X\((send_[dgf]|recv_[dgf]), (\w+), \w+\)", src):
        fields[buf].add(field)
    return fields


BUFFER_OF = {
    "gpu_pack_part_density": "send_d",
    "gpu_pack_part_gradient": "send_g",
    "gpu_pack_part_force": "send_f",
    "gpu_unpack_part_density": "recv_d",
    "gpu_unpack_part_gradient": "recv_g",
    "gpu_unpack_part_force": "recv_f",
}


def fmt(counter):
    return ", ".join(
        "{}x{}".format(k, v) if v != 1 else k for k, v in sorted(counter.items())
    )


def main():
    verbose = "-v" in sys.argv
    all_fields = buffer_field_lists()
    assert len(all_fields) == 6, all_fields.keys()
    ndiff = 0
    nchecked = 0
    for layout, macro in LAYOUTS.items():
        ref = None
        for split in SPLITS:
            used = effective_split(split, macro)
            if verbose:
                print("{} {}: uses the {} pack functions".format(layout, split, used))
            bodies = functions(preprocess(used, macro))
            missing = [f for f in FUNCS if f not in bodies]
            if missing:
                print("{} {}: missing functions {}".format(layout, split, missing))
                ndiff += 1
                continue
            acc = {f: accesses(f, bodies[f]) for f in FUNCS}
            if split == "none":
                ref = acc
                # the reference itself must access every buffer field
                for f in FUNCS:
                    expected = all_fields[BUFFER_OF[f]]
                    if set(acc[f][0]) != expected:
                        ndiff += 1
                        print("DIFF {} none {}: buffer fields {} vs field list {}".format(
                            layout, f, sorted(acc[f][0]), sorted(expected)))
            for f in FUNCS:
                nchecked += 1
                for what, got, exp in (
                    ("buffer fields", acc[f][0], ref[f][0]),
                    ("particle quantities", acc[f][1], ref[f][1]),
                ):
                    dup = {k: v for k, v in got.items() if v != 1}
                    if got != exp or dup:
                        ndiff += 1
                        print("DIFF {} {} {} {}:".format(layout, split, f, what))
                        if got - exp:
                            print("   extra:   ", fmt(got - exp))
                        if exp - got:
                            print("   missing: ", fmt(exp - got))
                        if dup:
                            print("   not exactly once:", fmt(collections.Counter(dup)))
                if verbose and split == "none":
                    print("{} {}: buffer [{}]".format(layout, f, fmt(acc[f][0])))
    print(
        "checked {} layout x split x function combinations, {} differences".format(
            nchecked, ndiff
        )
    )
    return 1 if ndiff else 0


if __name__ == "__main__":
    sys.exit(main())
