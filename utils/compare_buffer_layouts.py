#!/usr/bin/env python3

import os
import argparse
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

matplotlib.use("Agg")

from resultdata import ResultData
from plotting_utils import (
    get_result_fname,
    LOOP_SPLITS,
    LOOP_SPLIT_LABELS,
    mymplparams,
    mydpi,
    markers,
    linestyles,
)

matplotlib.rcParams.update(mymplparams)

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="""
Compare GPU transfer buffer layouts (AoS vs SoA buffers) for all particle
memory layouts and loop splitting variants, global-var particle access.

All times are relative to the SoA particle layout with SoA buffers and the
same loop split.

Expects the directory structure written by
sim/examples/job_buffer_layouts_*.slm:

  <srcdir>/<node>_5steps_buffer-<aos|soa>/
      <EXPERIMENT>_<NTHREADS>threads_global-var_<split>_noflush/results_<layout>.csv

For each experiment, writes
  - buffer_layouts_<node>_<experiment>_<nthreads>threads.{pdf,png}:
    rows: loop splits; left column: SoA buffers, right column: AoS buffers;
    x axis: pack/unpack tasks; one line per particle layout.
  - buffer_layouts_<node>_<experiment>_<nthreads>threads_summary.txt:
    total (pack+unpack) times, absolute and relative.
""",
)

parser.add_argument("srcdir", help="directory containing <node>_5steps_buffer-*")
parser.add_argument("node", help="node name, e.g. dine2 or gn003")
parser.add_argument(
    "-e",
    "--experiments",
    nargs="+",
    default=["EAGLE25_64threads", "Gresho256_64threads"],
    help="experiments to plot, as <EXPERIMENT>_<NTHREADS>threads",
)
parser.add_argument(
    "-o", "--outdir", default=".", help="where to write plots and summaries"
)
parser.add_argument("--png", action="store_true", help="write .png instead of .pdf")
parser.add_argument(
    "-l",
    "--label",
    default=None,
    help="node label used in titles and summary headers (default: node name), "
    'e.g. "gn002 (shared node)"',
)
parser.add_argument(
    "-s",
    "--suffix",
    default="",
    help="read <node>_5steps_buffer-<aos|soa><SUFFIX> (e.g. _v2) and append "
    "SUFFIX to the output file names",
)
parser.add_argument(
    "--no-tex", action="store_true", help="don't use LaTeX for text rendering"
)

LAYOUTS = [
    "soa",
    "aos",
    "upstream",
    "random-order",
    "pack-gradient",
    "pack-gradient-type",
    "pack-force",
    "pack-force-type",
    "pack-shared",
    "pack-shared-type",
]
BUFFERS = ["soa", "aos"]
TASK_LABELS = [
    "pack/density",
    "pack/gradient",
    "pack/force",
    "unpack/density",
    "unpack/gradient",
    "unpack/force",
]
PART_ACCESS = "global-var"
OTHER_VARIANT = "noflush"


SUFFIX = ""
LABEL = None


def read_results(srcdir, node, experiment, nthreads):
    """
    Returns dict results[buffer][split][layout] = timings array (ms), in the
    order of TASK_LABELS. Missing files are skipped with a warning.
    """
    results = {}
    for buffer in BUFFERS:
        results[buffer] = {}
        bufdir = os.path.join(srcdir, f"{node}_5steps_buffer-{buffer}{SUFFIX}")
        for split in LOOP_SPLITS:
            results[buffer][split] = {}
            for layout in LAYOUTS:
                try:
                    fname = get_result_fname(
                        bufdir,
                        experiment,
                        nthreads,
                        PART_ACCESS,
                        split,
                        OTHER_VARIANT,
                        layout,
                    )
                except FileNotFoundError:
                    continue
                if not os.path.exists(fname):
                    print(f"WARNING: missing {fname}")
                    continue
                res = ResultData(fname)
                assert list(res.tasks) == TASK_LABELS, res.tasks
                results[buffer][split][layout] = res.timings
    return results


def reference(results, split):
    """t(SoA particles, SoA buffers) for the given loop split."""
    return results["soa"][split].get("soa")


def plot_experiment(results, node, experiment, nthreads, outdir, png):

    nrows = len(LOOP_SPLITS)
    ncols = len(BUFFERS)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 3.2 * nrows),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    x = np.arange(len(TASK_LABELS))

    for r, split in enumerate(LOOP_SPLITS):
        ref = reference(results, split)
        for c, buffer in enumerate(BUFFERS):
            ax = axes[r, c]
            if ref is None:
                ax.text(0.5, 0.5, "no reference", transform=ax.transAxes)
                continue
            for l, layout in enumerate(LAYOUTS):
                t = results[buffer][split].get(layout)
                if t is None:
                    continue
                ax.plot(
                    x,
                    t / ref,
                    label=layout,
                    marker=markers[l % len(markers)],
                    linestyle=linestyles[l // len(markers) % len(linestyles)],
                    lw=1.5,
                    alpha=0.8,
                    markersize=5,
                )
            ax.axhline(1.0, color="k", lw=0.8, ls="--")
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_title(f"{buffer.upper()} buffers")
            if c == 0:
                ax.set_ylabel(
                    LOOP_SPLIT_LABELS[r] + "\n" + r"$t / t_{\mathrm{SoA,SoA}}$"
                )

    for ax in axes[-1, :]:
        ax.set_xticks(x)
        ax.set_xticklabels(TASK_LABELS, rotation=30, ha="right")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.0),
        fontsize=11,
    )
    fig.suptitle(
        f"{LABEL or node}, {experiment}, {nthreads} threads, global-var access", y=1.03
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    ext = "png" if png else "pdf"
    fname = os.path.join(
        outdir, f"buffer_layouts_{node}_{experiment}_{nthreads}threads{SUFFIX}.{ext}"
    )
    fig.savefig(fname, dpi=mydpi, bbox_inches="tight")
    plt.close(fig)
    print("written", fname)


def write_summary(results, node, experiment, nthreads, outdir):
    """
    Total pack+unpack time per layout, split and buffer: absolute [ms] and
    relative to SoA particles + SoA buffers (same split). Also per-task
    relative times.
    """
    lines = [
        f"# {LABEL or node} {experiment} {nthreads} threads, global-var access",
        "# total = sum of pack+unpack over density, gradient, force",
        "# rel = total / total(soa particles, soa buffers, same split)",
        "# buf_ratio = total(aos buffers) / total(soa buffers), same layout+split",
        "",
    ]
    header = f"{'split':20s} {'layout':20s} {'soa_buf[ms]':>12s} {'rel':>7s} {'aos_buf[ms]':>12s} {'rel':>7s} {'buf_ratio':>9s}"
    lines.append(header)
    for split in LOOP_SPLITS:
        ref = reference(results, split)
        reftot = ref.sum() if ref is not None else np.nan
        for layout in LAYOUTS:
            ts = results["soa"][split].get(layout)
            ta = results["aos"][split].get(layout)
            tot_s = ts.sum() if ts is not None else np.nan
            tot_a = ta.sum() if ta is not None else np.nan
            lines.append(
                f"{split:20s} {layout:20s} {tot_s:12.1f} {tot_s / reftot:7.3f} "
                f"{tot_a:12.1f} {tot_a / reftot:7.3f} {tot_a / tot_s:9.3f}"
            )
        lines.append("")

    lines.append("# per task, relative to soa particles + soa buffers (same split)")
    lines.append(
        f"{'split':20s} {'layout':20s} {'buffer':6s} "
        + " ".join(f"{t:>15s}" for t in TASK_LABELS)
    )
    for split in LOOP_SPLITS:
        ref = reference(results, split)
        if ref is None:
            continue
        for layout in LAYOUTS:
            for buffer in BUFFERS:
                t = results[buffer][split].get(layout)
                if t is None:
                    continue
                lines.append(
                    f"{split:20s} {layout:20s} {buffer:6s} "
                    + " ".join(f"{v:15.3f}" for v in t / ref)
                )
        lines.append("")

    fname = os.path.join(
        outdir, f"buffer_layouts_{node}_{experiment}_{nthreads}threads{SUFFIX}_summary.txt"
    )
    with open(fname, "w") as fp:
        fp.write("\n".join(lines) + "\n")
    print("written", fname)


if __name__ == "__main__":

    args = parser.parse_args()
    SUFFIX = args.suffix
    LABEL = args.label
    if args.no_tex:
        matplotlib.rcParams["text.usetex"] = False
    os.makedirs(args.outdir, exist_ok=True)

    for exp in args.experiments:
        experiment, nthreads_str = exp.rsplit("_", 1)
        if not nthreads_str.endswith("threads"):
            raise ValueError(f"expected <EXPERIMENT>_<N>threads, got {exp}")
        nthreads = int(nthreads_str[: -len("threads")])

        results = read_results(args.srcdir, args.node, experiment, nthreads)
        plot_experiment(results, args.node, experiment, nthreads, args.outdir, args.png)
        write_summary(results, args.node, experiment, nthreads, args.outdir)
