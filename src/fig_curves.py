"""Figure: event F1 versus collar for averaging and event fusion under both decoders, one panel per setting."""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AVG, FUS, SINGLE = "#eb6834", "#2a78d6", "#8c8c8c"
ARMS = [("cw:avg_prob", "averaging, median filter", AVG, "-", "s"),
        ("cw:wbf_median", "fusion, median filter", FUS, "-", "o"),
        ("cw:sebb:avg_prob", "averaging, cSEBB", AVG, "--", "s"),
        ("cw:wbf_median_sebb", "fusion, cSEBB", FUS, "--", "o"),
        ("cw:best_single_sebb", "best single, cSEBB", SINGLE, ":", None)]
TICKS = [20, 50, 100, 200, 500, 1000]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", nargs="+", required=True, help="label=path.json pairs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    plt.rcParams.update({"font.size": 8, "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman"],
                         "mathtext.fontset": "stix", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, len(a.settings), figsize=(7.0, 1.4), sharey=True)
    for ax, spec in zip(axes, a.settings):
        label, path = spec.split("=", 1)
        d = json.load(open(path))
        t, x = d["table"], [c * 1000 for c in d["collars"]]
        for arm, name, color, ls, marker in ARMS:
            ax.plot(x, t[arm]["f1"], color=color, ls=ls, lw=1.1 if marker else 0.9, marker=marker, ms=2.6,
                    label=name)
        ax.set_xscale("log")
        ax.set_xticks(TICKS)
        ax.set_xticklabels(["20", "50", "100", "200", "500", "1k"], fontsize=7)
        ax.minorticks_off()
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(label, fontsize=8)
        ax.grid(axis="y", lw=0.3, color="#e0e0e0")
        ax.set_xlabel("onset collar (ms)", fontsize=7.5)
    axes[0].set_ylabel("event F1", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(pad=0.3, w_pad=0.8, rect=(0, 0.1, 1, 1))
    fig.savefig(a.out, bbox_inches="tight")
    print(a.out)


if __name__ == "__main__":
    main()
