"""Write the paper's LaTeX tables from the result files, so no number is copied by hand."""
import json
import os

import sedlib as L

R = L.RESULTS
OUT = os.path.join(L.REPO, "tables")
COLLARS = [20, 50, 100, 200, 300, 500, 1000]
K200 = COLLARS.index(200)
MARK, NEG = r"$^\dagger$", r"$^\ast$"
# every run below includes per-class post-processing arms ("cw:")
SETTINGS = [("DESED, zero-shot", "cw_desed_raw"), ("\\quad calibrated", "cw_desed_cal"),
            ("URBAN-SED, zero-shot", "ut_urban_zs_raw"), ("\\quad calibrated", "ut_urban_zs_cal"),
            ("URBAN-SED, adapted", "ut_urban_ad_raw"), ("\\quad calibrated", "ut_urban_ad_cal"),
            ("TUT-SED, adapted", "tut_ad_raw"), ("\\quad calibrated", "tut_ad_cal")]
EXTRA = [("URBAN-SED, BiGRU heads, seed 0", "ut_urban_gru0_raw"), ("\\quad seed 1", "ut_urban_gru1_raw")]


def load(name):
    path = os.path.join(R, name + ".json")
    return json.load(open(path))["table"] if os.path.exists(path) else None


def fmt(v, signed=False):
    return (f"{v:+.3f}" if signed else f"{v:.3f}").replace("0.", ".", 1)


def sig(t, arm, ref, k):
    lo, hi = t[arm][f"d_{ref}"][0][k], t[arm][f"d_{ref}"][1][k]
    return 1 if lo > 0 else (-1 if hi < 0 else 0)


def collar_mean_ci(name, arm, ref, n=2000, seed=0, level=95.0):
    """Fusion minus averaging averaged over the seven collars, with a clip bootstrap shared across collars."""
    import numpy as np
    z = np.load(os.path.join(R, name + "_counts.npz"))
    a_, r_ = z[arm.replace(":", "__")], z[ref.replace(":", "__")]           # (K, N, C, 3)

    def macro(c):                                                          # (..., N, C, 3) -> (...,)
        s = c.sum(-3)
        f = np.where(s[..., 1] + s[..., 2] > 0, 2 * s[..., 0] / np.maximum(s[..., 1] + s[..., 2], 1), 0.0)
        return f.mean(-1)
    point = float((macro(a_) - macro(r_)).mean())
    rng = np.random.default_rng(seed)
    N = a_.shape[1]
    boot = []
    for _ in range(n):
        i = rng.integers(0, N, N)
        boot.append(float((macro(a_[:, i]) - macro(r_[:, i])).mean()))
    lo, hi = np.percentile(boot, [(100 - level) / 2, 100 - (100 - level) / 2])
    return point, lo, hi


BONF = 100 - 5.0 / 20   # family of the 20 primary collar-mean intervals (10 settings x 2 decoders)


def bonf_mark(name, arm, ref):
    _, lo, hi = collar_mean_ci(name, arm, ref, n=8000, level=BONF)
    return r"$^\ddagger$" if lo > 0 or hi < 0 else ""


PP = {"cw_desed_raw": "pp_desed_raw", "cw_desed_cal": "pp_desed_cal", "ut_urban_zs_raw": "pp_urban_zs_raw",
      "ut_urban_zs_cal": "pp_urban_zs_cal", "ut_urban_ad_raw": "pp_urban_ad_raw", "ut_urban_ad_cal": "pp_urban_ad_cal",
      "tut_ad_raw": "pp_tut_ad_raw", "tut_ad_cal": "pp_tut_ad_cal", "ut_urban_gru0_raw": "pp_urban_gru0_raw",
      "ut_urban_gru1_raw": "pp_urban_gru1_raw"}


def table_main():
    """Table 1: F1 at 200 ms under both decoders, with averaging also given gap filling and minimum duration, and
    collar-averaged fusion minus the strongest averaging arm of each decoder."""
    lines = [r"\begin{table*}[t]", r"\centering", r"\setlength{\tabcolsep}{3.0pt}", r"\footnotesize",
             r"\caption{Event F1 at 200\,ms, and fusion minus the strongest averaging arm (avg.+pp: with tuned gap filling "
             r"and minimum duration) averaged over the seven collars, with 95\% intervals. Per-class post-processing; DESED "
             r"cross-fitted, other rows scored once on the untouched test split. $\dagger$/$\ast$: fusion above/below "
             r"the strongest averaging arm at 200\,ms. $\ddagger$: collar-mean interval excludes zero after Bonferroni "
             r"correction over the 20 intervals.}",
             r"\label{tab:main}", r"\begin{tabular}{lccccccccc}", r"\toprule",
             r" & \multicolumn{4}{c}{median filter, F1 at 200\,ms} & \multicolumn{3}{c}{cSEBB, F1 at 200\,ms}"
             r" & \multicolumn{2}{c}{fusion $-$ averaging, collar mean} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}",
             r"setting & single & avg. & avg.+pp & fusion & single & avg. & fusion & median filter & cSEBB \\",
             r"\midrule"]
    for label, f in SETTINGS + EXTRA:
        t, tp = load(f), load(PP.get(f, ""))
        if t is None or tp is None:
            continue
        c = lambda tt, arm, ref=None: fmt(tt[arm]["f1"][K200]) + (
            (MARK if sig(tt, arm, ref, K200) > 0 else (NEG if sig(tt, arm, ref, K200) < 0 else "")) if ref else "")
        cells = [c(t, "cw:best_single"), c(t, "cw:avg_prob"), c(tp, "cw:avg_prob_pp"),
                 c(tp, "cw:wbf_median", "cw:avg_prob_pp"),
                 c(t, "cw:best_single_sebb"), c(t, "cw:sebb:avg_prob"), c(t, "cw:wbf_median_sebb", "cw:sebb:avg_prob")]
        for ff, arm, ref in [(PP[f], "cw:wbf_median", "cw:avg_prob_pp"), (f, "cw:wbf_median_sebb", "cw:sebb:avg_prob")]:
            p, lo, hi = collar_mean_ci(ff, arm, ref)
            cells.append(f"{fmt(p, True)} [{fmt(lo, True)}, {fmt(hi, True)}]" + bonf_mark(ff, arm, ref))
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\vspace{-3mm}", r"\end{table*}"]
    return "\n".join(lines)


def table_controls():
    """Table: controls for per-member decoding freedom, global post-processing, event F1 at 200 ms."""
    rows = [("DESED, zero-shot", "pp_desed_raw"), ("\\quad calibrated", "pp_desed_cal"),
            ("URBAN, zero-shot", "pp_urban_zs_raw"), ("URBAN, adapted", "pp_urban_ad_raw"),
            ("URBAN, BiGRU s0", "pp_urban_gru0_raw"), ("TUT, adapted", "pp_tut_ad_raw")]
    arms = [("avg_prob", None), ("avg_logit", None), ("avg_prob_pp", None), ("vote", "avg_prob"),
            ("wbf_shared", "avg_prob_pp"), ("wbf_median", "avg_prob_pp"), ("wbf_weighted", None)]
    lines = [r"\begin{table}[t]", r"\centering", r"\setlength{\tabcolsep}{2.4pt}", r"\footnotesize",
             r"\caption{Controls: event F1 at 200\,ms, median filter, global post-processing (so values differ from the "
             r"per-class Table~\ref{tab:main}). +pp: gap filling and "
             r"minimum duration; vote: frame voting on fusion's per-member decisions; fusion with one decoder setting for "
             r"all members (shared), per-member settings (own) or weighted edges (wtd.). $\dagger$/$\ast$: above/below "
             r"averaging (vote: plain; fusion: +pp).}",
             r"\label{tab:controls}", r"\begin{tabular}{lccccccc}", r"\toprule",
             r" & \multicolumn{3}{c}{averaging} & frame & \multicolumn{3}{c}{event fusion} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){6-8}",
             r"setting & prob. & logit & +pp & vote & shared & own & wtd. \\", r"\midrule"]
    for label, f in rows:
        t = load(f)
        if t is None:
            continue
        cells = []
        for arm, ref in arms:
            v = fmt(t[arm]["f1"][K200])
            if ref and f"d_{ref}" in t[arm]:
                sg = sig(t, arm, ref, K200)
                v += MARK if sg > 0 else (NEG if sg < 0 else "")
            cells.append(v)
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\vspace{-3mm}", r"\end{table}"]
    return "\n".join(lines)


def table_mechanism():
    """Table 3: edge error on commonly matched references, fragmentation and false alarms, with single-member controls."""
    sets = [("DESED, zero-shot", "support_v5_desed_raw", "BEATs"),
            ("URBAN, zero-shot", "support_v4_urban_zs_raw", "BEATs"),
            ("URBAN, adapted", "support_v4_urban_ad_raw", "BEATs-adapt")]
    sets = [(l, json.load(open(os.path.join(R, f + ".json")))["stats"], m) for l, f, m in sets]
    arms = [("single", "single:{m}"), ("avg.", "avg_prob"), ("avg.+pp", "avg_prob_pp"), ("fusion", "wbf_median"),
            ("single+cSEBB", "sebb:single:{m}"), ("avg.+cSEBB", "sebb:avg_prob"), ("fus.+cSEBB", "wbf_median_sebb")]
    n = len(sets)
    lines = [r"\begin{table}[t]", r"\centering", r"\setlength{\tabcolsep}{1.5pt}", r"\footnotesize",
             r"\caption{Error decomposition at 200\,ms (global post-processing, cross-fitted). On/off: median "
             r"onset/offset error (ms) on the references that every arm matches one-to-one at a 1\,s collar. Split: "
             r"references overlapped by two or more emitted events (\%). FA: false alarms overlapping no reference. "
             r"Single: BEATs. avg.+pp: DESED only.}",
             r"\label{tab:mech}", r"\begin{tabular}{l" + "ccc" * n + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{l}}}" for l, _, _ in sets) + r" \\",
             "".join(rf"\cmidrule(lr){{{2 + 3 * k}-{4 + 3 * k}}}" for k in range(n)),
             "arm" + " & on/off & split & FA" * n + r" \\", r"\midrule"]
    for label, key in arms:
        cells = []
        for _, st, m in sets:
            x = st.get(key.format(m=m))
            if x is None:
                cells += ["--"] * 3
                continue
            cells += [f"{1000 * x['common']['on']:.0f}/{1000 * x['common']['off']:.0f}",
                      f"{100 * x['frag']['split_2plus']:.1f}", f"{x['fp_kind'].get('spurious', 0)}"]
        lines.append(label + " & " + " & ".join(cells) + r" \\")
        if label == "fusion":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\vspace{-3mm}", r"\end{table}"]
    return "\n".join(lines)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [("tab_main.tex", table_main), ("tab_controls.tex", table_controls), ("tab_mech.tex", table_mechanism)]:
        with open(os.path.join(OUT, name), "w") as fh:
            fh.write(fn() + "\n")
        print(name)
