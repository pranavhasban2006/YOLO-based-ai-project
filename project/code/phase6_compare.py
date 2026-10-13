"""Phase 6: comparison tables, charts and error analysis from the Colab metrics.

Usage: python -I phase6_compare.py <project_dir>

Reads colab_results/ only. Writes results/comparison_all.{csv,png}, results/E3_summary.{csv,png},
results/E3_error_pairs.csv, results/E3_confusion_transcribed.csv and logs/phase_6_compare.json.
Does not touch state.json; the caller sets phase_done after this runs clean.
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase6_confusion as conf  # noqa: E402

ROOT = Path(sys.argv[1]).resolve()
SRC = ROOT / "colab_results"
RESULTS = ROOT / "results"
LOG = ROOT / "logs" / "phase_6_compare.json"
DATASETS = ["fer2013", "jaffe", "ravdess", "crema_d"]
SHARED6 = ["angry", "disgust", "fear", "happy", "neutral", "sad"]

# Colours from palette.md, categorical slots 1 and 2 (validated in both modes by validate_palette.js).
# Light surface, since these are PNG charts shipped on a light page.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
SLOT1 = "#2a78d6"
SLOT2 = "#eb6834"

# One reason per top-3 confused pair, from the four allowed by the spec. Each is a hypothesis
# from the pattern of counts and the dataset facts in DECISIONS.md. None of them was tested.
REASONS = {
    ("fer2013", "neutral", "disgust"): ("acted vs spontaneous",
        "fer2013 is spontaneous, 48 px, web-sourced; every training set (JAFFE, RAVDESS, CREMA-D) is acted."),
    ("fer2013", "happy", "disgust"): ("acted vs spontaneous",
        "Same domain gap: spontaneous faces do not look like the posed training smiles."),
    ("fer2013", "sad", "disgust"): ("acted vs spontaneous",
        "Same domain gap; disgust receives 60% of all fer2013 predictions (see log)."),
    ("jaffe", "angry", "neutral"): ("acted vs spontaneous",
        "JAFFE is posed; low-intensity posed anger reads as neutral. Only 30 angry images, so the rate is noisy."),
    ("jaffe", "disgust", "sad"): ("label noise",
        "Posed disgust and sad are close in JAFFE; 29 images in the class, so the ambiguity is per-image."),
    ("jaffe", "fear", "sad"): ("label noise",
        "Posed fear and sad are close in JAFFE; 32 images in the class."),
    ("ravdess", "angry", "neutral"): ("acted vs spontaneous",
        "Acted anger at normal intensity is low-arousal and reads as neutral."),
    ("ravdess", "sad", "disgust"): ("label noise",
        "Acted negative-valence clips overlap; the sad and disgust labels are not cleanly separated."),
    ("ravdess", "neutral", "happy"): ("label noise",
        "Documented label-definition mismatch: RAVDESS calm is merged into neutral (Phase 1 decision), so 'neutral' is not one emotion."),
    ("crema_d", "sad", "neutral"): ("label noise",
        "Low-intensity sad against neutral is a soft boundary, the usual place for label disagreement in acted sets."),
    ("crema_d", "angry", "neutral"): ("acted vs spontaneous",
        "Acted anger at low intensity reads as neutral."),
    ("crema_d", "fear", "neutral"): ("acted vs spontaneous",
        "Acted fear at low intensity reads as neutral."),
}


def load_metrics(run):
    return json.loads((SRC / run / "metrics.json").read_text(encoding="utf-8"))


def macro_f1_shared6(metrics_test):
    """Mean F1 over the six classes every dataset shares (no surprise). Derived from the saved per-class F1."""
    return float(np.mean([metrics_test["per_class"][c]["f1"] for c in SHARED6]))


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    log = {"skipped_e3": [], "e3_done": [], "warnings": []}

    # ---- E2 (combined training, per-dataset test splits) -----------------------------------------
    e2 = {}
    for arch in ["cnn", "resnet18"]:
        m = load_metrics(f"E2_{arch}")
        e2[arch] = {ds: m["tests"][ds] for ds in DATASETS}

    # ---- E3 (leave-one-dataset-out, CNN only: the Colab E2 winner) ------------------------------
    e3 = {}
    for ds in DATASETS:
        run = f"E3_cnn_holdout_{ds}"
        done = (SRC / run / "done.json").exists()
        if done:
            e3[ds] = load_metrics(run)["tests"][ds]
            log["e3_done"].append(ds)
        else:
            log["skipped_e3"].append(ds)

    # ---- comparison_all.csv -------------------------------------------------------------------
    rows = []
    for ds in DATASETS:
        for arch in ["cnn", "resnet18"]:
            t = e2[arch][ds]
            # E1 was not run on Colab. Phase 3 decided the E1 table reads from the E2 per-dataset test.
            rows.append({"experiment": "E1", "model": arch, "dataset": ds, "n_test": t["n"],
                         "accuracy": t["accuracy"], "macro_f1": t["macro_f1"], "status": "proxy_from_E2"})
            rows.append({"experiment": "E2", "model": arch, "dataset": ds, "n_test": t["n"],
                         "accuracy": t["accuracy"], "macro_f1": t["macro_f1"], "status": "done"})
        t = e3.get(ds)
        rows.append({"experiment": "E3", "model": "cnn", "dataset": ds,
                     "n_test": t["n"] if t else "", "accuracy": t["accuracy"] if t else "",
                     "macro_f1": t["macro_f1"] if t else "", "status": "done" if t else "skipped"})
        rows.append({"experiment": "E3", "model": "resnet18", "dataset": ds, "n_test": "",
                     "accuracy": "", "macro_f1": "", "status": "not_run_winner_only"})
    for arch in ["cnn", "resnet18"]:
        for exp, src in [("E1", e2[arch]), ("E2", e2[arch])]:
            accs = [src[d]["accuracy"] for d in DATASETS]
            f1s = [src[d]["macro_f1"] for d in DATASETS]
            rows.append({"experiment": exp, "model": arch, "dataset": "mean", "n_test": "",
                         "accuracy": float(np.mean(accs)), "macro_f1": float(np.mean(f1s)),
                         "status": "proxy_from_E2" if exp == "E1" else "done"})
    e3_cnn = [e3[d] for d in DATASETS if d in e3]
    if e3_cnn:
        rows.append({"experiment": "E3", "model": "cnn", "dataset": "mean", "n_test": "",
                     "accuracy": float(np.mean([t["accuracy"] for t in e3_cnn])),
                     "macro_f1": float(np.mean([t["macro_f1"] for t in e3_cnn])),
                     "status": "done" if len(e3_cnn) == len(DATASETS) else "partial"})
    write_csv(RESULTS / "comparison_all.csv", rows,
              ["experiment", "model", "dataset", "n_test", "accuracy", "macro_f1", "status"])

    # ---- comparison_all.png -----------------------------------------------------------------
    labels = DATASETS + ["mean"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True, facecolor=SURFACE)
    exps = [("E1", "E1 (proxy: E2 test splits)"), ("E2", "E2 (combined training)"), ("E3", "E3 (leave-one-dataset-out)")]
    x = np.arange(len(labels))
    w = 0.38
    for ax, (exp, title) in zip(axes, exps):
        ax.set_facecolor(SURFACE)
        for k, (arch, colour, off) in enumerate([("cnn", SLOT1, -w / 2), ("resnet18", SLOT2, w / 2)]):
            vals = []
            for ds in labels:
                if exp == "E3":
                    if ds == "mean":
                        vals.append(np.mean([e3[d]["macro_f1"] for d in e3]) if arch == "cnn" and e3 else np.nan)
                    else:
                        vals.append(e3[ds]["macro_f1"] if arch == "cnn" and ds in e3 else np.nan)
                else:
                    src = e2[arch]
                    vals.append(np.mean([src[d]["macro_f1"] for d in DATASETS]) if ds == "mean"
                                else src[ds]["macro_f1"])
            vals = np.array(vals, dtype=float)
            bars = ax.bar(x + off, np.nan_to_num(vals), w, color=colour, label=arch.replace("resnet18", "ResNet-18").replace("cnn", "CNN"))
            for b, v in zip(bars, vals):
                if np.isnan(v):
                    continue
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7, color=INK2)
        if exp == "E3":
            ax.text(0.5, 0.85, "ResNet-18 not run\n(Colab E3 used the winner, CNN)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color=INK2)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xticks(x, [d.replace("_", "-") if d != "mean" else "mean" for d in labels], fontsize=8, color=INK2)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(BASE)
        ax.tick_params(axis="y", colors=MUTED, labelsize=8, length=0)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("macro-F1", color=INK2, fontsize=9)
    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="upper right", frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Macro-F1 by dataset: CNN vs ResNet-18", x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.01, "crema_d has no surprise class and is scored on 6 classes. E1 was not run on Colab; "
                         "the E2 per-dataset test splits are shown in its place (Phase 3 decision).",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(RESULTS / "comparison_all.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)

    # ---- E3_summary.csv ----------------------------------------------------------------------
    summ = []
    for ds in DATASETS:
        in_ds = e2["cnn"][ds]
        t = e3.get(ds)
        row = {"held_out": ds, "status": "done" if t else "skipped",
               "in_dataset_E2_macro_f1": in_ds["macro_f1"], "in_dataset_E2_accuracy": in_ds["accuracy"],
               "E3_macro_f1": t["macro_f1"] if t else "", "E3_accuracy": t["accuracy"] if t else "",
               "macro_f1_drop": (in_ds["macro_f1"] - t["macro_f1"]) if t else "",
               "accuracy_drop": (in_ds["accuracy"] - t["accuracy"]) if t else "",
               "E3_macro_f1_6_shared": macro_f1_shared6(t) if t else "",
               "E3_n_test": t["n"] if t else ""}
        summ.append(row)
    write_csv(RESULTS / "E3_summary.csv", summ,
              ["held_out", "status", "in_dataset_E2_macro_f1", "in_dataset_E2_accuracy", "E3_macro_f1",
               "E3_accuracy", "macro_f1_drop", "accuracy_drop", "E3_macro_f1_6_shared", "E3_n_test"])

    # ---- E3_summary.png ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(len(DATASETS))
    w = 0.38
    ins = [summ[i]["in_dataset_E2_macro_f1"] for i in range(len(DATASETS))]
    held = [summ[i]["E3_macro_f1"] if summ[i]["E3_macro_f1"] != "" else np.nan for i in range(len(DATASETS))]
    b1 = ax.bar(x - w / 2, ins, w, color=SLOT1, label="in-dataset (E2, CNN)")
    b2 = ax.bar(x + w / 2, np.nan_to_num(held), w, color=SLOT2, label="held-out (E3, CNN)")
    for bars, vals in [(b1, ins), (b2, held)]:
        for b, v in zip(bars, vals):
            if np.isnan(v):
                continue
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=8, color=INK2)
    for i in range(len(DATASETS)):
        if not np.isnan(held[i]):
            ax.text(x[i], max(ins[i], held[i]) + 0.08, f"drop {ins[i] - held[i]:+.2f}", ha="center",
                    fontsize=8, color=INK, fontweight="bold")
    ax.set_xticks(x, [d.replace("_", "-") for d in DATASETS], fontsize=9, color=INK2)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("macro-F1", color=INK2, fontsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.tick_params(axis="y", colors=MUTED, labelsize=8, length=0)
    ax.tick_params(axis="x", length=0)
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    ax.set_title("E3: held-out macro-F1 against in-dataset score (CNN)", fontsize=11, color=INK, loc="left")
    fig.text(0.01, 0.01, "Skipped E3 runs: none. fer2013 held-out is below chance (accuracy 0.072; chance 0.143). See the phase 6 report.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(RESULTS / "E3_summary.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)

    # ---- error analysis ---------------------------------------------------------------------
    mats = conf.load_all()  # raises if the transcription does not match the saved metrics
    trans_rows = []
    for ds, cm in mats.items():
        for i, t in enumerate(conf.CLASSES):
            for j, p in enumerate(conf.CLASSES):
                trans_rows.append({"held_out": ds, "true": t, "predicted": p, "count": int(cm[i][j])})
    write_csv(RESULTS / "E3_confusion_transcribed.csv", trans_rows, ["held_out", "true", "predicted", "count"])

    err_rows = []
    for ds in DATASETS:
        for rank, (count, rate, t, p) in enumerate(conf.top_pairs(mats[ds], 3), start=1):
            reason, note = REASONS[(ds, t, p)]
            err_rows.append({"held_out": ds, "rank": rank, "true": t, "predicted": p, "count": count,
                             "row_rate": round(rate, 4), "likely_reason": reason, "reason_note": note})
    write_csv(RESULTS / "E3_error_pairs.csv", err_rows,
              ["held_out", "rank", "true", "predicted", "count", "row_rate", "likely_reason", "reason_note"])

    # ---- log ---------------------------------------------------------------------------------
    # Share of fer2013 predictions that land on disgust (the sink class) and accuracy vs chance.
    fer = mats["fer2013"]
    log["fer2013_pred_disgust_share"] = float(fer[:, 1].sum() / fer.sum())
    log["fer2013_accuracy"] = e3["fer2013"]["accuracy"] if "fer2013" in e3 else None
    log["e3_summary"] = summ
    log["error_pairs"] = err_rows
    LOG.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"e3 done: {log['e3_done']}  skipped: {log['skipped_e3']}")
    print(f"fer2013 predicted disgust share: {log['fer2013_pred_disgust_share']:.3f}")
    for r in err_rows:
        print(f"{r['held_out']:8} {r['rank']} {r['true']:>8} -> {r['predicted']:<8} {r['count']:>6} {r['row_rate']:.3f} {r['likely_reason']}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
