r"""Builds the Project Review 2 document (dataset description + methodology) from the real datasets.

Every count, sample image, waveform, signal plot and class-balance figure is computed from the data on disk;
diagrams are drawn with matplotlib. Output: project/reports/review2/Review2_Multimodal_Emotion.docx

Usage (repo root): project\.venv-train\Scripts\python project\code\make_review2.py
"""
import collections
import io
import sys
import tempfile
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import soundfile as sf  # noqa: E402
from docx import Document  # noqa: E402
from docx.enum.table import WD_TABLE_ALIGNMENT  # noqa: E402
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.shared import Inches, Pt, RGBColor  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "Datasets"
DATA = ROOT / "project" / "data"
OUT = ROOT / "project" / "reports" / "review2"
FIG = OUT / "fig"
sys.path.insert(0, str(Path(__file__).parent))

TITLE = "Multimodal Emotion Recognition using Hybrid Deep Learning Models on Facial, Speech, Text and Physiological Data"
TEAM = [("Prakul Jain", "24BIT0548"), ("Pranav Hasban", "24BIT0537"), ("Shreya Mehrotra", "24BIT0547")]
APP7 = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
C_IMB, C_BAL = "#d9534f", "#5cb85c"

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    p = FIG / f"{name}.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


# ------------------------------------------------------------------------------------------------ data stats

def fer_counts():
    return {s: {c: len(list((DATA / "fer2013" / s / c).glob("*"))) for c in APP7} for s in ("train", "test")}


def ck_files():
    z = zipfile.ZipFile(DS / "Image and Video Dataset" / "CKPLUS.zip")
    by = collections.defaultdict(dict)
    for n in z.namelist():
        if n.endswith(".png"):
            by[n.split("/")[-2]].setdefault(n.split("/")[-1], n)        # de-duplicate nested copies
    return z, {c: sorted(v.values()) for c, v in sorted(by.items())}


def ferplus_counts():
    z = zipfile.ZipFile(DS / "Image and Video Dataset" / "FERPlus-master.zip")
    df = pd.read_csv(z.open("FERPlus-master/fer2013new.csv"))
    cols = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt", "unknown", "NF"]
    return df, df[cols].idxmax(axis=1).value_counts().reindex(cols, fill_value=0)


RAV_EMO = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad", "05": "angry", "06": "fearful",
           "07": "disgust", "08": "surprised"}


def ravdess_video():
    files = []
    for i in range(1, 25):
        p = DS / f"Video_Speech_Actor_{i:02d}.zip"
        if p.exists():
            files += [(p, n) for n in zipfile.ZipFile(p).namelist() if n.endswith(".mp4")]
    return files


def audio_items():
    import train_audio as ta
    items = []
    for src in ("ravdess", "tess", "savee"):
        for p in sorted((DATA / "audio" / src).rglob("*.wav")):
            lab = ta.label_of(src, p)
            if lab:
                items.append((src, lab, p))
    return items


def text_sets():
    import train_text as tt
    go = {s: tt.load_split("goemotions", s) for s in ("train", "validation", "test")}
    em = {s: pq.read_table(DATA / "text" / "emotion" / f"{s}.parquet").to_pydict() for s in ("train", "validation", "test")}
    z = zipfile.ZipFile(DS / "tweeteval-main.zip")
    base = "tweeteval-main/datasets/emotion/"
    rd = lambda f: z.read(base + f).decode("utf-8").strip().split("\n")
    tw = {s: (rd(f"{s}_text.txt"), [int(v) for v in rd(f"{s}_labels.txt")]) for s in ("train", "val", "test")}
    tw_names = [l.split("\t")[1] for l in rd("mapping.txt")]
    return go, em, (tw, tw_names)


# ------------------------------------------------------------------------------------------------ figures

def bal_figure(name, title, classes, counts, note="SMOTE"):
    counts = np.asarray(counts)
    bal = np.full_like(counts, counts.max())
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, v, col, t in ((axs[0], counts, C_IMB, "Imbalanced (original)"),
                          (axs[1], bal, C_BAL, f"Balanced (after {note})")):
        b = ax.bar(classes, v, color=col)
        ax.bar_label(b, fontsize=8)
        ax.set_title(f"{t}: {v.sum():,} samples")
        ax.tick_params(axis="x", rotation=35)
    axs[0].set_ylabel("Number of samples")
    fig.suptitle(title, fontweight="bold", y=1.06)
    return save(fig, name)


def image_grid(name, title, classes, loaders, cmap="gray"):
    fig, axs = plt.subplots(2, len(classes), figsize=(1.6 * len(classes), 3.6))
    for j, c in enumerate(classes):
        for i in range(2):
            axs[i, j].imshow(loaders[c][i](), cmap=cmap)
            axs[i, j].axis("off")
        axs[0, j].set_title(c, fontsize=10)
    fig.suptitle(title, fontweight="bold")
    return save(fig, name)


def audio_figure(samples):
    import torch
    import torchaudio
    mel = torchaudio.transforms.MelSpectrogram(16000, n_fft=1024, hop_length=320, n_mels=64)
    fig, axs = plt.subplots(len(APP7), 4, figsize=(12, 13))
    for i, c in enumerate(APP7):
        for k, (src, p) in enumerate(samples[c]):
            w, sr = sf.read(p, dtype="float32", always_2d=True)
            w = w.mean(1)
            if sr != 16000:
                w = torchaudio.functional.resample(torch.from_numpy(w), sr, 16000).numpy()
            t = np.arange(len(w)) / 16000
            axs[i, 2 * k].plot(t, w, lw=0.4, color="#337ab7")
            axs[i, 2 * k].set_xlim(0, t[-1])
            axs[i, 2 * k].set_title(f"{c} – {src.upper()} – {p.name}", fontsize=7.5)
            m = torch.log(mel(torch.from_numpy(w)) + 1e-6).numpy()
            axs[i, 2 * k + 1].imshow(m, origin="lower", aspect="auto", cmap="magma", extent=[0, t[-1], 0, 64])
            axs[i, 2 * k + 1].set_title("log-mel spectrogram (64 bands)", fontsize=7.5)
            for a in axs[i, 2 * k:2 * k + 2]:
                a.tick_params(labelsize=6)
        axs[i, 0].set_ylabel(c, fontsize=10, fontweight="bold")
    fig.suptitle("Speech samples: 2 clips per emotion (waveform and log-mel spectrogram)", fontweight="bold")
    fig.tight_layout()
    return save(fig, "audio_samples")


def wesad_figure(x, y, classes):
    names = ["ECG", "EDA", "EMG", "Resp", "Temp", "ACC_x", "ACC_y", "ACC_z"]
    fig, axs = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    t = np.arange(x.shape[-1]) / 35
    for i, c in enumerate(classes):
        idx = np.flatnonzero(y == i)
        for k, j in enumerate((idx[len(idx) // 4], idx[3 * len(idx) // 4])):
            ax = axs[i, k]
            for ch, off in zip((0, 1, 3, 4), (0, -6, -12, -18)):
                ax.plot(t, x[j, ch].astype(np.float32) + off, lw=0.6, label=names[ch])
            ax.set_yticks([0, -6, -12, -18], ["ECG", "EDA", "Resp", "Temp"])
            ax.set_title(f"{c} – window #{j} (60 s, 35 Hz, z-scored)", fontsize=9)
    axs[-1, 0].set_xlabel("time (s)")
    axs[-1, 1].set_xlabel("time (s)")
    fig.suptitle("WESAD chest signals: 2 sample windows per class", fontweight="bold")
    fig.tight_layout()
    return save(fig, "wesad_samples")


# ------------------------------------------------------------------------------------------------ diagrams

def box(ax, x, y, w, h, text, fc="#e8f1fb", ec="#2e6da4", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, lw=1.2, color="#333"))


def canvas(w, h, size):
    fig, ax = plt.subplots(figsize=size)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def methodology_flowchart():
    fig, ax = canvas(10, 15, (8.5, 12.5))
    B = dict(fc="#e8f1fb", ec="#2e6da4")
    G = dict(fc="#eaf6ea", ec="#3c8d3c")
    O = dict(fc="#fff4e0", ec="#d58512")
    steps = [(14.2, "Load dataset\n(images / audio / text / signals → feature CSV)"),
             (12.9, "Data preprocessing\n(cleaning, resizing / resampling, normalisation, label mapping)"),
             (11.6, "Data split")]
    for y, t in steps:
        box(ax, 5, y, 6.2, 0.9, t, **B)
    arrow(ax, (5, 13.75), (5, 13.35))
    arrow(ax, (5, 12.45), (5, 12.05))
    box(ax, 2.6, 10.0, 3.6, 0.9, "Hold-out split\n80:20 and 70:30", **O)
    box(ax, 7.4, 10.0, 3.6, 0.9, "10-fold stratified\ncross-validation", **O)
    arrow(ax, (4.0, 11.15), (2.6, 10.45))
    arrow(ax, (6.0, 11.15), (7.4, 10.45))
    box(ax, 1.5, 8.4, 2.4, 0.9, "With SMOTE\n(training data only)", **G)
    box(ax, 3.9, 8.4, 2.2, 0.9, "Without SMOTE", **G)
    box(ax, 6.3, 8.4, 2.4, 0.9, "With SMOTE\n(training folds only)", **G)
    box(ax, 8.7, 8.4, 2.2, 0.9, "Without SMOTE", **G)
    for a, b in (((2.2, 9.55), (1.5, 8.85)), ((3.0, 9.55), (3.9, 8.85)),
                 ((7.0, 9.55), (6.3, 8.85)), ((7.8, 9.55), (8.7, 8.85))):
        arrow(ax, a, b)
    box(ax, 5, 6.7, 8.6, 1.0, "Building the model – training the algorithms\n"
        "(5 hybrid deep models per modality: face, speech, text, physiological)", **B)
    for xx in (1.5, 3.9, 6.3, 8.7):
        arrow(ax, (xx, 7.95), (5 + (xx - 5) * 0.8, 7.2))
    box(ax, 5, 5.1, 8.6, 1.0, "Overfitting / underfitting check & rectify\n"
        "(train-vs-validation gap; early stopping, L1 + L2 (elastic-net) regularisation, dropout)", **O)
    arrow(ax, (5, 6.2), (5, 5.6))
    box(ax, 5, 3.6, 8.6, 0.9, "Test the model\n(accuracy, precision, recall, macro-F1, confusion matrix, inference time)", **B)
    arrow(ax, (5, 4.6), (5, 4.05))
    box(ax, 5, 2.2, 6.2, 0.9, "Selection of the best model\n(per modality, by 10-fold macro-F1)", fc="#dff0d8",
        ec="#3c763d", bold=True)
    arrow(ax, (5, 3.15), (5, 2.65))
    box(ax, 5, 0.85, 6.2, 0.8, "Deploy best models in the web application\n(FastAPI backend + React frontend)",
        fc="#f2f2f2", ec="#777")
    arrow(ax, (5, 1.75), (5, 1.25))
    ax.set_title("Proposed methodology – flowchart", fontweight="bold", fontsize=13)
    return save(fig, "flowchart_methodology")


def system_architecture():
    fig, ax = canvas(14, 9, (13, 8.2))
    mods = [("Face images\nFER-2013 · CK+ · FER+", "Face crop · resize 224×224\nRGB · ImageNet norm · augment",
             "EffNetB3+CBAM · ResNet50+ViT\nDenseNet121+SE · MobileNetV3+Attn\nCNN+BiLSTM"),
            ("Speech audio\nRAVDESS · TESS · SAVEE", "Mono · 16 kHz · 4 s crop/pad\nlog-mel / SSL encoder frames",
             "wav2vec2+BiLSTM+Attn · CNN+BiLSTM+Attn\nHuBERT+Transformer · AST+MLP\nCNN+GRU"),
            ("Text\nGoEmotions · dair-ai Emotion\n(TweetEval)", "Clean · label map (Ekman)\nWordPiece/BPE tokens · len 64",
             "RoBERTa+BiLSTM+Attn · BERT+BiLSTM+Attn\nDeBERTa+Attn · DistilBERT+CNN\nBERT+CNN+BiLSTM"),
            ("Physiological\nWESAD (chest RespiBAN)", "700→35 Hz · per-subject z-score\n60 s windows, 5 s hop",
             "CNN-LSTM · CNN-BiLSTM · 1D-CNN-GRU\nAutoencoder-MLP\nTransformer-BiLSTM")]
    ys = [7.6, 5.6, 3.6, 1.6]
    for (src, pre, mdl), y in zip(mods, ys):
        box(ax, 1.6, y, 2.9, 1.4, src, fc="#fff4e0", ec="#d58512", fs=8.5)
        box(ax, 4.9, y, 3.0, 1.4, pre, fs=8)
        box(ax, 8.5, y, 3.6, 1.4, mdl, fc="#eaf6ea", ec="#3c8d3c", fs=7.6)
        arrow(ax, (3.05, y), (3.4, y))
        arrow(ax, (6.4, y), (6.7, y))
        arrow(ax, (10.3, y), (11.0, 4.6 + (y - 4.6) * 0.25))
    box(ax, 12.4, 4.6, 2.6, 2.6, "Feature CSV\n→ SMOTE · 80:20 / 70:30\n· 10-fold CV\n→ metrics\n→ best model\nper modality",
        fc="#dff0d8", ec="#3c763d", fs=8.5, bold=True)
    ax.text(1.6, 8.75, "Data source", ha="center", fontweight="bold")
    ax.text(4.9, 8.75, "Preprocessing", ha="center", fontweight="bold")
    ax.text(8.5, 8.75, "Hybrid deep models (5 each)", ha="center", fontweight="bold")
    ax.text(12.4, 6.35, "Evaluation & selection", ha="center", fontweight="bold")
    ax.set_title("Overall system architecture", fontweight="bold", fontsize=13)
    return save(fig, "system_architecture")


def arch_figure(name, title, rows):
    fig, ax = canvas(16, len(rows) * 1.5 + 0.4, (14, len(rows) * 1.25 + 0.6))
    for r, (label, blocks) in enumerate(rows):
        y = len(rows) * 1.5 - 0.6 - r * 1.5
        ax.text(0.05, y, label, fontweight="bold", fontsize=9, va="center")
        n = len(blocks)
        x0, span = 3.4, 12.4
        w = span / n - 0.35
        for k, b in enumerate(blocks):
            x = x0 + (k + 0.5) * span / n
            fc = "#fff4e0" if k == 0 else "#dff0d8" if k == n - 1 else "#e8f1fb"
            box(ax, x, y, w, 1.05, b, fc=fc, fs=7.4)
            if k:
                arrow(ax, (x - span / n + w / 2, y), (x - w / 2, y))
    ax.set_title(title, fontweight="bold", fontsize=12)
    return save(fig, name)


# ------------------------------------------------------------------------------------------------ document

class Doc:
    def __init__(self):
        self.d = Document()
        st = self.d.styles["Normal"]
        st.font.name, st.font.size = "Times New Roman", Pt(12)
        for s in ("Heading 1", "Heading 2", "Heading 3"):
            self.d.styles[s].font.name = "Times New Roman"
            self.d.styles[s].font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
        self.fig_no = 0
        self.tab_no = 0

    def h(self, text, level):
        self.d.add_heading(text, level)

    def p(self, text, bold_lead=None):
        para = self.d.add_paragraph()
        if bold_lead:
            para.add_run(bold_lead).bold = True
        para.add_run(text)
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        return para

    def bullets(self, items, style="List Bullet"):
        for it in items:
            para = self.d.add_paragraph(style=style)
            if isinstance(it, tuple):
                para.add_run(it[0]).bold = True
                para.add_run(it[1])
            else:
                para.add_run(it)

    def img(self, path, caption, width=6.3):
        self.d.add_picture(str(path), width=Inches(width))
        self.d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.fig_no += 1
        c = self.d.add_paragraph(f"Figure {self.fig_no}: {caption}")
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c.runs[0].italic = True
        c.runs[0].font.size = Pt(10)

    def table(self, header, rows, caption, widths=None, font=10):
        self.tab_no += 1
        c = self.d.add_paragraph(f"Table {self.tab_no}: {caption}")
        c.alignment = WD_ALIGN_PARAGRAPH.CENTER
        c.runs[0].bold = True
        c.runs[0].font.size = Pt(10)
        t = self.d.add_table(rows=1, cols=len(header))
        t.style = "Table Grid"
        t.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, h in enumerate(header):
            cell = t.rows[0].cells[i]
            cell.text = ""
            r = cell.paragraphs[0].add_run(str(h))
            r.bold = True
            r.font.size = Pt(font)
        for row in rows:
            cells = t.add_row().cells
            for i, v in enumerate(row):
                cells[i].text = ""
                cells[i].paragraphs[0].add_run(str(v)).font.size = Pt(font)
        if widths:
            for row in t.rows:
                for i, w in enumerate(widths):
                    row.cells[i].width = Inches(w)
        self.d.add_paragraph()

    def balance_table(self, classes, counts, caption):
        counts = list(map(int, counts))
        mx = max(counts)
        rows = [(c, f"{v:,}", f"{100 * v / sum(counts):.1f}%", f"{mx:,}", f"+{mx - v:,}") for c, v in zip(classes, counts)]
        rows.append(("Total", f"{sum(counts):,}", "100%", f"{mx * len(counts):,}", f"+{mx * len(counts) - sum(counts):,}"))
        self.table(["Class label", "Imbalanced size", "Share", "Balanced size (SMOTE)", "Synthetic samples added"],
                   rows, caption)
        ratio = mx / max(min(counts), 1)
        verdict = "imbalanced" if ratio > 1.5 else "approximately balanced"
        self.p(f"Imbalance ratio (largest class ÷ smallest class) = {ratio:.1f} → the dataset is {verdict}. "
               + ("SMOTE is therefore applied to the training portion so that every class reaches the size of the "
                  "majority class." if ratio > 1.5 else
                  "SMOTE is still evaluated (as required by the protocol) but changes the class sizes only slightly."))

    def save(self, path):
        self.d.save(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    doc = Doc()
    d = doc.d

    # ---------------------------------------------------------------- title block
    t = d.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("PROJECT REVIEW – 2")
    r.bold, r.font.size = True, Pt(16)
    t = d.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run(TITLE)
    r.bold, r.font.size = True, Pt(15)
    doc.table(["S. No.", "Name", "Registration number"], [(i + 1, n, reg) for i, (n, reg) in enumerate(TEAM)],
              "Team members", widths=[0.8, 2.6, 2.2], font=11)

    # ================================================================ 3. DATASETS
    doc.h("3. Dataset Description and Sample Data", 1)
    doc.p("The project recognises human emotion from four modalities – facial images (with video as an extension), "
          "speech audio, written text and physiological signals – so one or more public benchmark datasets were "
          "collected for every modality. Table 2 lists them. For each dataset the following subsections give the "
          "download link, a description (features, class label, size), sample data (2 samples per class), a "
          "class-imbalance check with the balanced/imbalanced size table and the corresponding figures.")
    vids = ravdess_video()
    n_act = len({Path(n).stem.split("-")[-1] for _, n in vids})
    overview = [
        ("Face (image)", "FER-2013", "35,887 grayscale images, 48×48", "7 emotions", "Primary – all 5 face models"),
        ("Face (image)", "CK+ (CK+48)", "981 grayscale images, 48×48", "7 (incl. contempt)", "Supplementary"),
        ("Face (labels)", "FER+", "35,887 crowd-sourced label votes", "8 + unknown", "Cleaner labels for FER-2013"),
        ("Video", "RAVDESS (video-speech)", f"{len(vids):,} .mp4 clips ({n_act} of 24 actors downloaded)", "8 emotions", "Extension"),
        ("Speech", "RAVDESS (audio-speech)", "1,440 .wav clips, 24 actors", "8 (7 used)", "Audio models"),
        ("Speech", "TESS", "5,600 .wav clips, 2 speakers", "7 emotions", "Audio models"),
        ("Speech", "SAVEE", "480 .wav clips, 4 speakers", "7 emotions", "Audio models"),
        ("Text", "GoEmotions", "54,263 Reddit comments", "27 + neutral → 7 (Ekman)", "Text models"),
        ("Text", "dair-ai Emotion", "20,000 English tweets", "6 emotions", "Text models"),
        ("Text", "TweetEval – emotion", "5,052 tweets", "4 emotions", "Supplementary"),
        ("Physiological", "WESAD", "15 subjects, 8 chest channels @ 700 Hz", "3 affective states",
         "Physiological models"),
    ]
    doc.table(["Modality", "Dataset", "Size", "Class labels", "Role in project"], overview, "Overview of datasets",
              font=9)

    # ---------------------------------------------------------------- 3.1 FER-2013
    doc.h("3.1 Facial Expression Datasets", 2)
    doc.h("3.1.1 FER-2013", 3)
    fc = fer_counts()
    doc.bullets([("Link: ", "https://www.kaggle.com/datasets/msambare/fer2013 "
                  "(original: ICML 2013 Challenges in Representation Learning)")])
    tot_tr, tot_te = sum(fc["train"].values()), sum(fc["test"].values())
    doc.p(f"FER-2013 contains {tot_tr + tot_te:,} grayscale face images collected automatically with Google image "
          f"search. Every image is 48 × 48 pixels, the face is roughly centred and occupies most of the frame. "
          f"The dataset is split into {tot_tr:,} training images and {tot_te:,} test images.", "Description: ")
    doc.bullets([("Features: ", "the 48 × 48 = 2,304 pixel intensities (0–255, one channel). For the deep models "
                  "the image is resized to 224 × 224 and replicated to 3 channels; the learned 512–2,048-dimensional "
                  "embedding of each model is exported as a feature CSV (f0 … fN) for the SMOTE / k-fold protocol."),
                 ("Class label: ", "7 emotions – angry, disgust, fear, happy, neutral, sad, surprise."),
                 ("Total number of images: ", f"{tot_tr + tot_te:,} (train {tot_tr:,} / test {tot_te:,}).")])
    doc.table(["Class label", "Train images", "Test images", "Total"],
              [(c, f"{fc['train'][c]:,}", f"{fc['test'][c]:,}", f"{fc['train'][c] + fc['test'][c]:,}") for c in APP7]
              + [("Total", f"{tot_tr:,}", f"{tot_te:,}", f"{tot_tr + tot_te:,}")], "FER-2013 class distribution")
    files = {c: sorted((DATA / "fer2013" / "train" / c).glob("*")) for c in APP7}
    pick = {c: [files[c][i] for i in rng.choice(len(files[c]), 2, replace=False)] for c in APP7}
    p = image_grid("fer_samples", "FER-2013 – 2 sample images per class", APP7,
                   {c: [lambda f=f: Image.open(f) for f in pick[c]] for c in APP7})
    doc.img(p, "FER-2013 sample images (2 per class)")
    doc.p("Sample data: the figure above shows two randomly chosen training images of each class. The images vary "
          "strongly in pose, illumination, age, occlusion (hands, glasses) and image quality, which makes FER-2013 "
          "a hard, realistic ('in the wild') benchmark.")
    doc.balance_table(APP7, [fc["train"][c] for c in APP7], "FER-2013 (training set) – imbalanced vs balanced size")
    doc.img(bal_figure("fer_balance", "FER-2013 training set – class distribution", APP7,
                       [fc["train"][c] for c in APP7]), "FER-2013 class distribution before and after SMOTE")
    doc.p("The 'disgust' class has only 436 training images (1.5%) against 7,215 'happy' images (25.1%), so the "
          "dataset is strongly imbalanced. Because SMOTE interpolates between feature vectors, it is applied to the "
          "exported deep-feature CSV (not to raw pixels) and only to the training part of every split.")

    # ---------------------------------------------------------------- CK+
    doc.h("3.1.2 CK+ (Extended Cohn-Kanade, CK+48 version)", 3)
    z, ck = ck_files()
    ck_classes = list(ck)
    ck_counts = [len(ck[c]) for c in ck_classes]
    doc.bullets([("Link: ", "https://www.kaggle.com/datasets/shawon10/ckplus  (original: Lucey et al., "
                  "CVPR Workshops 2010)")])
    doc.p(f"CK+ is a laboratory dataset of posed facial expressions recorded from 123 subjects. The CK+48 version "
          f"contains the last (peak-expression) frames of each sequence, cropped to the face and resized to "
          f"48 × 48 grayscale – {sum(ck_counts)} images in total.", "Description: ")
    doc.bullets([("Features: ", "48 × 48 grayscale pixel values (same format as FER-2013)."),
                 ("Class label: ", f"{len(ck_classes)} classes – " + ", ".join(ck_classes) + "."),
                 ("Total number of images: ", f"{sum(ck_counts)}.")])
    p = image_grid("ck_samples", "CK+ – 2 sample images per class", ck_classes,
                   {c: [lambda n=n: Image.open(io.BytesIO(z.read(n))) for n in (ck[c][0], ck[c][len(ck[c]) // 2])]
                    for c in ck_classes})
    doc.img(p, "CK+ sample images (2 per class)")
    doc.balance_table(ck_classes, ck_counts, "CK+ – imbalanced vs balanced size")
    doc.img(bal_figure("ck_balance", "CK+ – class distribution", ck_classes, ck_counts),
            "CK+ class distribution before and after SMOTE")

    # ---------------------------------------------------------------- FER+
    doc.h("3.1.3 FER+ (FER-2013 re-labelled by Microsoft)", 3)
    fp_df, fp_counts = ferplus_counts()
    doc.bullets([("Link: ", "https://github.com/microsoft/FERPlus")])
    doc.p("FER+ does not contain new images; it provides new labels for every FER-2013 image. Ten crowd-workers "
          "voted for each image, which removes much of the label noise of the original dataset. It is used as a "
          "cleaner-label variant of FER-2013.", "Description: ")
    doc.bullets([("Features: ", "the FER-2013 image (referenced by file name) + 10 vote columns."),
                 ("Class label: ", "neutral, happiness, surprise, sadness, anger, disgust, fear, contempt, "
                  "plus 'unknown' and 'not a face (NF)'; the majority vote is the label."),
                 ("Total number of rows: ", f"{len(fp_df):,}.")])
    vote_cols = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
    maj = fp_df[vote_cols + ["unknown", "NF"]].idxmax(axis=1)
    rows = []
    for c in vote_cols:
        for _, rr in fp_df[maj == c].head(2).iterrows():
            rows.append([rr["Usage"], rr["Image name"]] + [int(rr[v]) for v in vote_cols] + [c])
    doc.table(["Usage", "Image", "neu", "hap", "sur", "sad", "ang", "dis", "fea", "con", "Label"], rows,
              "FER+ sample rows (CSV) – 2 per class", font=8)
    doc.balance_table(list(fp_counts.index), fp_counts.values, "FER+ – majority-vote class sizes")
    doc.img(bal_figure("ferplus_balance", "FER+ – majority-vote label distribution", list(fp_counts.index),
                       fp_counts.values), "FER+ class distribution before and after SMOTE")

    # ---------------------------------------------------------------- 3.2 VIDEO
    doc.h("3.2 Video Dataset – RAVDESS (Video-Speech)", 2)
    emo_of = lambda n: RAV_EMO[Path(n).stem.split("-")[2]]
    vcount = collections.Counter(emo_of(n) for _, n in vids)
    actors = sorted({Path(n).stem.split("-")[-1] for _, n in vids})
    doc.bullets([("Link: ", "https://zenodo.org/records/1188976  (Livingstone & Russo, PLoS ONE 2018)")])
    doc.p(f"The Ryerson Audio-Visual Database of Emotional Speech and Song contains 24 professional actors "
          f"(12 female, 12 male) speaking two lexically-matched sentences ('Kids are talking by the door', "
          f"'Dogs are sitting by the door') in 8 emotions at two intensities. Each clip is 1280 × 720, 30 fps, "
          f"about 3–4 s long, provided as audio-video and video-only. Actors {int(actors[0])}–{int(actors[-1])} "
          f"({len(actors)} of 24) have been downloaded so far ({len(vids):,} clips)"
          + (f"; the remaining actors can be added from the same link." if len(actors) < 24 else ".")
          + " Each actor contributes 120 clips (60 audio-video + 60 video-only).", "Description: ")
    doc.bullets([("Features: ", "RGB video frames (face region) + synchronised speech; the file name encodes "
                  "modality, channel, emotion, intensity, statement, repetition and actor."),
                 ("Class label: ", "neutral, calm, happy, sad, angry, fearful, disgust, surprised."),
                 ("Total number of clips: ", f"{len(vids):,} downloaded (2,880 for all 24 actors).")])
    tmp = Path(tempfile.mkdtemp())
    import cv2
    frames = {}
    for e in RAV_EMO.values():
        cand = [(zp, n) for zp, n in vids if emo_of(n) == e and Path(n).stem.startswith("02")]
        frames[e] = []
        for zp, n in (cand[0], cand[len(cand) // 2]):
            f = tmp / Path(n).name
            f.write_bytes(zipfile.ZipFile(zp).read(n))
            cap = cv2.VideoCapture(str(f))
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.6))
            ok, fr = cap.read()
            cap.release()
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            h_, w_ = fr.shape[:2]
            frames[e].append(lambda fr=fr[:, (w_ - h_) // 2:(w_ + h_) // 2]: fr)
    p = image_grid("video_samples", "RAVDESS video – one frame from 2 clips per emotion", list(RAV_EMO.values()),
                   frames, cmap=None)
    doc.img(p, "RAVDESS video sample frames (2 clips per emotion)")
    vc = [vcount[e] for e in RAV_EMO.values()]
    doc.balance_table(list(RAV_EMO.values()), vc, "RAVDESS video (downloaded actors) – class sizes")
    doc.img(bal_figure("video_balance", "RAVDESS video – class distribution", list(RAV_EMO.values()), vc),
            "RAVDESS video class distribution before and after SMOTE")
    doc.p("RAVDESS is balanced by design except for 'neutral', which has no strong-intensity version and therefore "
          "half as many clips as every other emotion.")

    # ---------------------------------------------------------------- 3.3 AUDIO
    doc.h("3.3 Speech Emotion Datasets", 2)
    items = audio_items()
    src_count = collections.Counter(s for s, _, _ in items)
    doc.p("Three acted speech-emotion corpora are combined and mapped to the same 7 emotion classes used for the "
          "face model, giving one speech dataset of "
          f"{len(items):,} clips (RAVDESS {src_count['ravdess']:,} + TESS {src_count['tess']:,} + "
          f"SAVEE {src_count['savee']:,}). All clips are converted to mono, resampled to 16 kHz and centre-cropped or "
          "zero-padded to 4 s.")
    doc.table(["Dataset", "Link", "Speakers", "Clips used", "Original labels → project labels"], [
        ("RAVDESS (audio-speech)", "https://zenodo.org/records/1188976", "24 actors (12F/12M)",
         f"{src_count['ravdess']:,}", "8 emotions; 'calm' dropped (no counterpart), others mapped 1:1"),
        ("TESS", "https://tspace.library.utoronto.ca/handle/1807/24487", "2 actresses (26 & 64 yrs)",
         f"{src_count['tess']:,}", "7 emotions; 'pleasant surprise' → surprise"),
        ("SAVEE", "http://kahlan.eps.surrey.ac.uk/savee/", "4 male speakers", f"{src_count['savee']:,}",
         "7 emotions (a, d, f, h, n, sa, su)"),
        ("CREMA-D (not used yet)", "https://github.com/CheyneyComputerScience/CREMA-D", "91 actors", "0",
         "Downloaded copy contained Git-LFS pointer files only; to be re-downloaded")],
        "Speech datasets", font=8.5)
    doc.bullets([("Features: ", "raw waveform (64,000 samples at 16 kHz) → 64-band log-mel spectrogram for the "
                  "CNN models, or frame embeddings (768-d) from the pretrained wav2vec 2.0 / HuBERT / AST encoders."),
                 ("Class label: ", "7 emotions – " + ", ".join(APP7) + "."),
                 ("Total number of clips: ", f"{len(items):,}.")])
    by = collections.defaultdict(list)
    for s, lab, pth in items:
        by[lab].append((s, pth))
    # one clip from two different corpora per class, so the samples show speaker/recording variety
    pick_src = lambda c, s: next(x for x in by[c] if x[0] == s)
    samples = {c: [pick_src(c, "ravdess"), pick_src(c, "savee")] for c in APP7}
    rows = []
    for c in APP7:
        for s, pth in samples[c]:
            info = sf.info(pth)
            rows.append((c, s.upper(), pth.name, f"{info.samplerate} Hz", f"{info.duration:.2f} s"))
    doc.table(["Class label", "Dataset", "File", "Sample rate", "Duration"], rows,
              "Speech sample data – 2 clips per class", font=9)
    doc.img(audio_figure(samples), "Speech samples: waveform and log-mel spectrogram, 2 clips per class", width=6.5)
    ac = collections.Counter(l for _, l, _ in items)
    doc.balance_table(APP7, [ac[c] for c in APP7], "Combined speech dataset – imbalanced vs balanced size")
    doc.img(bal_figure("audio_balance", "Combined speech dataset – class distribution", APP7, [ac[c] for c in APP7]),
            "Speech class distribution before and after SMOTE")
    fig, ax = plt.subplots(figsize=(9, 3.3))
    bottom = np.zeros(7)
    for s, col in (("ravdess", "#337ab7"), ("tess", "#5cb85c"), ("savee", "#f0ad4e")):
        v = np.array([sum(1 for ss, l, _ in items if ss == s and l == c) for c in APP7])
        ax.bar(APP7, v, bottom=bottom, label=s.upper(), color=col)
        bottom += v
    ax.legend()
    ax.set_ylabel("Number of clips")
    ax.set_title("Speech clips per class and source dataset", fontweight="bold")
    doc.img(save(fig, "audio_sources"), "Contribution of each speech corpus to every class")

    # ---------------------------------------------------------------- 3.4 TEXT
    doc.h("3.4 Text Emotion Datasets", 2)
    go, em, (tw, tw_names) = text_sets()
    EM6 = ["sadness", "joy", "love", "anger", "fear", "surprise"]

    def text_samples(texts, labels, names, k=2):
        rows, seen = [], collections.Counter()
        for tx, lb in zip(texts, labels):
            if seen[lb] < k and 25 < len(tx) < 140:
                rows.append((names[lb], tx.replace("\n", " ")))
                seen[lb] += 1
        return sorted(rows, key=lambda r: names.index(r[0]))

    doc.h("3.4.1 GoEmotions", 3)
    n_go_raw = 43410 + 5426 + 5427
    go_tr = collections.Counter(go["train"][1])
    doc.bullets([("Link: ", "https://huggingface.co/datasets/google-research-datasets/go_emotions  "
                  "(Demszky et al., ACL 2020)")])
    doc.p(f"GoEmotions contains {n_go_raw:,} English Reddit comments annotated with 27 fine-grained emotions plus "
          f"neutral (multi-label). Following the authors' Ekman grouping, the 28 labels are merged into the project's "
          f"7 classes; comments whose labels fall into more than one group are removed, leaving "
          f"{sum(len(v[1]) for v in go.values()):,} single-label comments "
          f"(train {len(go['train'][1]):,} / validation {len(go['validation'][1]):,} / test {len(go['test'][1]):,}).",
          "Description: ")
    doc.bullets([("Features: ", "the comment text (tokenised to at most 64 sub-word tokens); columns in the file: "
                  "text, labels (list of label ids), id."),
                 ("Class label: ", "angry, disgust, fear, happy, neutral, sad, surprise (Ekman grouping)."),
                 ("Total number of rows: ", f"{n_go_raw:,} raw; {sum(len(v[1]) for v in go.values()):,} after "
                  "mapping.")])
    doc.table(["Class label", "Text (CSV sample)"], text_samples(*go["train"], APP7),
              "GoEmotions sample data – 2 rows per class", widths=[1.2, 5.3], font=9)
    doc.balance_table(APP7, [go_tr[i] for i in range(7)], "GoEmotions (training set) – imbalanced vs balanced size")
    doc.img(bal_figure("go_balance", "GoEmotions training set – class distribution", APP7,
                       [go_tr[i] for i in range(7)]), "GoEmotions class distribution before and after SMOTE")

    doc.h("3.4.2 dair-ai Emotion", 3)
    em_tr = collections.Counter(em["train"]["label"])
    doc.bullets([("Link: ", "https://huggingface.co/datasets/dair-ai/emotion  (Saravia et al., EMNLP 2018)")])
    doc.p("English Twitter messages labelled with six basic emotions using distant supervision (emotion hashtags), "
          "split 16,000 / 2,000 / 2,000 for train / validation / test.", "Description: ")
    doc.bullets([("Features: ", "text (lower-cased tweet); columns: text, label."),
                 ("Class label: ", "sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5)."),
                 ("Total number of rows: ", "20,000.")])
    doc.table(["Class label", "Text (CSV sample)"], text_samples(em["train"]["text"], em["train"]["label"], EM6),
              "dair-ai Emotion sample data – 2 rows per class", widths=[1.2, 5.3], font=9)
    doc.balance_table(EM6, [em_tr[i] for i in range(6)], "dair-ai Emotion (training set) – imbalanced vs balanced size")
    doc.img(bal_figure("em_balance", "dair-ai Emotion training set – class distribution", EM6,
                       [em_tr[i] for i in range(6)]), "dair-ai Emotion class distribution before and after SMOTE")

    doc.h("3.4.3 TweetEval – Emotion subset", 3)
    tw_tr = collections.Counter(tw["train"][1])
    n_tw = sum(len(v[1]) for v in tw.values())
    doc.bullets([("Link: ", "https://github.com/cardiffnlp/tweeteval  (Barbieri et al., Findings of EMNLP 2020)")])
    doc.p(f"The emotion task of the TweetEval benchmark (from SemEval-2018 Task 1) has {n_tw:,} tweets "
          f"(train {len(tw['train'][1]):,} / validation {len(tw['val'][1]):,} / test {len(tw['test'][1]):,}) "
          f"labelled with four emotions. It is kept as a supplementary text benchmark.", "Description: ")
    doc.bullets([("Features: ", "tweet text (user names replaced by @user); stored as parallel *_text.txt and "
                  "*_labels.txt files."),
                 ("Class label: ", ", ".join(f"{n} ({i})" for i, n in enumerate(tw_names)) + "."),
                 ("Total number of rows: ", f"{n_tw:,}.")])
    doc.table(["Class label", "Text (CSV sample)"], text_samples(*tw["train"], tw_names),
              "TweetEval-emotion sample data – 2 rows per class", widths=[1.2, 5.3], font=9)
    doc.balance_table(tw_names, [tw_tr[i] for i in range(len(tw_names))],
                      "TweetEval-emotion (training set) – imbalanced vs balanced size")
    doc.img(bal_figure("tw_balance", "TweetEval-emotion training set – class distribution", tw_names,
                       [tw_tr[i] for i in range(len(tw_names))]),
            "TweetEval-emotion class distribution before and after SMOTE")

    # ---------------------------------------------------------------- 3.5 WESAD
    doc.h("3.5 Physiological Dataset – WESAD", 2)
    wz = np.load(DATA / "wesad_windows.npz")
    wy, wsubj = wz["y"], wz["subj"]
    WC = ["baseline", "stress", "amusement"]
    doc.bullets([("Link: ", "https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html  (also UCI: "
                  "https://archive.ics.uci.edu/dataset/465/wesad) – Schmidt et al., ICMI 2018")])
    doc.p("WESAD (Wearable Stress and Affect Detection) records 15 subjects (S2–S17; S1 and S12 discarded by the "
          "authors) wearing a chest RespiBAN and a wrist Empatica E4 during a neutral baseline (~20 min), a stress "
          "condition (Trier Social Stress Test: public speaking + mental arithmetic, ~10 min) and an amusement "
          "condition (funny video clips, ~6.5 min). The chest device samples all channels at 700 Hz.",
          "Description: ")
    doc.bullets([("Features: ", "8 chest channels – ECG, EDA (electrodermal activity), EMG, respiration, skin "
                  "temperature and 3-axis acceleration. After preprocessing each sample is a 60-second window of "
                  "8 × 2,100 values (35 Hz); for the CSV representation each window is summarised by per-channel "
                  "statistics (mean, standard deviation, minimum, maximum → 32 features)."),
                 ("Class label: ", "baseline (1), stress (2), amusement (3) – the WESAD protocol labels."),
                 ("Total: ", f"15 subjects, ~2.25 GB raw; {len(wy):,} windows after segmentation.")])
    per = pd.crosstab(pd.Series(wsubj, name="subject"), pd.Series([WC[i] for i in wy], name="class"))
    per = per.reindex(sorted(per.index, key=lambda s: int(s[1:])))
    doc.table(["Subject"] + WC + ["Total"],
              [(s, *[int(per.loc[s, c]) for c in WC], int(per.loc[s].sum())) for s in per.index]
              + [("Total", *[int(per[c].sum()) for c in WC], int(per.values.sum()))],
              "WESAD – 60 s windows per subject and class", font=9)
    wx = wz["x"]
    rows = []
    for i, c in enumerate(WC):
        idx = np.flatnonzero(wy == i)
        for j in (idx[len(idx) // 4], idx[3 * len(idx) // 4]):
            w = wx[j].astype(np.float32)
            rows.append([wsubj[j]] + [f"{w[ch].mean():.2f}" for ch in (0, 1, 3, 4)]
                        + [f"{w[ch].std():.2f}" for ch in (0, 1, 3, 4)] + [c])
    doc.table(["Subj", "ECG μ", "EDA μ", "Resp μ", "Temp μ", "ECG σ", "EDA σ", "Resp σ", "Temp σ", "Label"], rows,
              "WESAD feature CSV sample – 2 windows per class (subset of the 32 statistical features; "
              "values are per-subject z-scores)", font=8)
    doc.img(wesad_figure(wx, wy, WC), "WESAD sample signal windows (2 per class)", width=6.5)
    wc_counts = [int((wy == i).sum()) for i in range(3)]
    doc.balance_table(WC, wc_counts, "WESAD windows – imbalanced vs balanced size")
    doc.img(bal_figure("wesad_balance", "WESAD – class distribution of 60 s windows", WC, wc_counts),
            "WESAD class distribution before and after SMOTE")
    del wx

    # ================================================================ 4. METHODOLOGY
    doc.h("4. Methodology", 1)
    doc.h("4.1 Steps Involved in the Proposed Methodology", 2)
    doc.bullets([
        "Data collection – download the face, video, speech, text and physiological datasets (Section 3).",
        "Data loading – read each dataset and convert it to a common tabular form (one row per sample, "
        "feature columns + class label).",
        "Data preprocessing – modality-specific cleaning, resizing / resampling, normalisation, segmentation, "
        "tokenisation and label mapping.",
        "Data splitting – (a) hold-out splits of 80:20 and 70:30, and (b) stratified 10-fold cross-validation; "
        "the same splits are used for every model of a modality.",
        "Class balancing – SMOTE applied to the training data only; every experiment is run both with and "
        "without SMOTE.",
        "Model building – train five hybrid deep-learning models per modality (20 models in total).",
        "Overfitting / underfitting check and rectification – compare training and validation scores; "
        "apply early stopping, L1 + L2 regularisation, dropout, data augmentation.",
        "Testing – evaluate on held-out data with accuracy, precision, recall, macro-F1, confusion matrix and "
        "inference time.",
        "Best-model selection – choose the best model per modality and integrate it into the web application.",
    ], style="List Number")
    doc.img(system_architecture(), "Overall system architecture of the proposed multimodal emotion recognition system",
            width=6.6)

    doc.h("4.2 Detailed Explanation of Each Step", 2)
    doc.h("Step 1 – Data collection and loading", 3)
    doc.p("All datasets are public research benchmarks (links in Section 3). Images are read from class folders, "
          "audio from .wav files whose labels are decoded from the file names, text from Parquet/TXT files and "
          "WESAD from the official per-subject pickle files (synchronised signals + labels). Every modality is "
          "finally represented as a CSV-style table with numeric feature columns (f0 … fN) and a 'label' column, "
          "which is the input format of the evaluation protocol.")
    doc.h("Step 2 – Data preprocessing", 3)
    doc.table(["Modality", "Preprocessing operations"], [
        ("Face images", "Decode → resize 48×48 to 224×224 → convert grayscale to 3 channels → normalise with "
         "ImageNet mean/std. Training-time augmentation: random horizontal flip, ±10° rotation, random resized "
         "crop, colour jitter. At inference the face is located with a face detector and cropped first."),
        ("Video", "Sample frames from each clip → detect and crop the face → process each frame like a face "
         "image; the per-frame features form a time sequence."),
        ("Speech", "Load → convert to mono → resample to 16 kHz → centre-crop / zero-pad to 4 s (64,000 samples) "
         "→ skip corrupt or placeholder files (< 1 kB) → map labels to 7 classes. CNN models compute a 64-band "
         "log-mel spectrogram (1024-point FFT, 20 ms hop) normalised per clip; SSL models use the encoder's own "
         "feature extractor."),
        ("Text", "Map labels (GoEmotions 28 → 7 Ekman groups; drop multi-group comments) → tokenise with the "
         "encoder's tokenizer (WordPiece / BPE / SentencePiece) → truncate / pad to 64 tokens → attention mask."),
        ("Physiological", "Take the 8 chest channels at 700 Hz → down-sample to 35 Hz by mean pooling (anti-alias "
         "+ 20× fewer samples) → z-score per subject (removes person-specific offsets) → keep only labels 1/2/3 → "
         "cut 60 s windows with a 5 s hop inside each continuous condition segment."),
    ], "Preprocessing per modality", widths=[1.3, 5.2], font=9)
    doc.h("Step 3 – Data splitting", 3)
    doc.p("Two evaluation schemes are used, exactly as in the flowchart: (a) a single stratified hold-out split, "
          "run once with 80% training / 20% testing and once with 70% / 30%; (b) stratified 10-fold "
          "cross-validation, where the data are cut into 10 equal folds with the same class proportions, each fold "
          "is used once as the test set while the other nine train the model, and the mean ± standard deviation of "
          "the 10 scores is reported. A fixed random seed makes the splits identical for every model so that the "
          "comparison is fair. For WESAD, overlapping neighbouring windows are kept together in the same fold "
          "(grouped splitting), and the deep models are trained on subjects different from the test subjects, so "
          "no information leaks from training to testing.")
    doc.h("Step 4 – Class balancing with SMOTE", 3)
    doc.p("SMOTE (Synthetic Minority Over-sampling Technique) creates new minority-class samples instead of "
          "duplicating existing ones: for a minority sample x it picks one of its k = 5 nearest neighbours x_nn of "
          "the same class and generates x_new = x + λ·(x_nn − x) with λ drawn uniformly from [0, 1]. This is "
          "repeated until every class has as many samples as the majority class (the 'balanced size' in the "
          "tables of Section 3). SMOTE is fitted on the training part only – never on the test part – otherwise "
          "synthetic copies of test samples would inflate the scores. Because SMOTE interpolates numeric vectors, "
          "it is applied to the feature CSV (deep embeddings or statistical features), not to raw images, audio or "
          "text.")
    doc.h("Step 5 – Building the models (hybrid deep learning)", 3)
    doc.p("For every modality the five hybrid architectures specified for the project are implemented in PyTorch. "
          "Pretrained backbones (ImageNet CNNs, self-supervised speech encoders, pretrained language models) "
          "provide strong general features; the attached attention / recurrent / transformer blocks model the "
          "emotion-specific spatial or temporal patterns. All models end in a dropout + fully-connected softmax "
          "layer. Training uses class-weighted cross-entropy (label smoothing 0.1 for images and speech), the AdamW "
          "optimiser with weight decay, a one-cycle (warm-up + cosine decay) learning-rate schedule, mixed-precision "
          "(AMP) on an NVIDIA RTX 3050 GPU and gradient clipping.")
    doc.table(["Modality", "Hybrid model", "How it works"], [
        ("Face", "EfficientNetB3 + CBAM", "EfficientNet-B3 feature map → Convolutional Block Attention Module "
         "(channel attention then spatial attention) → global pooling → classifier"),
        ("Face", "ResNet50 + ViT", "ResNet-50 (up to layer 3) gives a 14×14 grid of local features → 196 tokens + "
         "[CLS] token → 4-layer Vision-Transformer encoder → classifier on [CLS]"),
        ("Face", "DenseNet121 + SE", "DenseNet-121 features → Squeeze-and-Excitation channel re-weighting → "
         "global pooling → classifier"),
        ("Face", "MobileNetV3 + Attention", "Lightweight MobileNetV3-Large → learned spatial-attention pooling → "
         "classifier (fast, suitable for real-time use)"),
        ("Face", "CNN + BiLSTM", "ResNet-18 CNN → 7×7 feature map read as a 49-step sequence → bidirectional LSTM → "
         "attention pooling → classifier (on video the sequence is the frames)"),
        ("Speech", "wav2vec2 + BiLSTM + Attention", "wav2vec 2.0 frame embeddings (768-d) → BiLSTM → additive "
         "attention pooling → classifier"),
        ("Speech", "CNN + BiLSTM + Attention", "Log-mel spectrogram → 4-block 2-D CNN → BiLSTM over time → "
         "attention pooling → classifier"),
        ("Speech", "HuBERT + Transformer", "HuBERT frame embeddings → 2-layer Transformer encoder → mean pooling → "
         "classifier"),
        ("Speech", "AST + MLP", "Audio Spectrogram Transformer (AudioSet-pretrained) clip embedding → MLP "
         "(LayerNorm, 256 units, GELU, dropout)"),
        ("Speech", "CNN + GRU", "Log-mel spectrogram → 2-D CNN → GRU → last hidden state → classifier"),
        ("Text", "RoBERTa + BiLSTM + Attention", "RoBERTa-base token states → BiLSTM → attention pooling → "
         "classifier (encoder fine-tuned end-to-end)"),
        ("Text", "BERT + BiLSTM + Attention", "BERT-base token states → BiLSTM → attention pooling → classifier"),
        ("Text", "DeBERTa + Attention", "DeBERTa-v3-base (disentangled attention) → attention pooling → classifier"),
        ("Text", "DistilBERT + CNN", "DistilBERT token states → parallel 1-D convolutions (kernel 2/3/4) → "
         "max-over-time pooling → classifier"),
        ("Text", "BERT + CNN + BiLSTM", "BERT token states → 1-D CNN (local n-grams) → BiLSTM (context) → "
         "attention pooling → classifier"),
        ("Physiological", "CNN-LSTM", "3-block 1-D CNN (×8 down-sampling) → LSTM → mean pooling → classifier"),
        ("Physiological", "CNN-BiLSTM", "Same 1-D CNN → bidirectional LSTM → classifier"),
        ("Physiological", "1D-CNN-GRU", "Same 1-D CNN → GRU → classifier"),
        ("Physiological", "Autoencoder-MLP", "1-D convolutional autoencoder learns a compressed code "
         "(reconstruction loss) → MLP classifies the code; trained jointly"),
        ("Physiological", "Transformer-BiLSTM", "1-D CNN tokens + positional embedding → 2-layer Transformer "
         "encoder → BiLSTM → classifier"),
    ], "Hybrid models implemented for each modality", widths=[1.0, 1.8, 3.7], font=8.5)
    doc.img(arch_figure("arch_face", "Face – hybrid architectures", [
        ("EfficientNetB3+CBAM", ["Face 224×224", "EfficientNet-B3", "CBAM\n(channel+spatial)", "Global pool", "FC → 7"]),
        ("ResNet50+ViT", ["Face 224×224", "ResNet-50\n(14×14 map)", "196 tokens\n+ [CLS]", "ViT encoder\n×4", "FC → 7"]),
        ("DenseNet121+SE", ["Face 224×224", "DenseNet-121", "SE block", "Global pool", "FC → 7"]),
        ("MobileNetV3+Attn", ["Face 224×224", "MobileNetV3-L", "Spatial\nattention", "Weighted pool", "FC → 7"]),
        ("CNN+BiLSTM", ["Face 224×224", "ResNet-18 CNN", "49-step\nsequence", "BiLSTM +\nattention", "FC → 7"])]),
        "Block diagrams of the five face models")
    doc.img(arch_figure("arch_audio", "Speech – hybrid architectures", [
        ("wav2vec2+BiLSTM+Attn", ["Wave 16 kHz", "wav2vec 2.0\nframes", "BiLSTM", "Attention\npool", "FC → 7"]),
        ("CNN+BiLSTM+Attn", ["Wave 16 kHz", "Log-mel", "2-D CNN", "BiLSTM +\nattention", "FC → 7"]),
        ("HuBERT+Transformer", ["Wave 16 kHz", "HuBERT\nframes", "Transformer\n×2", "Mean pool", "FC → 7"]),
        ("AST+MLP", ["Wave 16 kHz", "Fbank\n128 bins", "AST encoder", "MLP", "FC → 7"]),
        ("CNN+GRU", ["Wave 16 kHz", "Log-mel", "2-D CNN", "GRU", "FC → 7"])]),
        "Block diagrams of the five speech models")
    doc.img(arch_figure("arch_text", "Text – hybrid architectures", [
        ("RoBERTa+BiLSTM+Attn", ["Text ≤64 tok", "RoBERTa-base", "BiLSTM", "Attention\npool", "FC → C"]),
        ("BERT+BiLSTM+Attn", ["Text ≤64 tok", "BERT-base", "BiLSTM", "Attention\npool", "FC → C"]),
        ("DeBERTa+Attn", ["Text ≤64 tok", "DeBERTa-v3", "Attention\npool", "Dropout", "FC → C"]),
        ("DistilBERT+CNN", ["Text ≤64 tok", "DistilBERT", "Conv k=2,3,4", "Max pool", "FC → C"]),
        ("BERT+CNN+BiLSTM", ["Text ≤64 tok", "BERT-base", "1-D CNN", "BiLSTM +\nattention", "FC → C"])]),
        "Block diagrams of the five text models (C = 7 for GoEmotions, 6 for dair-ai Emotion)")
    doc.img(arch_figure("arch_physio", "Physiological (WESAD) – hybrid architectures", [
        ("CNN-LSTM", ["8×2100\nwindow", "1-D CNN ×3", "LSTM", "Mean pool", "FC → 3"]),
        ("CNN-BiLSTM", ["8×2100\nwindow", "1-D CNN ×3", "BiLSTM", "Mean pool", "FC → 3"]),
        ("1D-CNN-GRU", ["8×2100\nwindow", "1-D CNN ×3", "GRU", "Mean pool", "FC → 3"]),
        ("Autoencoder-MLP", ["8×2100\nwindow", "Conv encoder\n(+ decoder)", "Latent code", "MLP", "FC → 3"]),
        ("Transformer-BiLSTM", ["8×2100\nwindow", "1-D CNN\ntokens", "Transformer\n×2", "BiLSTM", "FC → 3"])]),
        "Block diagrams of the five physiological models")
    doc.h("Step 6 – Overfitting / underfitting check and rectification", 3)
    doc.p("After training, the training accuracy is compared with the validation accuracy. A gap larger than "
          "10 percentage points is flagged as overfitting; low training accuracy (< 50%) is flagged as "
          "underfitting. The following remedies are built into training:")
    doc.bullets([("Early stopping: ", "a 10% validation slice is carved out of the training data; the model is "
                  "evaluated after every epoch and the weights with the best validation macro-F1 are kept; training "
                  "stops after 10 epochs without improvement."),
                 ("L1 and L2 regularisation: ", "L2 (weight decay 10⁻³) through the AdamW optimiser and an L1 "
                  "penalty (10⁻⁵·Σ|w|) added to the loss – together an elastic-net penalty that keeps weights small "
                  "and sparse."),
                 ("Dropout ", "(0.2–0.3) before every classifier, label smoothing and data augmentation for images "
                  "and signals (random jitter noise)."),
                 ("Underfitting remedies: ", "pretrained backbones, more epochs, larger heads or fine-tuning "
                  "more encoder layers.")])
    doc.h("Step 7 – Testing the model", 3)
    doc.p("Each trained model is evaluated on data it never saw during training. The metrics are accuracy; "
          "per-class and macro-averaged precision, recall and F1-score (macro-F1 treats every class equally and is "
          "therefore the main metric for imbalanced data); the confusion matrix; and inference time per sample on "
          "CPU and GPU (important for the real-time web application).")
    doc.table(["Metric", "Formula / meaning"], [
        ("Accuracy", "(TP + TN) / (TP + TN + FP + FN) – fraction of correct predictions"),
        ("Precision", "TP / (TP + FP) – of the samples predicted as a class, how many are correct"),
        ("Recall", "TP / (TP + FN) – of the samples of a class, how many are found"),
        ("F1-score", "2 · Precision · Recall / (Precision + Recall); macro-F1 = mean over classes"),
        ("Confusion matrix", "rows = true class, columns = predicted class; shows which emotions are confused"),
        ("Inference time", "milliseconds per sample on CPU and GPU (batch size 1)")],
        "Evaluation metrics", widths=[1.5, 5.0], font=9.5)
    doc.h("Step 8 – Selection of the best model", 3)
    doc.p("For every modality the model with the highest mean macro-F1 under 10-fold cross-validation is selected "
          "(cross-validation is the most reliable estimate because every sample is tested once); accuracy and "
          "inference time are used to break ties. The selected face, speech, text and physiological models are "
          "then deployed in the project's web application (FastAPI backend and React frontend), which shows the "
          "predicted emotion of each modality in real time.")

    doc.h("4.3 Flowchart", 2)
    doc.img(methodology_flowchart(), "Flowchart of the proposed methodology", width=5.6)
    doc.p("The flowchart follows the prescribed pipeline: the dataset (in CSV form) is loaded and preprocessed and "
          "then split in two ways. In the left branch a single hold-out split (80:20 or 70:30) is made and the "
          "model is trained both with SMOTE-balanced training data and with the original training data. In the "
          "right branch 10-fold cross-validation is performed, again with and without SMOTE on the training folds. "
          "All variants train the same models, are checked for over/underfitting (early stopping, L1/L2 "
          "regularisation), are tested with the same metrics and are finally compared to select the best model.")

    path = OUT / "Review2_Multimodal_Emotion_v2.docx"
    doc.save(path)
    print(f"saved {path} ({doc.fig_no} figures, {doc.tab_no} tables)")


if __name__ == "__main__":
    main()
