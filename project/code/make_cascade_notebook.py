r"""Builds Colab_FULL_upload/Emotion_Cascade_Colab.ipynb: the two-stage A1 -> A2 cascade of all 15 hybrids in Colab.

The notebook unpacks the code + data parts already on Google Drive (FULL_1, FULL_4), writes cascade.py from the
copy embedded below (so nothing new has to be uploaded) and saves models + results to MyDrive/Cascade_results.

Usage (repo root): project\.venv-train\Scripts\python project\code\make_cascade_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "Colab_FULL_upload" / "Emotion_Cascade_Colab.ipynb"
CASCADE = (ROOT / "project" / "code" / "cascade.py").read_text(encoding="utf-8")
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md("""
# Two-stage hybrids: A1 alone, then A1's output → A2
**Team:** Prakul Jain (24BIT0548) · Pranav Hasban (24BIT0537) · Shreya Mehrotra (24BIT0547)

Every hybrid model is written **A1 + A2** (for example *CNN + GRU*). Here each one is run as a cascade:

| Stage | What happens | What we report |
|---|---|---|
| **1. A1 alone** | A1 (e.g. the CNN) + a plain classifier is trained and tested **by itself** | A1 accuracy: e.g. 80 right, 20 wrong out of 100 |
| **2. A1 → A2** | A1 is frozen. For every sample, **A1's output** (its features and its class scores) is **the input of A2** (e.g. the GRU). A2 is trained to correct A1: *final scores = A1 scores + A2(A1 output)* | A1 → A2 accuracy: e.g. 90 right |
| **Comparison** | Same test samples, sample by sample | **fixed** (A1 wrong → A2 right), **broken** (A1 right → A2 wrong), **net gain** |

A2's last layer starts at zero, so before stage 2 is trained the cascade gives exactly A1's answers; everything A2
changes is a correction it learned. At test time A2 does not know which samples A1 got wrong, so it re-checks all of
them; that is why we report both *fixed* and *broken*.

**Splits** (same as the original models): face FER-2013 **90/10** train/validation + official test set ·
speech **70/15/15** · text official train/validation/test. Early stopping on validation macro-F1, AdamW weight decay (L2).

| Modality | Hybrid (A1 + A2) |
|---|---|
| Text | RoBERTa + BiLSTM + Attention · BERT + BiLSTM + Attention · DeBERTa + Attention · DistilBERT + CNN · BERT + CNN + BiLSTM |
| Speech | wav2vec 2.0 + BiLSTM + Attention · CNN + BiLSTM + Attention · HuBERT + Transformer · AST + MLP · CNN + GRU |
| Face | EfficientNetB3 + CBAM · ResNet50 + ViT · DenseNet121 + SE · MobileNetV3 + Attention · CNN + BiLSTM |

**Runtime:** `Runtime → Change runtime type → T4 GPU`. Speech ≈ 15 min, face ≈ 3–4 h, text ≈ 5–7 h.
Everything is **saved to Google Drive** (`MyDrive/Cascade_results`). If Colab disconnects, run cells 1–2 again and
then the cell that was running: finished models are skipped and it continues where it stopped.
""")

md("## 1. Unpack the code and the datasets from Google Drive")
code("""
from google.colab import drive
drive.mount('/content/drive')
import os, zipfile, time
SRC = '/content/drive/MyDrive/Colab_FULL_upload'          # the folder with the FULL_*.zip parts
for p in ['FULL_1_code_results.zip', 'FULL_4_data.zip']:  # code + datasets (the old trained models are not needed)
    done = f'/content/.{p}.done'
    if os.path.exists(done): print('already unpacked', p); continue
    t = time.time(); print('unpacking', p, '...', flush=True)
    zipfile.ZipFile(f'{SRC}/{p}').extractall('/content')
    open(done, 'w').close(); print(f'  done in {time.time() - t:.0f} s')
R = '/content/YOLO-based-ai-project'
OUT = '/content/drive/MyDrive/Cascade_results'             # models + results are kept here
os.makedirs(OUT, exist_ok=True)
os.environ['CASCADE_OUT'] = OUT
os.chdir(R); print('project root:', R, '| results ->', OUT)
""")
code("""
!pip -q install sentencepiece
import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE - switch to a T4 GPU runtime')
""")

md("""
## 2. The cascade code (`project/code/cascade.py`)
Running this cell writes the file. Read it here: `A1` = A1 alone (stage 1), `Cascade` = A1's output fed into A2
(stage 2), `stage2()` counts fixed / broken samples. The A1 and A2 of every hybrid are listed in the tables
`A1S` (speech), `FACE` (face) and `tt.MODELS` (text, from `train_text.py`).
""")
code("%%writefile /content/YOLO-based-ai-project/project/code/cascade.py\n" + CASCADE)

md("## 3. Speech: 5 hybrids (≈ 15 min)\nThe CNN is A1 of both **CNN + BiLSTM + Attention** and **CNN + GRU**, so it is trained once and both A2s start from the same CNN output.")
code("!python project/code/cascade.py audio 2>&1 | grep -v -i warning")

md("## 4. Face: 5 hybrids on FER-2013 (≈ 30–60 min each)\nOne cell per hybrid, so you can run them one at a time.")
for h in ["mobilenet_attn", "cnn_bilstm", "densenet_se", "effnet_cbam", "resnet50_vit"]:
    code(f"!python project/code/cascade.py face --only {h} 2>&1 | grep -v -i warning")

md("## 5. Text: 5 hybrids × 2 datasets\nBERT is A1 of both **BERT + BiLSTM + Attention** and **BERT + CNN + BiLSTM**. DeBERTa on GoEmotions is the slowest (≈ 2 h).")
code("!python project/code/cascade.py text --dataset emotion 2>&1 | grep -v -i warning")
code("!python project/code/cascade.py text --dataset goemotions 2>&1 | grep -v -i warning")

md("## 6. Results: A1 alone vs A1 → A2, for every hybrid")
code("""
!python project/code/cascade.py summary > /dev/null
import json, glob, pandas as pd, numpy as np, matplotlib.pyplot as plt
pd.set_option('display.width', 250)
S = pd.read_csv(f'{OUT}/reports/cascade_summary.csv')
display(S)
""")
code("""
# Accuracy of A1 alone vs A1 -> A2
lab = S.hybrid + '\\n(' + S.dataset + ')'
x = np.arange(len(S))
plt.figure(figsize=(max(10, len(S) * 0.9), 5))
plt.bar(x - 0.2, S['A1 alone acc'], 0.4, label='Stage 1: A1 alone')
plt.bar(x + 0.2, S['A1 + A2 (wrong only) acc'], 0.4, label="Stage 2: A1's wrong ones -> A2")
plt.xticks(x, lab, rotation=60, ha='right', fontsize=8); plt.ylabel('test accuracy'); plt.ylim(0, 1)
plt.legend(); plt.title('A1 alone vs A1 -> A2 (same test samples)'); plt.tight_layout(); plt.show()
""")
code("""
# Every hybrid in words: out of N test samples A1 got X right; its wrong ones go to A2, which fixes F of them
for f in sorted(glob.glob(f'{OUT}/reports/*.json')):
    r = json.load(open(f)); a, w = r['stage1_A1_alone'], r['wrong_only_to_A2']
    print(f"{r['A1']} + {r['A2']}  [{r['modality']}, {r['dataset']}, test = {r['test_size']}]")
    print(f"   stage 1  {r['A1']} alone: {a['correct']} right, {a['wrong']} wrong  -> accuracy {a['accuracy']:.2%}")
    print(f"   stage 2  the {w['fed_to_A2']} wrong ones -> {r['A2']}: fixed {w['fixed_by_A2']}  -> "
          f"{a['correct'] + w['fixed_by_A2']} right, accuracy {w['accuracy']:.2%} "
          f"({100 * (w['accuracy'] - a['accuracy']):+.2f} points)\\n")
""")
code("""
# Confusion matrices of one hybrid before / after A2 (change the name to see another one)
NAME = 'audio_cnn_gru'
r = json.load(open(f'{OUT}/reports/{NAME}.json'))
fig, axs = plt.subplots(1, 2, figsize=(13, 5))
for ax, key, t in zip(axs, ['confusion_A1', 'confusion_A1_then_A2'], [f"{r['A1']} alone", f"{r['A1']} -> {r['A2']}"]):
    m = np.array(r[key]); ax.imshow(m, cmap='Blues'); ax.set_title(t)
    ax.set_xticks(range(len(r['classes']))); ax.set_xticklabels(r['classes'], rotation=45)
    ax.set_yticks(range(len(r['classes']))); ax.set_yticklabels(r['classes'])
    for i in range(len(m)):
        for j in range(len(m)): ax.text(j, i, m[i, j], ha='center', va='center', fontsize=8)
    ax.set_xlabel('predicted'); ax.set_ylabel('true')
plt.tight_layout(); plt.show()
""")
code("""
# The trained models (A1 alone and A1 -> A2) saved on Google Drive
!ls -la {OUT}/models | head -60
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}, "colab": {"provenance": []},
                                   "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 0}
NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("notebook:", NB)
