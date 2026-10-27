r"""Builds Colab_FULL_upload/Emotion_FULL_Project_Colab.ipynb: unpacks the 5 FULL_*.zip parts from Google Drive
(the whole repo: code, app, 27 trained models, feature CSVs, every dataset) and lets you browse, re-test, re-run the
protocol and retrain every model in Colab.

Usage (repo root): project\.venv-train\Scripts\python project\code\make_full_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "Colab_FULL_upload" / "Emotion_FULL_Project_Colab.ipynb"
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md("""
# Multimodal Emotion Recognition – the COMPLETE project in Colab
**Team:** Prakul Jain (24BIT0548) · Pranav Hasban (24BIT0537) · Shreya Mehrotra (24BIT0547)

Everything from the project laptop: all code, the web app, the **27 trained models**, the **feature CSV of every model**,
**every dataset** (FER-2013 images, GoEmotions / dair-ai text, RAVDESS / TESS / SAVEE speech, raw + processed WESAD),
all logs and reports.

| Section | What you can do |
|---|---|
| 1 | Unpack the project from Google Drive |
| 2 | Browse every file and folder |
| 3 | Look at every dataset |
| 4 | Load all 27 trained models |
| 5 | **Re-test** the trained models on the held-out test sets |
| 6 | **Re-run the faculty protocol** (80:20, 70:30, 10-fold, ± SMOTE) on every model |
| 7 | **Retrain** any model from scratch (same splits, same code) |

Use a **GPU runtime**: `Runtime → Change runtime type → T4 GPU`.
""")

md("## 1. Unpack the project from Google Drive (folder `MyDrive/Colab_FULL_upload`)")
code("""
from google.colab import drive
drive.mount('/content/drive')
import os, zipfile, time
SRC = '/content/drive/MyDrive/Colab_FULL_upload'     # <- the folder you uploaded
INCLUDE_WESAD_RAW = True                              # False skips the 17 GB raw WESAD part (the processed windows are in part 4)
parts = ['FULL_1_code_results.zip', 'FULL_2_models.zip', 'FULL_3_features.zip', 'FULL_4_data.zip'] + (['FULL_5_wesad_raw.zip'] if INCLUDE_WESAD_RAW else [])
for p in parts:
    done = f'/content/.{p}.done'
    if os.path.exists(done): print('already unpacked', p); continue
    t = time.time(); print('unpacking', p, '...', flush=True)
    zipfile.ZipFile(f'{SRC}/{p}').extractall('/content')
    open(done, 'w').close(); print(f'  done in {time.time() - t:.0f} s')
R = '/content/YOLO-based-ai-project'
os.chdir(R); print('project root:', R)
""")
code("""
!pip -q install imbalanced-learn mediapipe
import glob, json, re, sys, numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
from IPython.display import display, Image as IPImage, Audio
pd.set_option('display.max_rows', 300); pd.set_option('display.max_colwidth', 120)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (CPU)')
""")

md("## 2. Every file and folder of the project")
code("""
rows = []
for root, dirs, fs in os.walk(R):
    for f in fs:
        p = os.path.join(root, f); rows.append((os.path.relpath(p, R), os.path.getsize(p)))
files = pd.DataFrame(rows, columns=['path', 'bytes'])
files['top'] = files.path.str.split('/').str[:2].str.join('/')
summary = files.groupby('top').agg(files=('path', 'count'), MB=('bytes', lambda b: round(b.sum() / 1e6, 1)))
print(f'{len(files):,} files, {files.bytes.sum() / 1e9:.1f} GB'); display(summary)
""")
code("""
# Every file except the dataset files themselves (code, models, CSVs, reports, logs)
display(files[~files.path.str.startswith('project/data/')][['path', 'bytes']].reset_index(drop=True))
""")
code("""
!apt -qq install tree > /dev/null 2>&1; tree -d -L 4 {R} | head -150
""")

md("## 3. The datasets")
md("### 3.1 FER-2013 (face images)")
code("""
FER = f'{R}/project/data/fer2013'
counts = pd.DataFrame({s: {c: len(os.listdir(f'{FER}/{s}/{c}')) for c in sorted(os.listdir(f'{FER}/{s}'))} for s in ('train', 'test')})
display(counts.assign(total=counts.sum(1)))
fig, axs = plt.subplots(3, 7, figsize=(14, 6.5))
for j, c in enumerate(sorted(os.listdir(f'{FER}/train'))):
    for i, f in enumerate(sorted(os.listdir(f'{FER}/train/{c}'))[:3]):
        axs[i, j].imshow(plt.imread(f'{FER}/train/{c}/{f}'), cmap='gray'); axs[i, j].axis('off')
    axs[0, j].set_title(c)
plt.show()
""")
md("### 3.2 Text datasets (GoEmotions, dair-ai Emotion)")
code("""
for f in sorted(glob.glob(f'{R}/project/data/text/**/*.parquet', recursive=True)):
    df = pd.read_parquet(f); print(f'{os.path.relpath(f, R)}: {len(df):,} rows'); display(df.head(3))
""")
md("### 3.3 Speech (RAVDESS, TESS, SAVEE) – listen to clips")
code("""
for ds in ['ravdess', 'tess', 'savee']:
    wavs = sorted(glob.glob(f'{R}/project/data/audio/{ds}/**/*.wav', recursive=True))
    print(f'{ds}: {len(wavs):,} clips')
    for w in wavs[:2]: print(os.path.basename(w)); display(Audio(w))
""")
md("### 3.4 WESAD – raw recordings of each subject and the processed 60 s windows")
code("""
W = f'{R}/project/data/wesad/WESAD'
if os.path.isdir(W):
    print(sorted(os.listdir(W)))
    import pickle
    d = pickle.load(open(f'{W}/S2/S2.pkl', 'rb'), encoding='latin1')
    chest = d['signal']['chest']; lab = d['label']
    print({k: v.shape for k, v in chest.items()}, 'labels', np.unique(lab))
    fig, axs = plt.subplots(4, 1, figsize=(14, 7), sharex=True)
    for a, k in zip(axs, ['ECG', 'EDA', 'Resp', 'Temp']):
        a.plot(chest[k][:700 * 600:70, 0], lw=.6); a.set_ylabel(k)
    axs[0].set_title('Subject S2, first 10 minutes, chest sensor (700 Hz, every 70th sample)'); plt.show()
w = np.load(f'{R}/project/data/wesad_windows.npz')
print('processed windows:', {k: w[k].shape for k in w.files}, 'class counts', np.bincount(w['y']))
""")

md("## 4. The 27 trained models")
code("""
rows = []
for f in sorted(glob.glob(f'{R}/models/*.pt')):
    sd = torch.load(f, map_location='cpu', weights_only=False)
    sd = sd.get('state_dict', sd) if isinstance(sd, dict) else sd
    n = sum(v.numel() for v in sd.values() if hasattr(v, 'numel')) if isinstance(sd, dict) else None
    rows.append((os.path.basename(f), round(os.path.getsize(f) / 1e6, 1), n))
display(pd.DataFrame(rows, columns=['checkpoint', 'MB', 'stored parameters']))
""")

md("""
## 5. Re-test the trained models on the held-out test sets
These commands load the saved checkpoint, run it on the test split and rewrite `project/reports/<model>_metrics.json`.
""")
code("""
%cd {R}
for m in ['cnn', 'resnet18', 'effnet_cbam', 'resnet50_vit', 'densenet_se', 'mobilenet_attn', 'cnn_bilstm']:
    !python project/code/train_face.py --model {m} --eval-only 2>&1 | grep -v Warning | tail -3
""")
code("""
for ds in ['goemotions', 'emotion']:
    for m in ['roberta_bilstm_attn', 'bert_bilstm_attn', 'deberta_attn', 'distilbert_cnn', 'bert_cnn_bilstm']:
        !python project/code/train_text.py --model {m} --dataset {ds} --eval-only 2>&1 | grep -v -i warn | tail -2
""")
code("""
# Test results of all 27 models
rows = []
for f in sorted(glob.glob(f'{R}/project/reports/*_metrics.json')):
    m = json.load(open(f))
    if 'per_class' in m: rows.append((os.path.basename(f)[:-13], m.get('dataset'), m.get('test_size'), m['accuracy'], m['macro_f1']))
display(pd.DataFrame(rows, columns=['model', 'dataset', 'test samples', 'accuracy', 'macro_F1']))
""")

md("""
## 6. Re-run the faculty protocol on every model
`protocol.py`: for each feature CSV → 80:20 and 70:30 hold-out and 10-fold CV, each with and without SMOTE
(training data only), early stopping + L1/L2, over/underfitting check. `wesad_protocol.py`: the same for WESAD,
split **by subject**, retraining the full networks. Results go to `project/reports/protocol/`.
""")
code("""
display(pd.DataFrame({'feature CSV': sorted(os.listdir(f'{R}/project/features'))}))
display(pd.read_csv(f'{R}/project/features/face_densenet_se.csv').head())
""")
code("""
# Face, speech and text: all 22 feature CSVs (about 1-2 hours on a T4)
!python project/code/protocol.py project/features/face_*.csv project/features/audio_*.csv project/features/text_*.csv 2>&1 | tail -40
""")
code("""
# WESAD: 5 networks x (4 hold-out + 2 x 10 folds), subject-wise (several hours; resumable, re-run the cell to continue)
!python project/code/wesad_protocol.py --budget 10000 2>&1 | tail -40
""")
code("""
!python project/code/protocol.py --summarize 2>&1 | tail -60
display(pd.read_csv(f'{R}/project/reports/protocol/summary.csv'))
""")

md("""
## 7. Retrain any model from scratch (same data, same split, same code)
Uncomment the line you want. Times are for a T4 GPU.
""")
code("""
# Face (FER-2013, 90% train / 10% validation, test = official test set), ~30-60 min per hybrid
# !python project/code/train_face.py --model densenet_se

# Speech (70/15/15 stratified). First build the 16 kHz cache, then any model, ~5-20 min
# !python project/code/train_audio.py --prepare
# !python project/code/train_audio.py --model hubert_transformer

# Text (official splits), ~30-60 min
# !python project/code/train_text.py --model roberta_bilstm_attn --dataset goemotions

# WESAD (subject-wise: test S2, S9, S16; validation S5, S13), ~2 min
# !python project/code/train_wesad.py --prepare      # needs the raw WESAD part
# !python project/code/train_wesad.py --model cnn_gru

# Export the deep-feature CSV of a trained model (input of the protocol)
# !python project/code/extract_features.py --modality face --model densenet_se --device cuda
""")

md("## 8. Reports, figures and logs")
code("""
for f in sorted(glob.glob(f'{R}/project/reports/**/*.docx', recursive=True)): print(f)
for f in sorted(glob.glob(f'{R}/project/reports/final/fig/*.png'))[:30]: print(os.path.basename(f)); display(IPImage(filename=f, width=850))
""")
code("""
print(open(f'{R}/project/logs/face_densenet_se.log', encoding='utf-8', errors='ignore').read()[-5000:])
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}, "colab": {"provenance": []},
                                   "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 0}
NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("notebook:", NB)
