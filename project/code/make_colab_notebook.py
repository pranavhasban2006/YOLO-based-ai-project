r"""Builds the Google Colab notebook + the zip to upload to Google Drive.

Output (repo root):
  Emotion_Recognition_Colab.ipynb   the notebook (open it in Colab)
  Emotion_Project_Colab.zip         Faculty_Files/ + project/code/*.py  -> upload to Google Drive (My Drive)

Usage (repo root): project\.venv-train\Scripts\python project\code\make_colab_notebook.py
"""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAC = next(p for p in (ROOT / "CSV AND IMAGE", ROOT / "Faculty_Files") if p.exists())   # user renamed it
ZIP = ROOT / "Emotion_Project_Colab.zip"
NB = ROOT / "Emotion_Recognition_Colab.ipynb"

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n")})


md("""
# Multimodal Emotion Recognition using Hybrid Deep Learning Models
**Team:** Prakul Jain (24BIT0548) · Pranav Hasban (24BIT0537) · Shreya Mehrotra (24BIT0547)

This notebook shows the complete project and **runs the evaluation protocol live**:

1. Load the datasets (CSV files) and look at sample data and images
2. Check class imbalance and the balanced (SMOTE) sizes
3. Load the **feature CSV** of each hybrid model
4. Data preprocessing → **80:20 and 70:30 split, with and without SMOTE** → **10-fold cross-validation, with and without SMOTE**
5. Train with **early stopping + L1/L2 regularisation**, check over/underfitting
6. Test: accuracy, precision, recall, F1, confusion matrix, inference time
7. Select the best model for every modality

The 27 deep networks (5 hybrids per modality) were trained on a local RTX 3050 GPU; their learned features were exported
to CSV. The training code of every model is included at the end of the notebook.

**Run:** `Runtime → Run all` (GPU runtime optional; everything also runs on CPU).
""")

md("## 0. Load the project files from Google Drive")
code("""
from google.colab import drive
drive.mount('/content/drive')

import os, zipfile
ZIP = '/content/drive/MyDrive/Emotion_Project_Colab.zip'   # <- change if you put the zip elsewhere
assert os.path.exists(ZIP), f'Upload Emotion_Project_Colab.zip to your Google Drive (My Drive) first: {ZIP}'
zipfile.ZipFile(ZIP).extractall('/content/project')
BASE = '/content/project/Faculty_Files'
print(open(f'{BASE}/README.txt').read())
""")
code("""
!pip -q install imbalanced-learn
import glob, json, time, numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
from torch import nn
from PIL import Image
from IPython.display import display, Image as IPImage
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
pd.set_option('display.max_colwidth', 120)
CSV = f'{BASE}/1_CSV_files'; IMG = f'{BASE}/2_Image_files'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('device:', DEVICE)
""")

md("""
## 1. Datasets (CSV files) and sample data
### 1.1 Face – FER-2013 (CSV layout: `emotion, pixels, Usage`)
""")
code("""
fer = pd.read_csv(f'{CSV}/a_dataset_csv/fer2013_sample_2_per_class.csv')
display(fer[['emotion', 'emotion_name', 'Usage', 'pixels']].head(14))
tr = fer[fer.Usage == 'Training'].reset_index(drop=True)
fig, axs = plt.subplots(2, 7, figsize=(14, 4.4))
for i, r in tr.iterrows():
    ax = axs[i % 2, i // 2]
    ax.imshow(np.array(r.pixels.split(), dtype=np.uint8).reshape(48, 48), cmap='gray'); ax.axis('off')
    ax.set_title(r.emotion_name)
plt.suptitle('FER-2013: images rebuilt from the CSV pixel column (2 per class)'); plt.show()
counts = pd.read_csv(f'{CSV}/a_dataset_csv/fer2013_class_counts.csv').pivot(index='emotion', columns='split', values='images')
display(counts)
""")
md("### 1.2 Sample images (2 per class): FER-2013, CK+, RAVDESS video frames")
code("""
for ds in ['FER-2013', 'CK+', 'RAVDESS_video_frames']:
    classes = sorted(os.listdir(f'{IMG}/a_dataset_samples/{ds}'))
    fig, axs = plt.subplots(2, len(classes), figsize=(1.9 * len(classes), 4.2))
    for j, c in enumerate(classes):
        files = sorted(glob.glob(f'{IMG}/a_dataset_samples/{ds}/{c}/*'))[:2]
        for i, f in enumerate(files):
            im = Image.open(f); axs[i, j].imshow(im, cmap='gray' if im.mode == 'L' else None); axs[i, j].axis('off')
        axs[0, j].set_title(c)
    plt.suptitle(ds); plt.show()
""")
md("### 1.3 Text datasets (CSV)")
code("""
for name in ['goemotions_ekman7', 'dair_ai_emotion', 'tweeteval_emotion']:
    df = pd.read_csv(f'{CSV}/a_dataset_csv/{name}_train.csv')
    print(f'{name}: {len(df):,} training rows'); display(df.groupby('emotion').head(2).sort_values('emotion'))
""")
md("### 1.4 Speech datasets (file list) and WESAD physiological CSV")
code("""
sp = pd.read_csv(f'{CSV}/a_dataset_csv/speech_ravdess_tess_savee_files.csv')
display(sp.groupby('emotion').head(2).sort_values('emotion'))
display(pd.crosstab(sp.emotion, sp.dataset, margins=True))
we = pd.read_csv(f'{CSV}/a_dataset_csv/wesad_window_features_32.csv')
print(f'WESAD: {len(we):,} windows x {we.shape[1] - 3} features, {we.subject.nunique()} subjects')
display(we.groupby('state').head(2))
""")

md("## 2. Class imbalance: imbalanced vs balanced (SMOTE) size")
code("""
def balance_report(name, labels):
    c = pd.Series(labels).value_counts().sort_index()
    t = pd.DataFrame({'imbalanced': c, 'balanced_SMOTE': c.max(), 'synthetic_added': c.max() - c})
    t.loc['Total'] = t.sum()
    ratio = c.max() / c.min()
    print(f'{name}: imbalance ratio {ratio:.1f} -> {"IMBALANCED" if ratio > 1.5 else "approximately balanced"}')
    display(t)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3), sharey=True)
    axs[0].bar(c.index.astype(str), c.values, color='#d9534f'); axs[0].set_title(f'Imbalanced ({c.sum():,})')
    axs[1].bar(c.index.astype(str), c.max(), color='#5cb85c'); axs[1].set_title(f'Balanced after SMOTE ({c.max() * len(c):,})')
    for a in axs: a.tick_params(axis='x', rotation=35)
    plt.suptitle(name); plt.show()

fc = pd.read_csv(f'{CSV}/a_dataset_csv/fer2013_class_counts.csv'); fc = fc[fc.split == 'train']
balance_report('FER-2013 (train)', np.repeat(fc.emotion.values, fc.images.values))
balance_report('Speech (RAVDESS+TESS+SAVEE)', sp.emotion)
balance_report('GoEmotions (train)', pd.read_csv(f'{CSV}/a_dataset_csv/goemotions_ekman7_train.csv').emotion)
balance_report('dair-ai Emotion (train)', pd.read_csv(f'{CSV}/a_dataset_csv/dair_ai_emotion_train.csv').emotion)
balance_report('WESAD (windows)', we.state)
""")

md("""
## 3. Stage 1 – Training the 27 deep hybrid models: dataset, split, settings
All models were trained on a local NVIDIA RTX 3050 (4 GB) with PyTorch. Each modality uses **one fixed split (seed 0)
shared by all of its models**, so the comparison is fair.
""")
code("""
P = '/content/project'
split_table = pd.DataFrame([
 ['Face', 'FER-2013', '5 hybrids + 2 baselines', 'official train/test; train -> 90% train / 10% validation (seed 0)', '25,839 / 2,870 / 7,178', 'train_face.py: n_val = len(idx) // 10', '15 (CNN 40)', 'AdamW, L2 weight_decay 1e-4, OneCycle LR, label smoothing 0.1', 'best-validation checkpoint'],
 ['Speech', 'RAVDESS + TESS + SAVEE', '5 hybrids', 'stratified by dataset+emotion: 70% / 15% / 15% (seed 0)', '5,129 / 1,099 / 1,100', 'train_audio.py: def splits()', '30', 'AdamW, L2 weight_decay 1e-2, OneCycle LR', 'best-validation checkpoint'],
 ['Text', 'GoEmotions (Ekman 7)', '5 hybrids', 'official train / validation / test', '39,555 / 4,946 / 4,968', 'train_text.py: load_split()', '4', 'AdamW, L2 weight_decay 1e-2', 'best-validation checkpoint'],
 ['Text', 'dair-ai Emotion (6)', '5 hybrids', 'official train / validation / test', '16,000 / 2,000 / 2,000', 'train_text.py: load_split()', '4', 'AdamW, L2 weight_decay 1e-2', 'best-validation checkpoint'],
 ['Physiological', 'WESAD chest, 60 s windows', '5 hybrids', 'BY SUBJECT: 10 train / 2 val (S5,S13) / 3 test (S2,S9,S16)', '4,098 / 820 / 1,206', 'train_wesad.py: SPLIT = {...}', '30', 'AdamW, L2 weight_decay 1e-3, OneCycle LR', 'best-validation checkpoint'],
], columns=['modality', 'dataset', 'models', 'split', 'train / val / test', 'where in code', 'epochs', 'optimiser + regularisation', 'early stopping'])
display(split_table)
""")
md("### 3.1 The exact split code in every script (with line numbers)")
code("""
import re
def show(file, pattern, n=12, before=0):
    lines = open(f'{P}/code/{file}', encoding='utf-8').read().splitlines()
    i = next(k for k, l in enumerate(lines) if re.search(pattern, l))
    print(f'---------- {file}  (line {i + 1}) ----------')
    for k in range(max(0, i - before), min(len(lines), i + n)): print(f'{k + 1:4d}  {lines[k]}')
    print()

show('train_face.py', r'train_full = datasets.ImageFolder', 12)       # FER-2013: 90% train / 10% validation
show('train_audio.py', r'^def splits', 6)                             # speech: 70 / 15 / 15 stratified
show('train_text.py', r'^def load_split', 14)                         # text: official splits
show('train_wesad.py', r'^SPLIT = ', 1)                               # WESAD: by subject
show('protocol.py', r'^def run', 12)                                  # protocol: 10% validation + SMOTE on training only
show('protocol.py', r'for test_size in', 30)                          # protocol: 80:20 / 70:30 / 10-fold
show('wesad_protocol.py', r'GroupShuffleSplit\\(1', 26, 2)             # WESAD protocol: by subject
""")
md("### 3.2 Optimiser, L2 (weight decay) and the training loop with early stopping (best-validation checkpoint)")
code("""
show('train_face.py', r'opt = torch.optim.AdamW', 30, 4)
show('train_audio.py', r'opt = torch.optim.AdamW', 26, 2)
show('train_text.py', r'opt = torch.optim.AdamW', 30, 2)
show('train_wesad.py', r'opt = torch.optim.AdamW', 28, 2)
""")
md("### 3.3 Training logs of every model (real output of the training runs) and learning curves")
code("""
LOGS = f'{P}/logs'
TRAIN_LOGS = sorted(f for f in glob.glob(f'{LOGS}/*.log') if os.path.basename(f).split('_')[0] in ('face', 'audio', 'text', 'wesad')
                    and not re.search(r'ensemble|prepare|_eval|protocol', os.path.basename(f)))
ep_re = re.compile(r'epoch (\\d+)/(\\d+).*?val_acc ([\\d.]+)(?: val_macroF1 ([\\d.]+))?')
curves, info = {}, []
for f in TRAIN_LOGS:
    name = os.path.basename(f)[:-4]; txt = open(f, encoding='utf-8', errors='ignore').read()
    sizes = re.search(r'(\\d+) train / (\\d+) val / (\\d+) test', txt)
    eps = [(int(a), float(c), float(d) if d else np.nan) for a, b, c, d in ep_re.findall(txt)]
    if not eps: continue
    curves[name] = eps
    best = max(eps, key=lambda e: e[1] if np.isnan(e[2]) else e[2])
    info.append(dict(model=name, train=sizes and int(sizes[1]), val=sizes and int(sizes[2]), test=sizes and int(sizes[3]),
                     epochs_run=len(eps), best_epoch_saved=best[0], best_val_acc=best[1], best_val_macroF1=best[2]))
display(pd.DataFrame(info))
""")
code("""
for mod in ['face', 'audio', 'text', 'wesad']:
    names = [n for n in curves if n.startswith(mod)]
    fig, axs = plt.subplots(1, 2, figsize=(14, 3.8))
    for n in names:
        e = np.array(curves[n]); axs[0].plot(e[:, 0], e[:, 1], marker='.', label=n); axs[1].plot(e[:, 0], e[:, 2], marker='.', label=n)
    axs[0].set_title(f'{mod}: validation accuracy per epoch'); axs[1].set_title(f'{mod}: validation macro-F1 per epoch')
    for a in axs: a.set_xlabel('epoch'); a.grid(alpha=.3)
    axs[1].legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left'); plt.show()
""")
code("""
# Full training log of any model (change the name: see the list above)
print(open(f'{LOGS}/face_densenet_se.log', encoding='utf-8', errors='ignore').read()[-6000:])
""")
md("### 3.4 Held-out test results of every model (per class + confusion matrix)")
code("""
for f in sorted(glob.glob(f'{P}/reports/*_metrics.json')):
    m = json.load(open(f))
    if 'per_class' not in m: continue
    name = os.path.basename(f)[:-13]
    print(f'\\n===== {name} | {m.get("dataset")} | test samples {m.get("test_size")} | accuracy {m["accuracy"]} | macro-F1 {m["macro_f1"]} | time {m["inference_time"]}')
    display(pd.DataFrame(m['per_class']).T.round(3))
    cm, labels = np.array(m['confusion_matrix']['rows_true_cols_pred']), m['confusion_matrix']['labels']
    n = cm / cm.sum(1, keepdims=True)
    plt.figure(figsize=(1 + 0.7 * len(labels), 0.7 * len(labels) + 0.5)); plt.imshow(n, cmap='Blues')
    for i in range(len(labels)):
        for j in range(len(labels)): plt.text(j, i, cm[i, j], ha='center', va='center', fontsize=7, color='white' if n[i, j] > .5 else 'k')
    plt.xticks(range(len(labels)), labels, rotation=35, fontsize=7); plt.yticks(range(len(labels)), labels, fontsize=7); plt.title(name); plt.show()
""")
md("""
### 3.5 Saved protocol results of ALL 27 models: every split and every one of the 10 folds
Output of `protocol.py` / `wesad_protocol.py` on the full feature CSVs: train/val accuracy, over/underfitting,
epochs until early stopping, training size after SMOTE, and for WESAD the test subjects of each split.
""")
code("""
SETTINGS = ['holdout_80_20_no_smote', 'holdout_80_20_smote', 'holdout_70_30_no_smote', 'holdout_70_30_smote', 'cv10_no_smote', 'cv10_smote']
KEEP = ['train_acc', 'val_acc', 'fit', 'epochs_until_early_stop', 'train_size_after_smote', 'test_subjects']
rows, folds = [], []
for f in sorted(glob.glob(f'{P}/reports/protocol/*.json')):
    d = json.load(open(f)); name = os.path.basename(f)[:-5]
    for s in SETTINGS:
        r = d.get(s)
        if r is None: continue
        if s.startswith('cv10'):
            rows.append(dict(model=name, setting=s, accuracy=r['accuracy_mean'], macro_f1=r['macro_f1_mean'], f1_std=r['macro_f1_std']))
            for i, fo in enumerate(r['folds'], 1):
                folds.append(dict(model=name, setting=s, fold=i, accuracy=fo['accuracy'], macro_f1=fo['macro_f1'], **{k: fo.get(k) for k in KEEP}))
        else:
            rows.append(dict(model=name, setting=s, accuracy=r['accuracy'], macro_f1=r['macro_f1'], **{k: r.get(k) for k in KEEP}))
allres, allfolds = pd.DataFrame(rows).round(4), pd.DataFrame(folds).round(4)
pd.set_option('display.max_rows', 500)
display(allres.pivot_table(index='model', columns='setting', values='macro_f1')[SETTINGS].round(3))
display(allres)
""")
code("""
# All 10 folds of a model (change the name)
display(allfolds[allfolds.model == 'face_densenet_se'])
display(allfolds[allfolds.model == 'wesad_cnn_gru'])
""")

md("""
## 4. The evaluation protocol (faculty flowchart) – re-run live in this notebook
`load CSV → preprocess → split (80:20 / 70:30, or 10-fold) → with / without SMOTE (training data only)
→ train (early stopping + L1 + L2) → over/underfitting check → test → best model`
""")
code("""
SEED, L1, L2, PATIENCE, MAX_EPOCHS = 0, 1e-5, 1e-3, 10, 200

def predict(model, x):
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float32, device=DEVICE)).argmax(1).cpu().numpy()

def train_mlp(xtr, ytr, xval, yval, k):
    torch.manual_seed(SEED)
    model = nn.Sequential(nn.Linear(xtr.shape[1], 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, k)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=L2)          # L2 regularisation
    ce = nn.CrossEntropyLoss()
    xt, yt = torch.tensor(xtr, dtype=torch.float32, device=DEVICE), torch.tensor(ytr, device=DEVICE)
    best, state, wait = -1, None, 0
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        for idx in torch.randperm(len(yt), device=DEVICE).split(256):
            loss = ce(model(xt[idx]), yt[idx]) + L1 * sum(p.abs().sum() for n, p in model.named_parameters() if 'weight' in n)
            opt.zero_grad(); loss.backward(); opt.step()
        f1 = f1_score(yval, predict(model, xval), average='macro')
        if f1 > best: best, state, wait = f1, {n: p.clone() for n, p in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE: break                                                # early stopping
    model.load_state_dict(state)
    return model, ep

def run(xtr, ytr, xte, yte, k, smote):
    xfit, xval, yfit, yval = train_test_split(xtr, ytr, test_size=0.1, stratify=ytr, random_state=SEED)
    sc = StandardScaler().fit(xfit)                                                    # preprocessing (fit on train only)
    xfit, xval, xte = sc.transform(xfit), sc.transform(xval), sc.transform(xte)
    x0, y0 = xfit, yfit
    if smote: xfit, yfit = SMOTE(random_state=SEED).fit_resample(xfit, yfit)          # SMOTE on training data only
    model, ep = train_mlp(xfit, yfit, xval, yval, k)
    t0 = time.perf_counter(); p = predict(model, xte); ms = (time.perf_counter() - t0) / len(xte) * 1000
    tr, va = accuracy_score(y0, predict(model, x0)), accuracy_score(yval, predict(model, xval))
    fit = 'overfitting' if tr - va > 0.10 else 'underfitting' if tr < 0.5 else 'good fit'
    return yte, p, dict(train_acc=tr, val_acc=va, fit=fit, epochs=ep, ms=ms)

def protocol(csv, groups_col=None, name=None):
    df = pd.read_csv(csv).replace([np.inf, -np.inf], np.nan).dropna()
    g = df.pop(groups_col).values if groups_col else None
    df = df.drop(columns=[c for c in df.columns if c in ('group', 'state')])
    x, y = df.drop(columns='label').values.astype(np.float32), df['label'].values
    k = int(y.max()) + 1; rows = []; last_cm = None
    for ts in (0.2, 0.3):
        if g is None: tr_i, te_i = train_test_split(np.arange(len(y)), test_size=ts, stratify=y, random_state=SEED)
        else: tr_i, te_i = next(GroupShuffleSplit(1, test_size=ts, random_state=SEED).split(x, y, g))
        for sm in (False, True):
            yt, p, info = run(x[tr_i], y[tr_i], x[te_i], y[te_i], k, sm)
            pr, rc, _, _ = precision_recall_fscore_support(yt, p, average='macro', zero_division=0)
            rows.append(dict(setting=f'{round((1-ts)*100)}:{round(ts*100)}' + (' + SMOTE' if sm else ''),
                             accuracy=accuracy_score(yt, p), precision=pr, recall=rc,
                             macro_f1=f1_score(yt, p, average='macro'), f1_std=np.nan, fit=info['fit'],
                             epochs=info['epochs'], ms_per_sample=info['ms']))
    for sm in (False, True):
        cv = StratifiedKFold(10, shuffle=True, random_state=SEED) if g is None else GroupKFold(10)
        accs, f1s, ys, ps = [], [], [], []
        for tr_i, te_i in cv.split(x, y, g):
            yt, p, _ = run(x[tr_i], y[tr_i], x[te_i], y[te_i], k, sm)
            accs.append(accuracy_score(yt, p)); f1s.append(f1_score(yt, p, average='macro')); ys += list(yt); ps += list(p)
        pr, rc, _, _ = precision_recall_fscore_support(ys, ps, average='macro', zero_division=0)
        rows.append(dict(setting='10-fold' + (' + SMOTE' if sm else ''), accuracy=np.mean(accs), precision=pr, recall=rc,
                         macro_f1=np.mean(f1s), f1_std=np.std(f1s), fit='-', epochs='-', ms_per_sample=np.nan))
        last_cm = confusion_matrix(ys, ps)
    res = pd.DataFrame(rows).round(4); res.insert(0, 'model', name or os.path.basename(csv)[:-4])
    return res, last_cm

def show_cm(cm, labels, title):
    n = cm / cm.sum(1, keepdims=True)
    plt.figure(figsize=(1 + 0.8 * len(labels), 0.8 * len(labels) + 0.6)); plt.imshow(n, cmap='Blues')
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='white' if n[i, j] > .5 else 'k', fontsize=8)
    plt.xticks(range(len(labels)), labels, rotation=35); plt.yticks(range(len(labels)), labels)
    plt.xlabel('predicted'); plt.ylabel('true'); plt.title(title); plt.show()
""")

md("""
## 5. Run the protocol on the feature CSV of each modality's best hybrid model
Each CSV has one row per held-out sample: `f0 … fN` = deep features learned by the hybrid model, `label` = class.
""")
code("""
display(pd.read_csv(f'{CSV}/b_feature_csv/PREVIEW_face_densenet_se.csv'))
""")
code("""
E7 = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
JOBS = [('face_densenet_se', 'Face – DenseNet121 + SE', E7),
        ('audio_hubert_transformer', 'Speech – HuBERT + Transformer', E7),
        ('text_roberta_bilstm_attn_goemotions', 'Text (GoEmotions) – RoBERTa + BiLSTM + Attn', E7),
        ('text_bert_cnn_bilstm_emotion', 'Text (dair-ai) – BERT + CNN + BiLSTM', ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'])]
all_results = []
for stem, title, labels in JOBS:
    t0 = time.time()
    res, cm = protocol(f'{CSV}/b_feature_csv/{stem}.csv', name=title)
    print(f'\\n=== {title}  ({time.time() - t0:.0f} s)'); display(res); all_results.append(res)
    show_cm(cm, labels, f'{title} – 10-fold + SMOTE (pooled)')
""")
md("""
### Physiological (WESAD): 32 statistical features per window, **split by subject**
(a person never appears in both training and test, so the score is for new people)
""")
code("""
res, cm = protocol(f'{CSV}/a_dataset_csv/wesad_window_features_32.csv', groups_col='subject', name='WESAD – statistical features + MLP')
display(res); all_results.append(res)
show_cm(cm, ['baseline', 'stress', 'amusement'], 'WESAD – 10-fold by subject + SMOTE')
""")
code("""
live = pd.concat(all_results, ignore_index=True)
display(live.pivot_table(index='model', columns='setting', values='macro_f1', sort=False).round(3))
""")

md("## 6. Comparison of all 27 trained models (results of the full training runs)")
code("""
summary = pd.read_csv(f'{CSV}/c_results_csv/protocol_summary_all_models.csv')
display(summary)
test = pd.read_csv(f'{CSV}/c_results_csv/test_results_all_models.csv')
display(test)
""")
code("""
for f in sorted(glob.glob(f'{IMG}/b_figures/final_results/*_compare.png')) + [f'{IMG}/b_figures/final_results/best_overall.png']:
    print(os.path.basename(f)); display(IPImage(filename=f, width=900))
""")
md("## 7. Best model per modality")
code("""
display(summary.loc[summary.best_in_modality == True, ['modality', 'model', 'cv10_smote_acc', 'cv10_smote_f1']].reset_index(drop=True))
""")

md("## 8. Methodology diagrams")
code("""
for f in ['flowchart_methodology', 'system_architecture', 'arch_face', 'arch_audio', 'arch_text', 'arch_physio']:
    display(IPImage(filename=f'{IMG}/b_figures/review2_dataset_and_methodology/{f}.png', width=900))
""")

md("""
## 9. Source code of the hybrid models
The complete training code (all 20 hybrid architectures, data preprocessing, training loops) is in `code/`.
""")
code("""
CODE = '/content/project/code'
print('\\n'.join(sorted(os.listdir(CODE))))
""")
code("""
# The five face hybrids: EfficientNetB3+CBAM, ResNet50+ViT, DenseNet121+SE, MobileNetV3+Attention, CNN+BiLSTM
src = open(f'{CODE}/train_face.py').read()
print(src[src.index('class CBAM'):src.index('HYBRID_BUILDERS')])
""")
code("""
# Build every face hybrid and count its parameters (downloads the ImageNet backbones)
import sys; sys.path.insert(0, CODE)
import train_face
for n, b in train_face.HYBRID_BUILDERS.items():
    m = b(); print(f'{n:16s} {sum(p.numel() for p in m.parameters()) / 1e6:6.1f} M parameters')
""")

code("""
# Print the COMPLETE source of any file (change the name). Files: see the list above.
print(open(f'{CODE}/train_face.py', encoding='utf-8').read())
""")
code("""
for f in ['train_audio.py', 'train_text.py', 'train_wesad.py', 'extract_features.py', 'protocol.py', 'wesad_protocol.py', 'face_ensemble.py']:
    print('=' * 30, f, '=' * 30); print(open(f'{CODE}/{f}', encoding='utf-8').read())
""")
md("""
## 10. The web application (FastAPI backend + React frontend)
Face (webcam/photo/video), speech, text and EEG tabs; the trained models are served by the backend and the results are fused.
""")
code("""
APP = f'{P}/app'
for root, _, files in os.walk(APP):
    for fn in sorted(files): print(os.path.relpath(os.path.join(root, fn), APP))
""")
code("""
for f in ['backend/app/main.py', 'backend/app/routes/predict.py', 'backend/app/face/pipeline.py', 'backend/app/modalities/audio.py',
          'backend/app/modalities/text.py', 'backend/app/fusion.py']:
    print('=' * 30, f, '=' * 30); print(open(f'{APP}/{f}', encoding='utf-8').read())
""")
code("""
for f in sorted(glob.glob(f'{P}/reports/ui_screenshots/*.png')): print(os.path.basename(f)); display(IPImage(filename=f, width=900))
""")
md("## 11. Audio test files (held-out clips) and the reports")
code("""
from IPython.display import Audio, Video
for f in sorted(glob.glob(f'{P}/test_samples/audio_mp4/*.mp4'))[:6]:
    print(os.path.basename(f)); display(Video(f, embed=True, width=300))
print('Reports (download from the Files panel on the left):')
for f in sorted(glob.glob(f'{P}/reports/**/*.docx', recursive=True)): print('  ', f)
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}, "colab": {"provenance": []},
                                   "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 0}
NB.write_text(json.dumps(nb, indent=1), encoding="utf-8")

with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for f in FAC.rglob("*"):
        if f.is_file():
            z.write(f, Path("Faculty_Files") / f.relative_to(FAC))      # fixed name inside the zip
    z.write(NB, NB.name)
    for f in (ROOT / "project" / "code").glob("*.py"):
        z.write(f, Path("code") / f.name)
    P = ROOT / "project"
    extra = [*P.glob("logs/*.log"), *P.glob("reports/*_metrics.json"), *P.glob("reports/protocol/*"),
             *P.glob("reports/ui_screenshots/*.png"), P / "reports/final/Final_Results_Multimodal_Emotion.docx",
             P / "reports/review2/Review2_Multimodal_Emotion_v2.docx"]
    for f in extra:
        if f.is_file():
            z.write(f, f.relative_to(P))
    for f in (ROOT / "test_samples").rglob("*"):
        if f.is_file():
            z.write(f, f.relative_to(ROOT))
    for sub in ("backend/app", "backend/tests", "frontend/src"):
        for f in (ROOT / sub).rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts:
                z.write(f, Path("app") / f.relative_to(ROOT))
    for f in ("README.md", "frontend/package.json", "backend/requirements.txt"):
        if (ROOT / f).is_file():
            z.write(ROOT / f, Path("app") / f)
print(f"notebook: {NB}\nzip: {ZIP} ({ZIP.stat().st_size / 1e6:.0f} MB)")
