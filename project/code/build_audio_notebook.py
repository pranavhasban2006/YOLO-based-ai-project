import nbformat as nbf
from pathlib import Path

C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("# Speech emotion training (Colab)\nRuns top to bottom with Runtime > Run all. Resumes after a disconnect and skips finished runs.\n"
   "Runs: **E2** combined training (test reported per dataset), then **E3** leave-one-dataset-out.")

code('''import os, sys, json, time, zipfile, random
TINY = os.environ.get("TINY") == "1"          # local smoke test only
import torch
if not TINY:
    assert torch.cuda.is_available(), "No GPU. Runtime > Change runtime type > T4 GPU, then Run all again."
    print("GPU:", torch.cuda.get_device_name(0))
if TINY:
    DRIVE_IN, DRIVE_OUT, DATA = os.environ["T_IN"], os.environ["T_OUT"], os.environ["T_DATA"]
else:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_IN  = "/content/drive/MyDrive/emotion_audio/colab_upload"
    DRIVE_OUT = "/content/drive/MyDrive/emotion_audio/colab_results"
    DATA = "/content/audio_16k"
os.makedirs(DRIVE_OUT, exist_ok=True)
if not os.path.exists(os.path.join(DATA, "index.csv")):
    with zipfile.ZipFile(os.path.join(DRIVE_IN, "audio_16k.zip")) as z: z.extractall(DATA)
print("data ready:", DATA)''')

code('''import numpy as np, pandas as pd, soundfile as sf, torch.nn as nn, torchaudio
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_recall_fscore_support
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
CLASSES = ["angry","disgust","fear","happy","neutral","sad","surprise"]
CID = {c:i for i,c in enumerate(CLASSES)}
SR, SEC = 16000, 4.0
N = int(SR*SEC)
SEED = 42
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def seed_all(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all()
json.dump(CLASSES, open(f"{DRIVE_OUT}/classes.json","w"))

df = pd.read_csv(f"{DATA}/index.csv")
if TINY: df = df.groupby(["dataset","split"]).head(12).reset_index(drop=True)
df["y"] = df.label.map(CID)
Y_ALL = df.y.values
def load(p):
    w,_ = sf.read(f"{DATA}/{p}", dtype="float32")
    if w.ndim>1: w = w[:,0]
    w = w[:N]
    return np.pad(w,(0,N-len(w))).astype(np.float16)          # pad with zeros to 4 s
t=time.time(); WAV = np.stack([load(p) for p in df.path]); print("loaded", WAV.shape, f"{time.time()-t:.0f}s")
DATASETS = sorted(df.dataset.unique()); print(df.groupby(["dataset","split"]).size().unstack())''')

code('''class DS(torch.utils.data.Dataset):
    def __init__(s, idx, train): s.idx, s.train = idx, train
    def __len__(s): return len(s.idx)
    def __getitem__(s, i):
        j = s.idx[i]; w = torch.from_numpy(WAV[j].astype(np.float32))
        if s.train:
            w = torch.roll(w, random.randint(-SR//4, SR//4))
            w = w*random.uniform(0.6,1.4) + torch.randn_like(w)*random.uniform(0,0.005)
        return w, int(Y_ALL[j])

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=512, hop_length=160, n_mels=64)
        s.tm = torchaudio.transforms.TimeMasking(30); s.fm = torchaudio.transforms.FrequencyMasking(10)
        ch=[1,32,64,128,256]; L=[]
        for a,b in zip(ch,ch[1:]): L += [nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(),nn.Conv2d(b,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(),nn.MaxPool2d(2)]
        s.cnn = nn.Sequential(*L); s.drop = nn.Dropout(0.4); s.fc = nn.Linear(256,7)
    def forward(s, w):
        with torch.autocast(device_type=w.device.type, enabled=False):
            x = torch.log(s.mel(w.float())+1e-6)
            x = (x - x.mean((1,2),keepdim=True))/(x.std((1,2),keepdim=True)+1e-5)
        x = x.unsqueeze(1)
        if s.training: x = s.fm(s.tm(x))
        x = s.cnn(x).mean((2,3))
        return s.fc(s.drop(x))

def make_sampler(idx):
    # each dataset contributes at most 50% of an epoch
    d = df.dataset.values[idx]
    share = {k:(d==k).sum() for k in np.unique(d)}; tot = sum(share.values())
    frac = {k:min(share[k]/tot, 0.5) for k in share}; z = sum(frac.values())
    w = np.array([(frac[k]/z)/share[k] for k in d])
    return torch.utils.data.WeightedRandomSampler(w, len(idx), replacement=True)

@torch.no_grad()
def predict(model, idx):
    model.eval(); dl = torch.utils.data.DataLoader(DS(idx,False), batch_size=128, num_workers=0)
    P=[];Y=[]
    for w,y in dl:
        with torch.autocast(device_type=dev.type, enabled=dev.type=="cuda"):
            P.append(model(w.to(dev)).argmax(1).cpu())
        Y.append(y)
    return torch.cat(Y).numpy(), torch.cat(P).numpy()

def metrics(y, p):
    labels = list(range(7))
    pr,rc,f,_ = precision_recall_fscore_support(y,p,labels=labels,zero_division=0)
    return {"n":int(len(y)),"accuracy":float(accuracy_score(y,p)),
            "macro_f1_present_classes":float(f1_score(y,p,labels=sorted(set(y)),average="macro",zero_division=0)),
            "per_class":{CLASSES[i]:{"precision":float(pr[i]),"recall":float(rc[i]),"f1":float(f[i])} for i in labels}}

def cm_png(y,p,path,title):
    cm = confusion_matrix(y,p,labels=range(7))
    fig,ax=plt.subplots(figsize=(5.5,5)); ax.imshow(cm,cmap="Blues"); ax.set_title(title)
    ax.set_xticks(range(7)); ax.set_xticklabels(CLASSES,rotation=45); ax.set_yticks(range(7)); ax.set_yticklabels(CLASSES)
    for i in range(7):
        for j in range(7): ax.text(j,i,cm[i,j],ha="center",va="center",fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); fig.tight_layout(); fig.savefig(path,dpi=120); plt.close(fig)''')

code('''EPOCHS  = 2 if TINY else 40
WARMUP  = 1 if TINY else 2
PATIENCE = 6
BS = 16 if TINY else 64

def train_run(name, train_idx, val_idx, test_sets, epochs=EPOCHS):
    """test_sets: {label: idx}. Resumable via last.pt, skipped when done.json exists."""
    out = f"{DRIVE_OUT}/{name}"; os.makedirs(out, exist_ok=True)
    if os.path.exists(f"{out}/done.json"):
        print("skip (done):", name); return json.load(open(f"{out}/metrics.json"))
    seed_all()
    model = Net().to(dev)
    cnt = np.bincount(Y_ALL[train_idx], minlength=7).astype(float)
    cw = torch.tensor(np.where(cnt>0, cnt.sum()/(7*np.maximum(cnt,1)), 0), dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: (e+1)/WARMUP if e<WARMUP else 0.5*(1+np.cos(np.pi*(e-WARMUP)/max(1,epochs-WARMUP))))
    scaler = torch.amp.GradScaler(enabled=dev.type=="cuda")
    ep0, best, bad, hist = 0, -1, 0, []
    if os.path.exists(f"{out}/last.pt"):
        ck = torch.load(f"{out}/last.pt", map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        ep0, best, bad, hist = ck["ep"]+1, ck["best"], ck["bad"], ck["hist"]; print("resumed", name, "at epoch", ep0)
    dl = torch.utils.data.DataLoader(DS(train_idx,True), batch_size=BS, sampler=make_sampler(train_idx), num_workers=0, drop_last=len(train_idx)>BS)
    for ep in range(ep0, epochs):
        model.train(); tl=0; t=time.time()
        for w,y in dl:
            w,y = w.to(dev),y.to(dev)
            with torch.autocast(device_type=dev.type, enabled=dev.type=="cuda"):
                loss = crit(model(w), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); tl+=loss.item()
        sched.step()
        yv,pv = predict(model, val_idx); vf = f1_score(yv,pv,labels=sorted(set(yv)),average="macro",zero_division=0)
        hist.append({"epoch":ep,"loss":tl/len(dl),"val_macro_f1":vf,"val_acc":accuracy_score(yv,pv)})
        print(f"{name} ep{ep} loss {tl/len(dl):.3f} val_f1 {vf:.3f} ({time.time()-t:.0f}s)")
        if vf>best: best,bad=vf,0; torch.save(model.state_dict(), f"{out}/best.pt")
        else: bad+=1
        torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"sched":sched.state_dict(),"ep":ep,"best":best,"bad":bad,"hist":hist}, f"{out}/last.pt")
        if bad>=PATIENCE: print("early stop"); break
    pd.DataFrame(hist).to_csv(f"{out}/history.csv", index=False)
    model.load_state_dict(torch.load(f"{out}/best.pt", map_location=dev))
    res = {"run":name,"best_val_macro_f1":best,"tests":{}}
    for k,idx in test_sets.items():
        y,p = predict(model, idx); res["tests"][k] = metrics(y,p); cm_png(y,p,f"{out}/cm_{k}.png",f"{name} on {k}")
    json.dump(res, open(f"{out}/metrics.json","w"), indent=1); json.dump({"done":True}, open(f"{out}/done.json","w"))
    return res''')

md("## E2: train on all datasets together, test each dataset separately")
code('''ix = lambda ds,sp: np.where(df.dataset.isin(ds) & (df.split==sp))[0]
ALL = list(DATASETS)
tests = {d: ix([d],"test") for d in ALL}; tests["ALL"] = ix(ALL,"test")
e2 = train_run("E2_combined", ix(ALL,"train"), ix(ALL,"val"), tests)
rows=[{"experiment":"E2","test_set":k,"accuracy":v["accuracy"],"macro_f1":v["macro_f1_present_classes"],"n":v["n"]} for k,v in e2["tests"].items()]
print(pd.DataFrame(rows))''')

md("## E3: leave-one-dataset-out (train on the others, test on the whole held-out dataset)")
code('''for d in ALL:
    rest = [x for x in ALL if x!=d]
    r = train_run(f"E3_holdout_{d}", ix(rest,"train"), ix(rest,"val"), {d: np.where(df.dataset==d)[0]})
    v = r["tests"][d]
    rows.append({"experiment":"E3","test_set":d,"accuracy":v["accuracy"],"macro_f1":v["macro_f1_present_classes"],"n":v["n"]})
summ = pd.DataFrame(rows); summ.to_csv(f"{DRIVE_OUT}/summary.csv", index=False); print(summ)
fig,ax=plt.subplots(figsize=(7,4)); summ[summ.test_set!="ALL"].pivot(index="test_set",columns="experiment",values="macro_f1").plot.bar(ax=ax)
ax.set_ylabel("macro-F1 (classes present)"); fig.tight_layout(); fig.savefig(f"{DRIVE_OUT}/summary.png",dpi=120)
import shutil
for d in os.listdir(DRIVE_OUT):
    p=f"{DRIVE_OUT}/{d}"
    if os.path.isdir(p) and os.path.exists(f"{p}/metrics.json"):
        shutil.copy(f"{p}/metrics.json", f"{DRIVE_OUT}/{d}.json")
        shutil.copy(f"{p}/best.pt", f"{DRIVE_OUT}/{d}_best.pt")
        for f in os.listdir(p):
            if f.startswith("cm_"): shutil.copy(f"{p}/{f}", f"{DRIVE_OUT}/cm_{d}_{f[3:]}")
shutil.copy(f"{DRIVE_OUT}/E2_combined/best.pt", f"{DRIVE_OUT}/best_model.pt")
print("ALL DONE"); print(sorted(os.listdir(DRIVE_OUT)))''')

nb = nbf.v4.new_notebook(); nb.cells = C
nb.metadata["accelerator"] = "GPU"
out = Path(__file__).resolve().parents[1] / "colab_upload_audio" / "audio_colab_train.ipynb"
nbf.write(nb, out); print(out)
