"""Shared definitions for the speech emotion pipeline (audio phases 0-2).

Classes are the same seven as the face model, in the same order (project/models/classes.json).
Every parser returns (speaker, label) or None when a clip must be excluded, with the reason.
"""
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]          # .../project
REPO = PROJECT.parent                                  # .../YOLO
DATASETS = REPO / "datasets"
DATA = PROJECT / "data"
LOGS = PROJECT / "logs"
STATE = PROJECT / "audio_state.json"
INDEX = DATA / "audio_index.csv"
CLASSES = json.loads((PROJECT / "models" / "classes.json").read_text(encoding="utf-8"))
SEED = 42
MAX_SECONDS = 4.0
SR = 16000

# name -> (dataset folder candidates, wav folder under it, expected speech clip count)
DATASET_SPECS = {
    "ravdess": (["RAVDESS", "ravdess", "ravdess_speech"], "Audio_Speech", 1440),
    "crema_d": (["crema_d", "crema-d", "cremad"], "AudioWAV", 7442),
    "tess": (["tess", "TESS"], ".", 2800),
    "emodb": (["emodb", "emo-db", "emo_db"], "wav", 535),
    "savee": (["savee", "SAVEE"], "wav", 480),
    "esd": (["esd", "ESD"], "English", 17500),
    "meld": (["meld", "MELD"], None, None),   # wavs are extracted by audio_meld_extract.py into MELD_WAV
}
MELD_WAV = DATA / "meld_wav"
MELD_DIRS = {"train": "train_splits", "dev": "dev_splits_complete", "test": "output_repeated_splits_test"}

RAVDESS_MAP = {"01": "neutral", "02": "neutral", "03": "happy", "04": "sad",
               "05": "angry", "06": "fear", "07": "disgust", "08": "surprise"}
CREMA_MAP = {"ANG": "angry", "DIS": "disgust", "FEA": "fear", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
TESS_MAP = {"angry": "angry", "disgust": "disgust", "fear": "fear", "happy": "happy",
            "neutral": "neutral", "sad": "sad", "ps": "surprise"}
# EMO-DB letter codes (German): W Wut, L Langeweile, E Ekel, A Angst, F Freude, T Trauer, N neutral.
# L (boredom) has no class in the seven-class set and is excluded (see DECISIONS.md).
EMODB_MAP = {"W": "angry", "E": "disgust", "A": "fear", "F": "happy", "T": "sad", "N": "neutral", "L": None}


def parse_ravdess(stem):
    p = stem.split("-")
    if len(p) != 7:
        return None, "bad_name"
    if p[1] != "01":  # vocal channel: 01 = speech, 02 = song
        return None, "not_speech"
    label = RAVDESS_MAP.get(p[2])
    if label is None:
        return None, "unknown_emotion"
    return p[6], label


def parse_crema(stem):
    p = stem.split("_")
    if len(p) != 4:
        return None, "bad_name"
    label = CREMA_MAP.get(p[2])
    if label is None:
        return None, "unknown_emotion"
    return p[0], label


def parse_tess(stem):
    p = stem.split("_")
    if len(p) < 3:
        return None, "bad_name"
    label = TESS_MAP.get(p[-1])
    if label is None:
        return None, "unknown_emotion"
    return p[0], label


def tess_word(stem):
    p = stem.split("_")
    return "_".join(p[1:-1]) if len(p) >= 3 else stem


def parse_emodb(stem):
    if len(stem) != 7:
        return None, "bad_name"
    if stem[5] not in EMODB_MAP:
        return None, "unknown_emotion"
    label = EMODB_MAP[stem[5]]
    if label is None:
        return None, "boredom_excluded"
    return stem[:2], label


SAVEE_MAP = {"a": "angry", "d": "disgust", "f": "fear", "h": "happy", "n": "neutral", "sa": "sad", "su": "surprise"}


def parse_savee(stem):
    p = stem.split("_")                       # DC_a01 -> speaker DC, code a
    if len(p) != 2 or len(p[1]) < 3:
        return None, "bad_name"
    label = SAVEE_MAP.get(p[1][:-2])
    return (p[0], label) if label else (None, "unknown_emotion")


# ESD English: per speaker 1750 clips, utterance blocks of 350 in the order below.
# Verified against the Neutral/Angry/Happy/Sad/Surprise folders (0011_000351 is in Angry, 0011_001401 in Surprise).
ESD_ORDER = ["neutral", "angry", "happy", "sad", "surprise"]


def parse_esd(stem):
    p = stem.split("_")
    if len(p) != 2 or not p[1].isdigit():
        return None, "bad_name"
    k = (int(p[1]) - 1) // 350
    if not 0 <= k < 5:
        return None, "unknown_emotion"
    return p[0], ESD_ORDER[k]


MELD_EMO = {"anger": "angry", "disgust": "disgust", "fear": "fear", "joy": "happy",
            "neutral": "neutral", "sadness": "sad", "surprise": "surprise"}
_MELD = {}


def meld_table():
    """{(split, dialogue, utterance): (speaker, label)} from the three MELD csv files."""
    if not _MELD:
        import pandas as pd
        base = find_dataset_dir("meld") / "MELD.Raw"
        for split, fn in (("train", "train_sent_emo.csv"), ("dev", "dev_sent_emo.csv"), ("test", "test_sent_emo.csv")):
            df = pd.read_csv(base / fn)
            for r in df.itertuples():
                _MELD[(split, int(r.Dialogue_ID), int(r.Utterance_ID))] = (str(r.Speaker), MELD_EMO.get(r.Emotion))
    return _MELD


def meld_split(stem):
    return stem.split("_")[0]


def parse_meld(stem):
    p = stem.split("_")                       # train_dia0_utt3
    try:
        key = (p[0], int(p[1][3:]), int(p[2][3:]))
    except (IndexError, ValueError):
        return None, "bad_name"
    row = meld_table().get(key)
    if row is None or row[1] is None:
        return None, "unknown_emotion"
    return row


PARSERS = {"ravdess": parse_ravdess, "crema_d": parse_crema, "tess": parse_tess, "emodb": parse_emodb,
           "savee": parse_savee, "esd": parse_esd, "meld": parse_meld}


def find_dataset_dir(name):
    """Return the existing dataset folder for a canonical name, or None."""
    for cand in DATASET_SPECS[name][0]:
        p = DATASETS / cand
        if p.is_dir():
            return p
    return None


def wav_root(name):
    if name == "meld":
        return MELD_WAV
    base = find_dataset_dir(name)
    if base is None:
        return None
    sub = DATASET_SPECS[name][1]
    return base if sub == "." else base / sub


def list_wavs(name):
    root = wav_root(name)
    if root is None or not root.is_dir():
        return []
    # Top level only for TESS and EMO-DB; the RAVDESS and CREMA-D roots contain video and CSVs we skip.
    if name in ("tess", "emodb", "savee"):
        return sorted(root.glob("*.wav"))
    if name == "meld":
        return sorted(root.glob("*.wav"))
    return sorted(root.rglob("*.wav"))
