"""Text emotion modality: RoBERTa + BiLSTM + attention, fine-tuned on GoEmotions (Ekman 7 classes).

The architecture must stay identical to project/code/train_text.py (TextModel with BiLSTMAttnHead),
otherwise the checkpoint will not load. GoEmotions-Ekman uses the same class order as config.CLASSES.
Until the checkpoint loads (file missing, transformers not installed, tokenizer not cached) the
predictor behaves like the old mock and reports mock=True, so the app still runs.
"""
import logging
import os
import threading

import torch
from torch import nn

from .. import config
from ..schemas import Prediction
from .base import MockPredictor

log = logging.getLogger(__name__)

ENCODER = "roberta-base"
MAX_LEN = 64
CHECKPOINT = config.MODELS_DIR / os.getenv("TEXT_CHECKPOINT", "text_roberta_bilstm_attn_goemotions.pt")


class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.score = nn.Linear(d, 1)

    def forward(self, s, mask):
        a = self.score(s).squeeze(-1).masked_fill(mask == 0, -1e4)
        return (s * torch.softmax(a, 1).unsqueeze(-1)).sum(1)


class BiLSTMAttnHead(nn.Module):
    def __init__(self, h, n, k=256):
        super().__init__()
        self.lstm = nn.LSTM(h, k, batch_first=True, bidirectional=True)
        self.pool, self.out = AttnPool(2 * k), nn.Linear(2 * k, n)

    def forward(self, s, mask):
        return self.out(self.pool(self.lstm(s)[0], mask))


class TextModel(nn.Module):
    def __init__(self, encoder, n):
        super().__init__()
        self.encoder = encoder
        self.drop = nn.Dropout(0.2)
        self.head = BiLSTMAttnHead(encoder.config.hidden_size, n)

    def forward(self, input_ids, attention_mask):
        s = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(self.drop(s), attention_mask)


class TextPredictor(MockPredictor):
    """Real model once `load` succeeds; until then identical to the mock (mock=True)."""

    def __init__(self):
        super().__init__("text")
        self.model = None
        self.tokenizer = None
        self._lock = threading.Lock()

    @property
    def info(self) -> dict:
        return {"modality": "text", "mock": self.is_mock,
                "model": None if self.is_mock else "RoBERTa + BiLSTM + Attention (GoEmotions)"}

    def load(self) -> None:
        if not CHECKPOINT.exists():
            log.warning("Text checkpoint missing at %s; text stays mocked", CHECKPOINT)
            return
        try:
            from transformers import AutoConfig, AutoModel, AutoTokenizer
            try:                                   # prefer the local Hugging Face cache (no network at startup)
                self.tokenizer = AutoTokenizer.from_pretrained(ENCODER, local_files_only=True)
                enc_cfg = AutoConfig.from_pretrained(ENCODER, local_files_only=True)
            except OSError:
                self.tokenizer = AutoTokenizer.from_pretrained(ENCODER)
                enc_cfg = AutoConfig.from_pretrained(ENCODER)
            # Build the encoder from its config; every weight comes from the fine-tuned checkpoint.
            model = TextModel(AutoModel.from_config(enc_cfg), len(config.CLASSES))
            model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True))
            self.model = model.eval()
            self.is_mock = False
            log.info("Text model loaded from %s", CHECKPOINT.name)
        except Exception:  # a missing dependency should not take the server down
            log.exception("Could not load the text model; text stays mocked")

    def predict(self, data) -> Prediction:
        if self.model is None:
            return super().predict(data)
        enc = self.tokenizer([str(data)], truncation=True, max_length=MAX_LEN, return_tensors="pt")
        with self._lock, torch.no_grad():
            probs = torch.softmax(self.model(enc["input_ids"], enc["attention_mask"]), 1)[0]
        return Prediction.from_probs("text", probs.tolist())


MockTextPredictor = TextPredictor  # old name, kept for imports
