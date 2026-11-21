"""Find and remove personal information in German clinical text.

The model is a token classifier with one class "O" (not personal) and one class per identifier type.
The redaction score of a token is 1 - P(O). A character gets the highest score of all tokens and windows that cover it.
Characters at or above the redaction threshold are redacted; the type of a span comes from the most likely type class.
"""
from __future__ import annotations

import numpy as np

MAX_LEN = 512
STRIDE = 128
DEFAULT_THRESHOLD = 0.5  # used only when config.json has no "deid_threshold"


def windows(tok, text, max_len=MAX_LEN, stride=STRIDE):
    """Tokenize the whole text once, then cut overlapping windows, so every token has context on both sides."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
    ids, offsets = enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]
    size = max_len - 2
    out, start = [], 0
    while True:
        out.append((ids[start:start + size], offsets[start:start + size]))
        if start + size >= len(ids):
            return out
        start += size - stride


def mostly_upper(text):
    letters = [c for c in text if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) > 0.8 * len(letters)


def char_spans(text, mask):
    """Runs of masked characters. Runs separated only by spaces or tabs become one span. Edges are trimmed of whitespace."""
    n, out, i = len(text), [], 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n:
            if mask[j]:
                j += 1
                continue
            k = j
            while k < n and text[k] in " \t":
                k += 1
            if k > j and k < n and mask[k]:
                j = k + 1
            else:
                break
        s, e = i, j
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        # A span without a letter or digit cannot identify anyone (table rules like "-----" scored high, 20 Nov 2026)
        if e > s and any(c.isalnum() for c in text[s:e]):
            out.append((s, e))
        i = j
    return out


class OnnxModel:
    """Stands in for the PyTorch model: same inputs, logits from onnxruntime on the CPU."""

    def __init__(self, path, config):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.config = config

    def __call__(self, input_ids, attention_mask):
        import torch

        feed = {"input_ids": input_ids.cpu().numpy(), "attention_mask": attention_mask.cpu().numpy()}
        return type("Output", (), {"logits": torch.from_numpy(self.sess.run(["logits"], feed)[0])})


class Deidentifier:
    def __init__(self, model, threshold=None, type_threshold=None, device=None, batch_size=16, onnx_file=None):
        """model: Hub id or local folder. onnx_file: path inside it (for example "onnx/model_fp16.onnx") to run with onnxruntime."""
        import torch
        from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        if onnx_file:
            import os

            from huggingface_hub import hf_hub_download

            path = os.path.join(model, onnx_file) if os.path.isdir(model) else hf_hub_download(model, onnx_file)
            self.model, self.device = OnnxModel(path, AutoConfig.from_pretrained(model)), "cpu"
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(model).eval()
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        cfg = self.model.config
        self.labels = [cfg.id2label[i] for i in range(cfg.num_labels)]
        assert self.labels[0] == "O", "class 0 must be O"
        # The redaction threshold was chosen on development data and is stored with the model.
        self.threshold = float(threshold if threshold is not None else getattr(cfg, "deid_threshold", DEFAULT_THRESHOLD))
        self.type_threshold = float(type_threshold if type_threshold is not None else getattr(cfg, "deid_type_threshold", 0.5))
        self.batch_size = batch_size

    def _read(self, text):
        """Per character: redaction score and the type index of the token that gave it."""
        torch, tok = self.torch, self.tok
        score = np.zeros(len(text), dtype=np.float32)
        kind = np.zeros(len(text), dtype=np.int16)
        wins = windows(tok, text)
        for i in range(0, len(wins), self.batch_size):
            part = wins[i:i + self.batch_size]
            width = max(len(w) for w, _ in part) + 2
            ids = torch.full((len(part), width), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(part), width), dtype=torch.long)
            for r, (w, _) in enumerate(part):
                row = [tok.cls_token_id] + w + [tok.sep_token_id]
                ids[r, :len(row)] = torch.tensor(row)
                att[r, :len(row)] = 1
            with torch.no_grad():
                logits = self.model(input_ids=ids.to(self.device), attention_mask=att.to(self.device)).logits
            probs = torch.softmax(logits.float(), dim=-1)
            p = (1 - probs[..., 0]).cpu().numpy()
            k = (probs[..., 1:].argmax(-1) + 1).cpu().numpy()
            for r, (_, offsets) in enumerate(part):
                for t, (a, b) in enumerate(offsets):
                    if b > a:
                        kind[a:b] = np.where(p[r, t + 1] > score[a:b], k[r, t + 1], kind[a:b])
                        score[a:b] = np.maximum(score[a:b], p[r, t + 1])
        return score, kind

    def read(self, text):
        """Scores and types per character. All-caps text (exports from hospital systems) is also read in lower case,
        because the model was trained mostly on mixed case and misses names in capitals; the higher score wins per character."""
        score, kind = self._read(text)
        lower = text.lower()
        if mostly_upper(text) and len(lower) == len(text):
            score2, kind2 = self._read(lower)
            kind = np.where(score2 > score, kind2, kind)
            score = np.maximum(score, score2)
        return score, kind

    def spans(self, text, threshold=None, scores=None):
        """Redaction spans: list of (start, end, score)."""
        score = self.read(text)[0] if scores is None else scores[0]
        thr = self.threshold if threshold is None else threshold
        return [(s, e, float(score[s:e].max())) for s, e in char_spans(text, score >= thr)]

    def span_type(self, kind, s, e):
        """Most frequent type over the characters of a span."""
        k = kind[s:e]
        k = k[k > 0]
        return self.labels[int(np.bincount(k).argmax())] if len(k) else "PHI"

    def entities(self, text, threshold=None, scores=None):
        """Typed spans at the type threshold. A span is cut where the predicted type changes, so
        "Hans Müller, 12345 Berlin" gives a name, a postal code and a city. Each piece runs from its first to its last letter or digit."""
        score, kind = self.read(text) if scores is None else scores
        thr = self.type_threshold if threshold is None else threshold
        out = []
        for s, e in char_spans(text, score >= thr):
            start = cur = last = None
            for i in range(s, e):
                if not text[i].isalnum():
                    continue
                k = int(kind[i])
                if cur is None:
                    start, cur = i, k
                elif k != cur:
                    out.append({"start": start, "end": last + 1, "label": self.labels[cur], "text": text[start:last + 1]})
                    start, cur = i, k
                last = i
            if cur is not None:
                out.append({"start": start, "end": last + 1, "label": self.labels[cur], "text": text[start:last + 1]})
        return out

    def redact(self, text, placeholder="[PHI]"):
        parts, pos = [], 0
        for s, e, _ in self.spans(text):
            parts += [text[pos:s], placeholder]
            pos = e
        return "".join(parts) + text[pos:]
