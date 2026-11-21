"""Train the German clinical de-identification model.

  python rerender.py --download train.jsonl --n 2 --seed 1   # 2,550 letters + 2 re-renderings each = 7,650 letters
  python train.py --data train.jsonl --seed 44 --out model

Recipe (one NVIDIA A10G, about 45 minutes per seed for gelectra-large):
  - token classification with one class per identifier type (IO tags) plus O
  - a token gets the type of the characters it covers; letters are cut into windows of 512 tokens with a stride of 128
  - AdamW, learning rate 3e-5, weight decay 0.01, 10 % warmup, batch 8 x 2 accumulation, 3 epochs
  - the checkpoint with the best token F1 (PHI against O) on a 10 % synthetic validation split is kept
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os

import numpy as np
import torch

# Identifier types of the training data (label names of the GraSCCo corpus). The order fixes the class ids of the shipped model.
LABELS = ["O", "DATE", "ID", "NAME_DOCTOR", "NAME_TITLE", "LOCATION_CITY", "NAME_PATIENT", "LOCATION_ZIP", "LOCATION_STREET", "AGE",
          "LOCATION_HOSPITAL", "NAME_USERNAME", "CONTACT_PHONE", "CONTACT_FAX", "CONTACT_EMAIL", "CONTACT_URL", "PROFESSION",
          "NAME_RELATIVE", "NAME_EXT"]
THRESHOLD = 0.01  # redaction threshold of the shipped model, chosen on development data (GraSCCo dev, MDeID German train split)
TYPE_THRESHOLD = 0.5
IGNORE = -100


def load_letters(path):
    letters = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            spans = sorted((s["start"], s["end"], s["label"]) for s in r["spans"] if s["end"] > s["start"])
            letters.append({"id": r["doc_id"], "text": r["text"], "spans": spans})
    return letters


def hash_split(letters, frac, seed):
    """Stable split on md5(seed:id). Re-renderings have their own ids and are split independently of their original."""
    rest, held = [], []
    for d in letters:
        h = hashlib.md5(f"{seed}:{d['id']}".encode()).hexdigest()
        (held if int(h[:8], 16) / 0xFFFFFFFF < frac else rest).append(d)
    return rest, held


def windows(tok, text, max_len, stride):
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
    ids, offsets = enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]
    size = max_len - 2
    out, start = [], 0
    while True:
        out.append((ids[start:start + size], offsets[start:start + size]))
        if start + size >= len(ids):
            return out
        start += size - stride


def encode(letters, tok, max_len, stride):
    items = []
    for d in letters:
        mask = bytearray(len(d["text"]))
        for s, e, label in d["spans"]:
            for k in range(max(0, s), min(len(mask), e)):
                mask[k] = LABELS.index(label) if label in LABELS else LABELS.index("ID")
        for ids, offsets in windows(tok, d["text"], max_len, stride):
            # A token that touches two types gets the higher class id; this is rare and was kept as trained.
            labels = [max(mask[a:b]) if b > a else IGNORE for a, b in offsets]
            items.append({"input_ids": [tok.cls_token_id] + ids + [tok.sep_token_id],
                          "attention_mask": [1] * (len(ids) + 2),
                          "labels": [IGNORE] + labels + [IGNORE]})
    return items


def token_f1(eval_pred):
    """Token F1 of PHI against O, ignoring the type: the model is selected for redaction quality."""
    logits, labels = eval_pred
    keep = labels != IGNORE
    p, y = (np.argmax(logits, axis=-1)[keep] > 0), (labels[keep] > 0)
    tp, fp, fn = int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": pr, "recall": rc, "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="jsonl from rerender.py (letters with spans)")
    ap.add_argument("--model", default="deepset/gelectra-large")
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--out", default="model")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--work", default="checkpoints")
    a = ap.parse_args()

    from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer,
                              TrainingArguments)

    letters = load_letters(a.data)
    # 10 % stay out of training as a synthetic hold-out, then 10 % of the rest is the validation split for checkpoint selection.
    rest, holdout = hash_split(letters, 0.1, 42)
    train, val = hash_split(rest, 0.1, 43)
    tok = AutoTokenizer.from_pretrained(a.model)
    train_ds = encode(train, tok, a.max_length, a.stride)
    val_ds = encode(val, tok, a.max_length, a.stride)
    steps = math.ceil(math.ceil(len(train_ds) / 8) / 2 * a.epochs)
    print(f"{len(train)} train letters ({len(train_ds)} windows), {len(val)} validation, {len(holdout)} hold-out, {steps} steps", flush=True)

    model = AutoModelForTokenClassification.from_pretrained(
        a.model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id={k: i for i, k in enumerate(LABELS)})
    cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=a.work, num_train_epochs=a.epochs, per_device_train_batch_size=8, per_device_eval_batch_size=16,
        gradient_accumulation_steps=2, learning_rate=a.lr, weight_decay=0.01, warmup_steps=int(0.1 * steps), logging_steps=20,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="f1", greater_is_better=True, seed=a.seed, fp16=cuda, report_to=[], disable_tqdm=True)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorForTokenClassification(tok), compute_metrics=token_f1)
    trainer.train()
    for h in trainer.state.log_history:
        if "eval_f1" in h:
            print(f"epoch {h['epoch']:.0f}: validation token F1 {h['eval_f1']:.4f}", flush=True)

    model.config.deid_threshold = THRESHOLD
    model.config.deid_type_threshold = TYPE_THRESHOLD
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
