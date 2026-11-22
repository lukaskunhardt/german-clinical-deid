"""Train the smaller model (gbert-base) by distillation from the large one.

  python rerender.py --download train_r10.jsonl --n 10 --seed 7   # 2,550 letters + 10 re-renderings each = 28,050 letters
  python distill.py --data train_r10.jsonl --seed 42 --out model-base

Recipe (one NVIDIA A10G: about 25 minutes for the teacher pass, 15 minutes to train the student):
  - the teacher is the released large model; it reads every letter once and gives each character a class distribution
  - the student target for a token is the mean of the teacher's distributions over the token's characters
  - loss per token: alpha * KL(teacher || student) + (1 - alpha) * cross-entropy on the gold label, alpha 0.5
  - same windows (512 tokens, stride 128), optimiser and learning rate as train.py, but 1 epoch on 3.7 times as many letters
  - 5 % of the letters are a validation split (gold labels only); the model after the last step is kept
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # also under python -I
from train import IGNORE, LABELS, THRESHOLD, TYPE_THRESHOLD, encode, hash_split, load_letters, windows  # noqa: E402


@torch.no_grad()
def teacher_probs(model, tok, text, device, max_len=512, stride=128, batch=16):
    """Class distribution per character (len(text) x classes). Where windows overlap, the window with the higher PHI probability wins."""
    dist = np.zeros((len(text), len(LABELS)), dtype=np.float32)
    dist[:, 0] = 1.0
    phi = np.zeros(len(text), dtype=np.float32)
    wins = windows(tok, text, max_len, stride)
    for i in range(0, len(wins), batch):
        part = wins[i:i + batch]
        width = max(len(ids) for ids, _ in part) + 2
        input_ids = torch.full((len(part), width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(part), width), dtype=torch.long)
        for r, (ids, _) in enumerate(part):
            row = [tok.cls_token_id] + ids + [tok.sep_token_id]
            input_ids[r, :len(row)] = torch.tensor(row)
            mask[r, :len(row)] = 1
        logits = model(input_ids=input_ids.to(device), attention_mask=mask.to(device)).logits
        probs = torch.softmax(logits.float(), -1).cpu().numpy()
        for r, (_, offsets) in enumerate(part):
            for t, (a, b) in enumerate(offsets):
                if b > a:
                    v = probs[r, t + 1]
                    higher = phi[a:b] < 1.0 - v[0]
                    dist[a:b][higher] = v
                    phi[a:b][higher] = 1.0 - v[0]
    return dist


def encode_soft(letters, tok, teacher, teacher_tok, device, max_len, stride):
    """Windows with gold labels (as in train.py) plus the teacher target per token, stored as uint8 to keep 28,050 letters in memory."""
    items = []
    for n, d in enumerate(letters):
        dist = teacher_probs(teacher, teacher_tok, d["text"], device, max_len, stride)
        for item, (ids, offsets) in zip(encode([d], tok, max_len, stride), windows(tok, d["text"], max_len, stride)):
            soft = np.zeros((len(ids) + 2, len(LABELS)), dtype=np.uint8)
            for t, (a, b) in enumerate(offsets):
                if b > a:
                    soft[t + 1] = np.round(dist[a:b].mean(0) * 255).astype(np.uint8)
            items.append({**item, "soft": soft})
        if n % 1000 == 0:
            print(f"teacher: {n}/{len(letters)} letters, {len(items)} windows", flush=True)
    return items


def collate(pad_id):
    def pad(batch):
        width = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": torch.full((len(batch), width), pad_id, dtype=torch.long),
               "attention_mask": torch.zeros((len(batch), width), dtype=torch.long),
               "labels": torch.full((len(batch), width), IGNORE, dtype=torch.long)}
        if "soft" in batch[0]:
            out["soft"] = torch.zeros((len(batch), width, len(LABELS)), dtype=torch.float32)
        for r, b in enumerate(batch):
            n = len(b["input_ids"])
            out["input_ids"][r, :n] = torch.tensor(b["input_ids"])
            out["attention_mask"][r, :n] = 1
            out["labels"][r, :n] = torch.tensor(b["labels"])
            if "soft" in b:
                out["soft"][r, :n] = torch.from_numpy(b["soft"].astype(np.float32) / 255.0)
        return out
    return pad


def distill_loss(logits, labels, soft, alpha):
    """alpha * KL(teacher || student) + (1 - alpha) * cross-entropy, both averaged over labelled tokens. Without soft targets: cross-entropy."""
    logits = logits.float()
    ce = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=IGNORE)
    if soft is None:
        return ce
    keep = (labels != IGNORE).float()
    # the uint8 rounding leaves sums slightly off 1, so the target is normalised again
    q = soft / soft.sum(-1, keepdim=True).clamp_min(1e-6)
    kl = (q * (torch.log(q.clamp_min(1e-6)) - torch.log_softmax(logits, -1))).sum(-1)
    kl = (kl * keep).sum() / keep.sum().clamp_min(1.0)
    return alpha * kl + (1 - alpha) * ce


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="jsonl from rerender.py (letters with spans)")
    ap.add_argument("--teacher", default="LukasKunhardt/german-clinical-deid")
    ap.add_argument("--model", default="deepset/gbert-base")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--out", default="model-base")
    ap.add_argument("--work", default="checkpoints")
    a = ap.parse_args()

    from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val = hash_split(load_letters(a.data), 0.05, 43)
    teacher_tok = AutoTokenizer.from_pretrained(a.teacher)
    teacher = AutoModelForTokenClassification.from_pretrained(a.teacher).to(device).eval()
    if device == "cuda":
        teacher = teacher.half()
    assert [teacher.config.id2label[i] for i in range(len(LABELS))] == LABELS, "the teacher has a different label set"
    tok = AutoTokenizer.from_pretrained(a.model)
    train_ds = encode_soft(train, tok, teacher, teacher_tok, device, a.max_length, a.stride)
    val_ds = encode(val, tok, a.max_length, a.stride)
    del teacher
    steps = math.ceil(math.ceil(len(train_ds) / 8) / 2 * a.epochs)
    print(f"{len(train)} train letters ({len(train_ds)} windows), {len(val)} validation, {steps} steps", flush=True)

    model = AutoModelForTokenClassification.from_pretrained(
        a.model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id={k: i for i, k in enumerate(LABELS)})

    class DistillTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels, soft = inputs.pop("labels"), inputs.pop("soft", None)
            out = model(**inputs)
            loss = distill_loss(out.logits, labels, soft, a.alpha)
            return (loss, out) if return_outputs else loss

    args = TrainingArguments(
        output_dir=a.work, num_train_epochs=a.epochs, per_device_train_batch_size=8, gradient_accumulation_steps=2,
        learning_rate=a.lr, weight_decay=0.01, warmup_steps=int(0.1 * steps), logging_steps=50, eval_strategy="epoch",
        save_strategy="no", seed=a.seed, fp16=device == "cuda", report_to=[], disable_tqdm=True, remove_unused_columns=False)
    trainer = DistillTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collate(tok.pad_token_id))
    trainer.train()
    for h in trainer.state.log_history:
        if "eval_loss" in h:
            print(f"epoch {h['epoch']:.0f}: validation loss {h['eval_loss']:.4f}", flush=True)

    model.config.deid_threshold = THRESHOLD
    model.config.deid_type_threshold = TYPE_THRESHOLD
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
