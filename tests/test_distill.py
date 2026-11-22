"""Check distill.py against the research distillation script that trained the released smaller model.

  python -I tests/test_distill.py --research <research repo> --data <train_r10.jsonl> --tokenizer <gbert-base folder or Hub id>

1. Loss: on a random batch, distill_loss gives the same value as the research trainer's compute_loss (with and without teacher targets).
2. Split: the same train and validation letters.
3. Targets: with a tiny random teacher, teacher_probs and encode_soft give the same character distributions, token labels and
   uint8 targets as the research code, on the first letters of the file.
Runs on a CPU in about a minute; no trained model is needed.
"""
import argparse
import ast
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import distill  # noqa: E402
from train import IGNORE, LABELS, hash_split, load_letters  # noqa: E402


def research_trainer(path, alpha):
    """The research script defines its trainer inside main(); take that class out of the source and build it with the same names."""
    from transformers import Trainer
    tree = ast.parse(open(path, encoding="utf-8").read())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "DistillTrainer")
    scope = {"Trainer": Trainer, "torch": torch, "IGNORE": IGNORE, "alpha": alpha}
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), scope)
    return object.__new__(scope["DistillTrainer"])


class FixedLogits(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits

    def forward(self, **_):
        return type("Out", (), {"logits": self.logits})()


def check_loss(research_file):
    g = torch.Generator().manual_seed(0)
    B, T, K = 3, 17, len(LABELS)
    logits = torch.randn(B, T, K, generator=g) * 3
    labels = torch.randint(0, K, (B, T), generator=g)
    labels[:, 0] = labels[:, -1] = IGNORE
    labels[1, 10:] = IGNORE
    soft = torch.from_numpy(np.round(torch.softmax(torch.randn(B, T, K, generator=g) * 4, -1).numpy() * 255) / 255).float()
    ok = True
    for alpha in (0.5, 0.3):
        trainer = research_trainer(research_file, alpha)
        for with_soft in (True, False):
            inputs = {"input_ids": torch.zeros(B, T, dtype=torch.long), "labels": labels.clone()}
            if with_soft:
                inputs["soft"] = soft.clone()
            old = trainer.compute_loss(FixedLogits(logits), inputs).item()
            new = distill.distill_loss(logits, labels, soft if with_soft else None, alpha).item()
            same = abs(old - new) < 1e-6
            ok &= same
            print(f"loss alpha {alpha}, teacher targets {with_soft}: research {old:.6f}, distill.py {new:.6f}, {'same' if same else 'DIFFERENT'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--research", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", default="deepset/gbert-base")
    ap.add_argument("--letters", type=int, default=30)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.research, "bench"))
    import core
    import train_distill as old

    ok = check_loss(os.path.join(a.research, "bench", "train_distill.py"))

    new_letters = load_letters(a.data)
    old_docs = core.load_sparcling(a.data)
    split_new = [[d["id"] for d in part] for part in hash_split(new_letters, 0.05, 43)]
    split_old = [[d.doc_id for d in part] for part in old.hash_split(old_docs, 0.05, 43)]
    print(f"split: {len(split_new[0])} train, {len(split_new[1])} validation, {'same' if split_new == split_old else 'DIFFERENT'} as research")
    ok &= split_new == split_old

    from transformers import AutoTokenizer, BertConfig, BertForTokenClassification
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=tok.vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                     num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id={k: i for i, k in enumerate(LABELS)})
    teacher = BertForTokenClassification(cfg).eval()
    n = a.letters
    dist_same = all(np.array_equal(distill.teacher_probs(teacher, tok, d["text"], "cpu"), old.lehrer_verteilung(teacher, tok, d["text"], "cpu"))
                    for d in new_letters[:n])
    new_items = distill.encode_soft(new_letters[:n], tok, teacher, tok, "cpu", 512, 128)
    old_items = old.DistillChunks(old_docs[:n], tok, teacher, tok, "cpu", 512, 128).items
    items_same = len(new_items) == len(old_items) and all(
        x["input_ids"] == y["input_ids"] and x["labels"] == y["labels"] and np.array_equal(x["soft"], y["soft"]) for x, y in zip(new_items, old_items))
    batch_new, batch_old = distill.collate(tok.pad_token_id)(new_items), old.collate(tok.pad_token_id)(old_items)
    batch_same = all(torch.equal(batch_new[k], batch_old[k]) for k in batch_old)
    print(f"{n} letters, {len(new_items)} windows: character distributions {'same' if dist_same else 'DIFFERENT'}, "
          f"windows and targets {'same' if items_same else 'DIFFERENT'}, padded batch {'same' if batch_same else 'DIFFERENT'}")
    ok &= dist_same and items_same and batch_same
    print("all checks passed" if ok else "CHECK FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
