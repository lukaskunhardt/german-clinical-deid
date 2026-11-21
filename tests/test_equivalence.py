"""Check that this repo gives the same results as the research code that produced the released numbers.

  python tests/test_equivalence.py --model <model folder> --research <research repo> --grascco <GraSCCo folder> [--saved <saved dev predictions .jsonl>]

1. Package: the same redaction spans, typed spans and scores as the research package, on the 22 GraSCCo dev letters,
   the quick-start sentence and an all-caps copy of one letter (this exercises the lower-case second reading).
2. Saved predictions: the same spans as the saved development run of the research measurement code (optional).
3. Measures: evaluate.py gives the same numbers as the research scoring code for the same predictions.
4. Training data: train.py builds the same splits and token labels as the research training script.
Only development letters are read.
"""
import argparse
import gc
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import evaluate  # noqa: E402

QUICK = "Herr Max Muster (geb. 03.02.1961) wurde am 12.05.2024 im Klinikum Nord aufgenommen."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--research", required=True)
    ap.add_argument("--grascco", required=True)
    ap.add_argument("--saved", default="")
    ap.add_argument("--train-data", default="", help="training jsonl with re-renderings, for check 4")
    a = ap.parse_args()
    docs = evaluate.load_grascco(a.grascco, "dev")
    texts = [d["text"] for d in docs] + [QUICK, docs[0]["text"].upper()]
    print(f"{len(docs)} dev letters + 2 extra texts")

    sys.path.insert(0, os.path.join(a.research, "pakete"))
    from medlm_deid import Schwaerzer
    old = Schwaerzer(a.model, schwelle=0.01)
    ref = []
    for t in texts:
        p = old.wahrscheinlichkeit(t)
        ref.append((p, old.spans(t, p), old.spans_nach_art(t, p, 0.5)))
    del old
    gc.collect()

    from german_clinical_deid import Deidentifier
    new = Deidentifier(a.model)
    assert new.threshold == 0.01 and new.type_threshold == 0.5, (new.threshold, new.type_threshold)
    same_spans = same_types = 0
    max_diff = 0.0
    preds = {}
    for i, t in enumerate(texts):
        scores = new.read(t)
        spans = new.spans(t, scores=scores)
        ents = [(x["start"], x["end"], x["label"]) for x in new.entities(t, scores=scores)]
        max_diff = max(max_diff, float(np.abs(scores[0] - ref[i][0]).max()) if len(t) else 0.0)
        same_spans += [(s, e) for s, e, _ in spans] == [(s, e) for s, e, _ in ref[i][1]]
        same_types += ents == ref[i][2]
        if i < len(docs):
            preds[docs[i]["id"]] = {"spans": [(s, e) for s, e, _ in spans], "entities": ents}
    print(f"1. package: spans identical {same_spans}/{len(texts)}, typed spans identical {same_types}/{len(texts)}, "
          f"max score difference {max_diff:.2e}")
    print("   quick start:", new.redact(QUICK))
    print("   entities:", [(x["text"], x["label"]) for x in new.entities(QUICK)])

    if a.saved:
        saved = {r["doc_id"]: r for r in map(json.loads, open(a.saved)) if r["satz"] == "grascco_de"}
        ok_s = sum([tuple(x) for x in saved[k]["pred_t"]["0.01"]] == preds[k]["spans"] for k in preds)
        ok_t = sum([tuple(x) for x in saved[k]["arten"]["nach_art"]["0.5"]] == preds[k]["entities"] for k in preds)
        print(f"2. saved development run: spans identical {ok_s}/{len(preds)}, typed spans identical {ok_t}/{len(preds)}")

    sys.path.insert(0, os.path.join(a.research, "bench"))
    import core
    m = evaluate.score(docs, preds)
    # gold of the research side: its own copy of the GraSCCo dev letters, so the loader in evaluate.py is checked too
    gold = {d.doc_id: d for d in core.load_gold_jsonl(os.path.join(a.research, "data", "bank", "grascco_de", "dev.jsonl"))}
    same_gold = sorted(gold) == sorted(preds) and all(
        [tuple(s[:3]) for s in gold[d["id"]].spans] == list(d["spans"]) and gold[d["id"]].text == d["text"] for d in docs)
    print(f"   GraSCCo dev gold identical in both loaders: {same_gold}")
    agg = core.aggregate([core.score_doc(gold[k], preds[k]["spans"]) for k in preds])
    g_typed = [(set((s, e, k) for s, e, k, *_ in gold[i].spans), set(preds[i]["entities"])) for i in preds]
    tp = sum(len(g & p) for g, p in g_typed)
    fp = sum(len(p - g) for g, p in g_typed)
    fn = sum(len(g - p) for g, p in g_typed)
    typed_f1 = 2 * tp / (2 * tp + fp + fn)
    checks = {"char_f1": (m["char_f1"], agg["char_f1"]), "char_p": (m["char_p"], agg["char_p"]), "char_r": (m["char_r"], agg["char_r"]),
              "leak_rate": (m["leak_rate"], agg["doc_leak_rate"]), "strict_typed_f1": (m["strict_typed_f1"], typed_f1)}
    for k, (x, y) in checks.items():
        print(f"3. {k}: evaluate.py {x:.4f}, research code {y:.4f}, {'same' if abs(x - y) < 1e-9 else 'DIFFERENT'}")
    print(f"   leaks {m['leaks']}/{m['docs_with_direct']}, strict leaks {m['leaks_strict']}/{m['docs_with_direct']}")

    if a.train_data:
        import train
        from transformers import AutoTokenizer
        sys.argv = ["train_tc.py"]
        import train_tc
        train_tc.TYPEN[:] = [True]
        letters = train.load_letters(a.train_data)
        rest, hold = train.hash_split(letters, 0.1, 42)
        tr, val = train.hash_split(rest, 0.1, 43)
        old_docs = core.load_sparcling(a.train_data)
        o_rest, o_hold = train_tc.hash_split(old_docs, 0.1, 42)
        o_tr, o_val = train_tc.hash_split(o_rest, 0.1, 43)
        print(f"4. splits train/val/hold-out: new {len(tr)}/{len(val)}/{len(hold)}, research {len(o_tr)}/{len(o_val)}/{len(o_hold)}, "
              f"same ids {[d['id'] for d in tr] == [d.doc_id for d in o_tr]}")
        tok = AutoTokenizer.from_pretrained(a.model)
        sample = list(range(0, len(tr), 50))
        new_items = train.encode([tr[i] for i in sample], tok, 512, 128)
        old_items = train_tc.Chunks([o_tr[i] for i in sample], tok, 512, 128).items
        print(f"   token labels of {len(sample)} letters ({len(new_items)} windows): identical {new_items == old_items}")


if __name__ == "__main__":
    main()
