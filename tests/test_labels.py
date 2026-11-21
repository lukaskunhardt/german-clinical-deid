"""Check labels.py against the original research code. No API calls.

  python -I tests/test_labels.py <research repo> [old labels.py]

1. Shipped corpus: labels.py on the 3,144 raw records of the shipped corpus must give exactly the 3,023 labelled
   rows the original run wrote (text, spans, metadata in the published format; the original rows also carried a
   clinical span layer and more metadata, which are left out here).
2. Old code: labels.py and the old labels.py (default: the research repo's current one) on every raw record of a
   tagless writer found under runs/: same accept/reject decision, same reasons, same rows.
3. The 10-gram function of the duplicate check gives the same sets as the original on the shipped texts.
"""
import glob
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import labels  # noqa: E402

REASONS = {"leer": "empty", "fremde Schrift": "foreign script", "Markierung übrig": "markup", "Markierung kaputt": "markup",
           "Entwurf": "draft", "Schweiz": "Swiss", "zu kurz": "too short", "ICD-10-CM": "ICD-10-CM", "Vorname ohne Markierung": "first name also a word"}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def strip(row):
    """Old row in the published format: no clinical span layer, English metadata keys."""
    m = row["metadata"]
    return {"doc_id": row["doc_id"], "text": row["text"], "spans": row["spans"],
            "metadata": {"model": m["model"], "temperature": m["temperature"], "document_type": m["dokumentart"],
                         "specialty": m["fachgebiet"], "main_diagnosis": m["hauptdiagnose"]}}


def main(repo, old_path):
    ok = True
    # 1. shipped corpus
    shipped = {r["doc_id"]: strip(r) for r in map(json.loads, open(os.path.join(repo, "runs/v3e_ds3/korpus.jsonl"), encoding="utf-8"))}
    new = {}
    for rec in map(json.loads, open(os.path.join(repo, "runs/v3e_ds3/alle.jsonl"), encoding="utf-8")):
        row, _ = labels.finalize(rec) if "error" not in rec else (None, [])
        if row:
            new[row["doc_id"]] = row
    same = sum(json.dumps(new[k], ensure_ascii=False) == json.dumps(shipped.get(k), ensure_ascii=False) for k in new)
    print(f"shipped corpus: {len(new)} labelled (original run {len(shipped)}), {same} rows byte-identical")
    ok &= same == len(new) == len(shipped)

    # 2. old code on every tagless raw record
    old = load(old_path, "old_labels")
    seen, n, same_decision, same_row, same_reasons, skipped = set(), 0, 0, 0, 0, 0
    for path in sorted(glob.glob(os.path.join(repo, "runs/**/briefe.jsonl"), recursive=True)):
        if os.path.realpath(path) in seen:
            continue
        seen.add(os.path.realpath(path))
        for rec in map(json.loads, open(path, encoding="utf-8")):
            if "error" in rec or not rec.get("ohne_markierung") or rec.get("spans"):
                skipped += 1  # tagged writers of earlier corpus versions; labels.py only handles tagless letters
                continue
            n += 1
            a, ra = old.finalize(rec)
            b, rb = labels.finalize(rec)
            same_decision += (a is None) == (b is None)
            same_row += (a is None and b is None) or (a is not None and b is not None and json.dumps(strip(a), ensure_ascii=False) == json.dumps(b, ensure_ascii=False))
            old_reasons = {REASONS.get(r.split(":")[0], r.split(":")[0]) for r in ra if not r.startswith(("Angabe offen", "Datum offen", "Titel offen", "Alter offen"))}
            same_reasons += old_reasons == {r.split(":")[0] for r in rb}
    print(f"old code ({os.path.relpath(old_path, repo) if old_path.startswith(repo) else os.path.basename(old_path)}): {n} tagless raw records in "
          f"{len(seen)} files ({skipped} other records skipped): {same_decision} same decision, {same_row} same row, {same_reasons} same reasons")
    ok &= n == same_decision == same_row == same_reasons

    # 3. duplicate check
    sys.path.insert(0, os.path.join(repo, "bench"))
    dup = load(os.path.join(repo, "bench/dubletten.py"), "old_dup")
    texts = [r["text"] for r in shipped.values()]
    same = sum(dup.grams(t) == labels.grams(t) for t in texts)
    print(f"10-grams: {same} of {len(texts)} texts identical")
    ok &= same == len(texts)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    repo = os.path.abspath(sys.argv[1])
    main(repo, os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(repo, "fabrik/labels.py"))
