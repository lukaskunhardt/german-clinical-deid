"""Check rerender.py against the shipped training file (originals plus two renders each, seed 1).

  python -I tests/test_rerender.py <shipped file with renders>

The originals are the rows with metadata.render == 0. They are passed to rerender.py in their order, reduced to
doc_id, text and spans as in the published dataset, and every output row must match the shipped row in doc_id, text and spans.
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def main(shipped_path):
    shipped = [json.loads(l) for l in open(shipped_path, encoding="utf-8")]
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "train.jsonl"), os.path.join(tmp, "out.jsonl")
        with open(src, "w", encoding="utf-8") as fh:
            for r in shipped:
                if r["metadata"]["render"] == 0:
                    fh.write(json.dumps({"doc_id": r["doc_id"], "text": r["text"], "spans": r["spans"]}, ensure_ascii=False) + "\n")
        subprocess.run([sys.executable, "-I", os.path.join(HERE, "..", "rerender.py"), src, dst, "--n", "2", "--seed", "1"], check=True)
        out = [json.loads(l) for l in open(dst, encoding="utf-8")]
    key = lambda r: (r["doc_id"], r["text"], json.dumps(r["spans"]))  # noqa: E731
    same = sum(key(a) == key(b) for a, b in zip(out, shipped))
    renders = sum(key(a) == key(b) for a, b in zip(out, shipped) if b["metadata"]["render"] != 0)
    print(f"{len(out)} rows written, {len(shipped)} shipped: {same} identical (doc_id, text, spans), of which {renders} renders")
    sys.exit(0 if same == len(out) == len(shipped) else 1)


if __name__ == "__main__":
    main(sys.argv[1])
