"""Measure the model on your own annotated documents. The script prints only counts and scores, never text.

Your documents stay where they are. Nothing is sent anywhere: after the model files are downloaded once
(or copied in, see below), the script runs offline. The output is one JSON file with sums over all documents.
It contains no text, no file names and no per-document rows, so it can be shared.

Supported annotation formats:
  brat       folder with doc.txt + doc.ann pairs (entity lines "T1<TAB>LABEL start end<TAB>text")
  inception  folder with UIMA CAS JSON files exported from INCEpTION (as in GraSCCo); give --layer and --feature
  gate       folder with GATE XML files ("Save as XML" in GATE Developer); give --annotation-set (default phi-annotations)
  jsonl      one JSON object per line: {"text": "...", "spans": [{"start": 0, "end": 5, "label": "NAME"}, ...]}

  python partner_eval.py --format brat --path my_notes/ --direct NAME,PATIENT,ID,PHONE,STREET --out counts.json
  python partner_eval.py --format inception --path export/ --layer webanno.custom.PHI --feature kind --direct NAME_PATIENT,ID

Offline use: copy the model folder (git clone https://huggingface.co/LukasKunhardt/german-clinical-deid) and pass --model <folder>,
then set HF_HUB_OFFLINE=1.

Two readings are measured, as on the model card:
  helper    the german_clinical_deid package: redacts every token whose probability of being personal is >= the stored threshold
  pipeline  transformers pipeline("token-classification", aggregation_strategy="simple", stride=128)

Measures (micro over all documents, annotation type ignored, so any label scheme works):
  token P/R/F1   whitespace words: a word is positive if it overlaps a span
  char P/R/F1    characters inside gold spans against redacted characters
  per label      for each of your labels: number of gold spans, and how many are fully / at least half / not at all redacted
  missed direct  documents in which at least one span with a label from --direct is less than half redacted
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter


def load_brat(folder):
    docs = []
    for path in sorted(glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True)):
        ann = path[:-4] + ".ann"
        if not os.path.exists(ann):
            continue
        text = open(path, encoding="utf-8").read()
        spans = []
        for line in open(ann, encoding="utf-8"):
            if line.startswith("T"):
                parts = line.rstrip("\n").split("\t")[1].replace(";", " ").split()
                nums = [int(x) for x in parts[1:] if x.isdigit()]
                if nums:
                    spans.append((min(nums), max(nums), parts[0]))
        docs.append({"text": text, "spans": sorted(spans)})
    return docs


def load_inception(folder, layer, feature):
    docs = []
    for path in sorted(glob.glob(os.path.join(folder, "**", "*.json"), recursive=True)):
        fss = json.load(open(path, encoding="utf-8")).get("%FEATURE_STRUCTURES", [])
        sofa = next((x for x in fss if x.get("%TYPE") == "uima.cas.Sofa"), None)
        if sofa is None:
            continue
        spans = sorted((x["begin"], x["end"], str(x.get(feature) or "PHI")) for x in fss
                       if x.get("%TYPE") == layer and x.get("begin") is not None and x.get("end", 0) > x["begin"])
        docs.append({"text": sofa["sofaString"], "spans": spans})
    return docs


def load_gate(folder, annotation_set):
    """GATE XML: the text is spread over <TextWithNodes> with <Node id=offset/> markers; annotations point at node ids."""
    import xml.etree.ElementTree as ET
    docs = []
    for path in sorted(glob.glob(os.path.join(folder, "**", "*.xml"), recursive=True)):
        root = ET.parse(path).getroot()
        twn = root.find("TextWithNodes")
        if twn is None:
            continue
        parts, nodes = [twn.text or ""], {}
        pos = len(parts[0])
        for node in twn:
            nodes[node.get("id")] = pos
            parts.append(node.tail or "")
            pos += len(node.tail or "")
        text = "".join(parts)
        spans = []
        for aset in root.findall("AnnotationSet"):
            if (aset.get("Name") or "") != annotation_set:
                continue
            for a in aset.findall("Annotation"):
                s = nodes.get(a.get("StartNode"), int(a.get("StartNode")))
                e = nodes.get(a.get("EndNode"), int(a.get("EndNode")))
                if e > s:
                    spans.append((s, e, a.get("Type")))
        docs.append({"text": text, "spans": sorted(spans)})
    return docs


def load_jsonl(path):
    docs = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            docs.append({"text": r["text"], "spans": sorted((s["start"], s["end"], s.get("label", "PHI")) for s in r["spans"])})
    return docs


def prf(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return round(p, 4), round(r, 4), round(2 * p * r / (p + r), 4) if p + r else 0.0


def score(docs, redactions, direct):
    """docs: [{"text", "spans": [(s, e, label)]}]; redactions: one list of (s, e) per document. Returns sums only."""
    c = Counter()
    per_label = {}
    for d, red_spans in zip(docs, redactions):
        n = len(d["text"])
        gold, red = bytearray(n), bytearray(n)
        for s, e, _ in d["spans"]:
            gold[s:e] = b"\x01" * (e - s)
        for s, e in red_spans:
            red[s:e] = b"\x01" * (e - s)
        c["c_gold"] += sum(gold); c["c_pred"] += sum(red); c["c_both"] += sum(a & b for a, b in zip(gold, red))
        for m in re.finditer(r"\S+", d["text"]):
            g, p = any(gold[m.start():m.end()]), any(red[m.start():m.end()])
            c["t_gold"] += g; c["t_pred"] += p; c["t_both"] += g and p
        missed_direct = False
        for s, e, label in d["spans"]:
            share = sum(red[s:e]) / (e - s)
            row = per_label.setdefault(label, Counter())
            row["gold_spans"] += 1
            row["fully_redacted"] += share == 1.0
            row["at_least_half"] += share >= 0.5
            row["not_redacted"] += share == 0.0
            if label in direct and share < 0.5:
                missed_direct = True
        if any(label in direct for _, _, label in d["spans"]):
            c["docs_with_direct"] += 1
            c["docs_missed_direct"] += missed_direct
    tp, tr, tf = prf(c["t_both"], c["t_pred"], c["t_gold"])
    cp, cr, cf = prf(c["c_both"], c["c_pred"], c["c_gold"])
    return {"documents": len(docs), "token_p": tp, "token_r": tr, "token_f1": tf, "char_p": cp, "char_r": cr, "char_f1": cf,
            "docs_with_direct": c["docs_with_direct"], "docs_missed_direct": c["docs_missed_direct"],
            "per_label": {k: dict(v) for k, v in sorted(per_label.items())}}


def redact_helper(model, docs, threshold):
    from german_clinical_deid import Deidentifier
    deid = Deidentifier(model, threshold=threshold)
    return [[(s, e) for s, e, _ in deid.spans(d["text"])] for d in docs], deid.threshold


def redact_pipeline(model, docs):
    from transformers import pipeline
    ner = pipeline("token-classification", model=model, aggregation_strategy="simple", stride=128)
    return [[(x["start"], x["end"]) for x in ner(d["text"])] for d in docs]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="LukasKunhardt/german-clinical-deid")
    ap.add_argument("--format", choices=["brat", "inception", "gate", "jsonl"], required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--layer", default="webanno.custom.PHI", help="inception: annotation type")
    ap.add_argument("--feature", default="kind", help="inception: feature with the label")
    ap.add_argument("--annotation-set", default="phi-annotations", help="gate: name of the annotation set with the gold labels")
    ap.add_argument("--direct", default="", help="comma-separated labels of direct identifiers (names, IDs, phone, street ...)")
    ap.add_argument("--ignore", default="", help="comma-separated labels to drop from the gold annotation")
    ap.add_argument("--readings", default="helper,pipeline")
    ap.add_argument("--threshold", type=float, default=None, help="helper reading: default is the threshold stored with the model")
    ap.add_argument("--out", default="partner_counts.json")
    a = ap.parse_args(argv)

    if a.format == "brat":
        docs = load_brat(a.path)
    elif a.format == "inception":
        docs = load_inception(a.path, a.layer, a.feature)
    elif a.format == "gate":
        docs = load_gate(a.path, a.annotation_set)
    else:
        docs = load_jsonl(a.path)
    ignore = {x for x in a.ignore.split(",") if x}
    for d in docs:
        d["spans"] = [s for s in d["spans"] if s[2] not in ignore]
    if not docs:
        sys.exit("no documents found")
    direct = {x for x in a.direct.split(",") if x}
    unknown = direct - {s[2] for d in docs for s in d["spans"]}
    if unknown:
        print("warning: --direct labels not in the data:", ", ".join(sorted(unknown)), file=sys.stderr)

    out = {"model": a.model, "format": a.format, "direct_labels": sorted(direct), "ignored_labels": sorted(ignore), "readings": {}}
    for reading in a.readings.split(","):
        if reading == "helper":
            red, thr = redact_helper(a.model, docs, a.threshold)
            out["readings"]["helper"] = {"threshold": thr, **score(docs, red, direct)}
        elif reading == "pipeline":
            out["readings"]["pipeline"] = score(docs, redact_pipeline(a.model, docs), direct)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
