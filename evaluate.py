"""Measure a de-identification model on GraSCCo or MDeID with the measures of their papers.

  python evaluate.py --model LukasKunhardt/german-clinical-deid --data grascco --split dev
  python evaluate.py --model LukasKunhardt/german-clinical-deid --data mdeid --path MDeID/german/test   # brat folder

Measures (micro over the whole set):
  char P/R/F1     characters inside gold spans against redacted characters, type ignored
                  (as in the SPARCLING / GraSCCo harness of Lehmann and Werthebach 2026)
  token P/R/F1    whitespace words: a word is positive if it overlaps a span, type ignored
  strict typed F1 a predicted span counts only if start, end and category equal a gold span (measure of the MDeID paper);
                  on MDeID my types are mapped to its 28 categories, my single ID type cannot match its 10 ID categories
  leak rate       share of documents with a direct identifier (name, ID, phone, fax, e-mail, street) in which at least one
                  direct identifier is less than half redacted; one leak can be enough to find the patient
  strict leaks    the same, but a single visible character of a direct identifier counts as a leak
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import re
import urllib.request
import zipfile

GRASCCO_ZIP = "https://zenodo.org/api/records/11502329/files/grascco_phi_annotation_json.zip/content"
DIRECT = {"NAME_PATIENT", "NAME_DOCTOR", "NAME_RELATIVE", "NAME_EXT", "NAME_USERNAME", "ID", "CONTACT_PHONE", "CONTACT_FAX",
          "CONTACT_EMAIL", "LOCATION_STREET"}
# My label -> MDeID category. ID, NAME_RELATIVE, NAME_EXT and NAME_TITLE have no MDeID counterpart.
TO_MDEID = {"NAME_PATIENT": "PATIENT", "NAME_DOCTOR": "DOCTOR", "NAME_USERNAME": "USERNAME", "PROFESSION": "PROFESSION",
            "LOCATION_STREET": "STREET", "LOCATION_CITY": "CITY", "LOCATION_ZIP": "ZIP", "LOCATION_COUNTRY": "COUNTRY",
            "LOCATION_HOSPITAL": "HOSPITAL", "LOCATION_ORGANIZATION": "ORGANIZATION", "AGE": "AGE", "DATE": "DATE",
            "CONTACT_PHONE": "PHONE", "CONTACT_FAX": "FAX", "CONTACT_EMAIL": "EMAIL", "CONTACT_URL": "URL"}
MDEID_DIRECT = {"PATIENT", "DOCTOR", "USERNAME", "MEDICALRECORD", "IDNUM", "ACCOUNT", "HEALTHPLAN", "LICENSE", "DEVICE", "VEHICLE",
                "BIOID", "SOCIALSECURITY", "IPADDRESS", "PHONE", "FAX", "EMAIL", "STREET"}


def grascco_part(doc_id):
    """Fixed split of GraSCCo into dev and test: letters of one family (Tupolev_1..4) stay together."""
    family = re.sub(r"_(?:\d+|[A-Z])$", "", doc_id)
    return "dev" if int(hashlib.sha1(family.encode()).hexdigest(), 16) % 2 == 0 else "test"


def load_grascco(root, split):
    if not glob.glob(os.path.join(root, "**", "*_phi.json"), recursive=True):
        req = urllib.request.Request(GRASCCO_ZIP, headers={"User-Agent": "python-urllib"})
        with urllib.request.urlopen(req, timeout=300) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(root)
    docs = []
    for path in sorted(glob.glob(os.path.join(root, "**", "*_phi.json"), recursive=True)):
        doc_id = os.path.basename(path).replace("_phi.json", "").replace(".txt", "")
        if split != "all" and grascco_part(doc_id) != split:
            continue
        fss = json.load(open(path, encoding="utf-8"))["%FEATURE_STRUCTURES"]
        text = next(x for x in fss if x.get("%TYPE") == "uima.cas.Sofa")["sofaString"]
        spans = sorted((x["begin"], x["end"], x.get("kind") or "PHI") for x in fss
                       if x.get("%TYPE") == "webanno.custom.PHI" and x.get("begin") is not None and x.get("end", 0) > x["begin"])
        docs.append({"id": doc_id, "text": text, "spans": spans, "direct": [s for s in spans if s[2] in DIRECT]})
    return docs


def load_brat(folder):
    """MDeID in brat format with its own 28 categories. Discontinuous spans run from the first start to the last end."""
    docs = []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        text = open(path, encoding="utf-8").read()
        spans = []
        for line in open(path[:-4] + ".ann", encoding="utf-8"):
            if line.startswith("T"):
                parts = line.rstrip("\n").split("\t")[1].replace(";", " ").split()
                nums = [int(x) for x in parts[1:] if x.isdigit()]
                if nums:
                    spans.append((min(nums), max(nums), parts[0]))
        spans.sort()
        docs.append({"id": os.path.basename(path)[:-4], "text": text, "spans": spans,
                     "direct": [s for s in spans if s[2] in MDEID_DIRECT]})
    return docs


def prf(tp, n_pred, n_gold):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def score(docs, preds):
    """docs: gold; preds: {id: {"spans": [(s, e)], "entities": [(s, e, label)]}} with labels already in the gold category space."""
    c = dict.fromkeys(["c_gold", "c_pred", "c_both", "t_gold", "t_pred", "t_both", "s_tp", "s_pred", "s_gold", "leak", "leak_strict", "n_direct"], 0)
    for d in docs:
        n, pr = len(d["text"]), preds[d["id"]]
        gold, red = bytearray(n), bytearray(n)
        for s, e, *_ in d["spans"]:
            gold[s:e] = b"\x01" * (e - s)
        for s, e in pr["spans"]:
            red[s:e] = b"\x01" * (e - s)
        c["c_gold"] += sum(gold); c["c_pred"] += sum(red); c["c_both"] += sum(a & b for a, b in zip(gold, red))
        for m in re.finditer(r"\S+", d["text"]):
            g, p = any(gold[m.start():m.end()]), any(red[m.start():m.end()])
            c["t_gold"] += g; c["t_pred"] += p; c["t_both"] += g and p
        g_set, p_set = set(d["spans"]), set(pr["entities"])
        c["s_tp"] += len(g_set & p_set); c["s_pred"] += len(p_set); c["s_gold"] += len(g_set)
        if d["direct"]:
            c["n_direct"] += 1
            c["leak"] += any(sum(red[s:e]) / (e - s) < 0.5 for s, e, _ in d["direct"])
            c["leak_strict"] += any(sum(red[s:e]) < e - s for s, e, _ in d["direct"])
    cp, cr, cf = prf(c["c_both"], c["c_pred"], c["c_gold"])
    tp_, tr, tf = prf(c["t_both"], c["t_pred"], c["t_gold"])
    sp, sr, sf = prf(c["s_tp"], c["s_pred"], c["s_gold"])
    return {"n_docs": len(docs), "char_p": cp, "char_r": cr, "char_f1": cf, "token_p": tp_, "token_r": tr, "token_f1": tf,
            "strict_typed_p": sp, "strict_typed_r": sr, "strict_typed_f1": sf,
            "docs_with_direct": c["n_direct"], "leaks": c["leak"], "leak_rate": c["leak"] / max(1, c["n_direct"]),
            "leaks_strict": c["leak_strict"], "leak_rate_strict": c["leak_strict"] / max(1, c["n_direct"])}


def predict(deid, docs, to_gold=None):
    out = {}
    for d in docs:
        scores = deid.read(d["text"])
        ents = [(x["start"], x["end"], x["label"]) for x in deid.entities(d["text"], scores=scores)]
        if to_gold is not None:
            ents = [(s, e, to_gold.get(k, k)) for s, e, k in ents]
        out[d["id"]] = {"spans": [(s, e) for s, e, _ in deid.spans(d["text"], scores=scores)], "entities": ents}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LukasKunhardt/german-clinical-deid")
    ap.add_argument("--data", choices=["grascco", "mdeid"], default="grascco")
    ap.add_argument("--path", default="grascco", help="GraSCCo download folder, or MDeID brat folder")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="dev", help="GraSCCo only")
    ap.add_argument("--threshold", type=float, default=None, help="redaction threshold (default: the one stored with the model)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from german_clinical_deid import Deidentifier

    docs = load_grascco(a.path, a.split) if a.data == "grascco" else load_brat(a.path)
    deid = Deidentifier(a.model, threshold=a.threshold)
    m = score(docs, predict(deid, docs, TO_MDEID if a.data == "mdeid" else None))
    m.update(model=a.model, data=a.data, split=a.split if a.data == "grascco" else a.path, threshold=deid.threshold,
             type_threshold=deid.type_threshold)
    print(json.dumps(m, indent=1))
    if a.out:
        json.dump(m, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
