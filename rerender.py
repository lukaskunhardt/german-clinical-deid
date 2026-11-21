"""Render every letter again with fresh fake identities. No LLM: the spans are known, so code swaps the values.

  python rerender.py train.jsonl train_r2.jsonl --n 2 --seed 1
  python rerender.py --download train_r2.jsonl        (reads data/train.jsonl of the published dataset)

Writes each original letter followed by n renders (doc_id + "_r1", "_r2", ...). Training on the originals plus
two renders gave fewer leaked names than three times as many epochs on the originals alone (20.11.2026).

Replaced: names (patient, doctor, relative, other), user codes, IDs, street, postcode + city (as a pair when
adjacent), phone and fax digits. Kept: dates, ages (a new age would contradict the unchanged birth date), titles,
hospitals, e-mail, URL, profession; their surface forms come from the writer and the case sheet.
- Within a letter the same original token always maps to the same new token ("Herr Müller" and "Hans Müller" stay one person).
- Names come from Faker in the locale mix of factory.py, places from data/places.tsv, streets from Faker de_DE.
  20 % of every pool (SHA-1(value) mod 5 == 0) is never used here, so a later test can use names and places the model never saw.
- Short IDs (up to 4 characters, e.g. rooms) keep their shape. Longer IDs: 50 % keep their shape with new characters,
  43 % take a new German format, 7 % a mixed-case random string (few on purpose: no evidence such strings occur in German letters).
- Light hospital-system noise: 4 % of name mentions in upper case ("MÜLLER"), 2 % of "Last, First" without the blank.
One random stream runs through the whole file, so the output depends on the input order (and on the Faker version, 40.5.4).
"""
import argparse
import hashlib
import json
import os
import random
import re
import string
import sys

from faker import Faker

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # also under python -I
from factory import DATA, LOCALES  # noqa: E402

NAMES = {"NAME_PATIENT", "NAME_DOCTOR", "NAME_RELATIVE", "NAME_EXT"}
SEPARATOR = r"\s+|,\s*|-"
INITIAL = r"[A-ZÄÖÜ]\."


def held_out(value):
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest(), 16) % 5 == 0


def pools(seed):
    first, last, streets = set(), set(), set()
    for loc, w in LOCALES:
        fk = Faker(loc)
        fk.seed_instance(seed)
        for _ in range(60 * w):
            first.add(fk.first_name_male()); first.add(fk.first_name_female()); last.add(fk.last_name())
    fk = Faker("de_DE")
    fk.seed_instance(seed)
    for _ in range(4000):
        streets.add(fk.street_name())
    places = [l.rstrip("\n").split("\t")[:2] for l in open(os.path.join(DATA, "places.tsv"), encoding="utf-8")]
    keep = lambda xs: sorted(x for x in xs if not held_out(x))  # noqa: E731
    return {"first": keep(first), "last": keep(last), "street": keep(streets),
            "place": [p for p in places if not held_out(p[0]) and not held_out(p[1])]}


def same_shape(s, rng):
    return "".join(rng.choice(string.digits) if c.isdigit() else rng.choice(string.ascii_uppercase) if c.isupper()
                   else rng.choice(string.ascii_lowercase) if c.islower() else c for c in s)


def new_id(rng):
    d = lambda n: "".join(rng.choice(string.digits) for _ in range(n))  # noqa: E731
    u = lambda n: "".join(rng.choice(string.ascii_uppercase) for _ in range(n))  # noqa: E731
    return rng.choice([
        lambda: d(rng.randint(6, 10)),
        lambda: u(1) + d(rng.randint(6, 9)),
        lambda: u(1) + d(9),                          # health insurance number
        lambda: f"{u(1)}{d(6)}-{d(1)}",               # pathology block number
        lambda: f"{u(1)}{d(2)}{u(2)}-{u(3)}{d(2)}",   # study code
        lambda: f"{rng.randint(2015, 2027)}-{d(6)}",  # year-prefixed case number
        lambda: f"{d(2)}.{d(4)}.{d(3)}",              # dotted lab number
        lambda: f"{u(2)}{d(rng.randint(5, 8))}",
        lambda: f"{d(3)} {d(3)} {d(3)}",
    ])()


def random_chain(rng):
    a = "".join(rng.choice(string.ascii_letters) for _ in range(rng.randint(4, 6)))
    b = "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(rng.randint(3, 5)))
    return a + rng.choice(["-", ""]) + b


def render(doc, p, rng):
    text = doc["text"]
    spans, last_end = [], -1
    for s in sorted(doc["spans"], key=lambda s: (s["start"], s["end"])):  # skip overlaps
        if s["start"] >= last_end and s["end"] > s["start"]:
            spans.append(s)
            last_end = s["end"]
    tokens, values, city_of_zip = {}, {}, {}
    out, new_spans, pos = [], [], 0
    for i, s in enumerate(spans):
        orig, label = text[s["start"]:s["end"]], s["label"]
        rep = orig
        if label in NAMES:
            parts = re.split(f"({SEPARATOR})", orig)
            words = [t for t in parts if t and not re.fullmatch(SEPARATOR, t)]
            res = []
            for t in parts:
                if not t or re.fullmatch(SEPARATOR, t):
                    res.append(t)
                elif re.fullmatch(INITIAL, t):
                    res.append(tokens.setdefault(t, rng.choice(string.ascii_uppercase) + "."))
                else:
                    if t not in tokens:  # first name: first word of "First Last", or last word of "Last, First"
                        is_first = len(words) > 1 and (t == words[0] and "," not in orig or "," in orig and t == words[-1])
                        tokens[t] = rng.choice(p["first"] if is_first else p["last"])
                    res.append(tokens[t].upper() if t.isupper() and len(t) > 1 else tokens[t])
            rep = "".join(res)
            if rng.random() < 0.04:
                rep = rep.upper()
            if ", " in rep and rng.random() < 0.02:
                rep = rep.replace(", ", ",", 1)
        elif label == "NAME_USERNAME":
            if orig not in values:
                if len(orig) <= 4 or rng.random() < 0.6:
                    values[orig] = same_shape(orig, rng)
                else:
                    values[orig] = rng.choice([lambda: rng.choice(p["last"])[:6].upper() + rng.choice(string.ascii_uppercase),
                                               lambda: rng.choice(p["first"])[0].lower() + rng.choice(p["last"]).lower()[:7],
                                               lambda: random_chain(rng)])()
            rep = values[orig]
        elif label == "ID":
            if orig not in values:
                x = 0.0 if len(orig) <= 4 else rng.random()
                values[orig] = same_shape(orig, rng) if x < 0.5 else new_id(rng) if x < 0.93 else random_chain(rng)
            rep = values[orig]
        elif label == "LOCATION_ZIP" and re.fullmatch(r"\d{5}", orig):
            if orig not in values:
                plz, city = rng.choice(p["place"])
                values[orig] = plz
                city_of_zip[plz] = city
            rep = values[orig]
            nxt = spans[i + 1] if i + 1 < len(spans) else None
            if nxt and nxt["label"] == "LOCATION_CITY" and nxt["start"] - s["end"] <= 2:  # "12345 Town": keep the pair real
                values.setdefault(text[nxt["start"]:nxt["end"]], city_of_zip[rep])
        elif label == "LOCATION_CITY":
            if orig not in values:
                values[orig] = rng.choice(p["place"])[1]
            rep = values[orig]
        elif label == "LOCATION_STREET":
            if orig not in values:
                m = re.search(r"\s+\d+\s*[a-zA-Z]?$", orig)
                values[orig] = rng.choice(p["street"]) + (m.group() if m else "")
            rep = values[orig]
        elif label in ("CONTACT_PHONE", "CONTACT_FAX"):
            if orig not in values:  # new digits, same format; a leading 0 or the country code stays
                values[orig] = "0" + re.sub(r"\d", lambda _: rng.choice(string.digits), orig[1:]) if orig[:1] == "0" \
                    else re.sub(r"(?<=\d{2})\d", lambda _: rng.choice(string.digits), orig)
            rep = values[orig]
        out.append(text[pos:s["start"]])
        start = sum(len(x) for x in out)
        out.append(rep)
        new_spans.append({**s, "start": start, "end": start + len(rep)})
        pos = s["end"]
    out.append(text[pos:])
    return "".join(out), new_spans


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", nargs="?", help="JSONL with doc_id, text, spans (e.g. data/train.jsonl of the dataset)")
    ap.add_argument("dst")
    ap.add_argument("--download", action="store_true", help="read the published dataset from the Hugging Face Hub")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.download:
        from huggingface_hub import hf_hub_download
        a.src = hf_hub_download("LukasKunhardt/synthetic-german-clinical-letters", "data/train.jsonl", repo_type="dataset")
    p = pools(a.seed)
    rng = random.Random(a.seed)
    print("pools after holding out 20 %:", {k: len(v) for k, v in p.items()})
    n = 0
    with open(a.dst, "w", encoding="utf-8") as fh:
        for line in open(a.src, encoding="utf-8"):
            d = json.loads(line)
            meta = d.get("metadata") or {}
            fh.write(json.dumps({**d, "metadata": {**meta, "render": 0, "source": d["doc_id"]}}, ensure_ascii=False) + "\n")
            for k in range(1, a.n + 1):
                text, spans = render(d, p, rng)
                fh.write(json.dumps({**d, "doc_id": f"{d['doc_id']}_r{k}", "text": text, "spans": spans,
                                     "metadata": {**meta, "render": k, "source": d["doc_id"]}}, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} letters, {a.n} renders each")


if __name__ == "__main__":
    main()
