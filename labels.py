"""Label the letters from factory.py and filter them into a training corpus.

The writer gets no markup instructions, so every label comes from code. Code knows every identity value on
the case sheet (names, addresses, phone numbers, identifiers, ages) and finds them in the text by exact match.
Dates, ages, titles and ward or room numbers are found by pattern. Labels follow the GraSCCo/GeMTeX scheme
(18 kinds) with a tier: DIRECT identifies a person on its own, QUASI only in combination.

A letter is rejected when it is empty, too short (under 500 characters), contains a foreign script, leftover
markup, draft placeholders, Swiss spellings or US ICD-10-CM codes, or when a first name from the sheet also
occurs as an ordinary lower-case word (then code cannot tell name from word).

Two more filters made the shipped corpus (3,144 raw records -> 3,023 labelled -> 2,561 -> 2,550):
1. Two LLM judges (Gemini 4 Argon via the paid API and gpt-oss-120b) read each letter with every label masked
   and list anything that could still identify someone. A finding from either judge drops the letter.
2. Duplicate check: a letter that shares 3 or more word 10-grams with one test letter (GraSCCo, SPARCLING) is dropped.

  python labels.py runs/s21/raw.jsonl corpus.jsonl --judges google/gemini-4-argon,openai/gpt-oss-120b --test-sets grascco.jsonl,sparcling.jsonl
"""
import argparse
import collections
import concurrent.futures as cf
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # factory.py, also under python -I

MONTH = r"(?:Jan(?:uar|\.)?|Feb(?:ruar|\.)?|März|Mrz\.?|Apr(?:il|\.)?|Mai|Jun[i.]?|Jul[i.]?|Aug(?:ust|\.)?|Sep(?:tember|t?\.)?|Okt(?:ober|\.)?|Nov(?:ember|\.)?|Dez(?:ember|\.)?)"
TITLE = re.compile(r"(?<![\w-])(?:(?:Ao\.\s?)?(?:Univ\.-)?Prof\.|PD(?=\s)|Priv\.-Doz\.|Dr\.|Dipl\.-\w+\.?|Doz\.)(?:\s?(?:Dr\.|med\.|habil\.|univ\.|dent\.|phil\.|rer\.\s?nat\.|rer\.\s?medic\.|h\.\s?c\.|mult\.|Dipl\.-\w+\.?))*")
TITLE_AFTER = re.compile(r"(?<=[a-zäöüß])(?:,\s?|\s)((?:MD|PhD|MBA|MSc|FEBU|FRCS|M\.D\.)(?:(?:,\s?|\s)(?:PhD|MSc|MBA|FEBU))*)(?![\w.])")
DATES = [
    re.compile(r"(?<![\d-])(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])(?![\d-])"),  # ISO, copied from the sheet
    re.compile(r"(?<![\d.,])(?:0?[1-9]|[12]\d|3[01])\.\s?(?:0?[1-9]|1[0-2])\.(?:\s?(?:19|20)\d{2}|\d{2})?(?![\d])"),
    re.compile(r"(?<!\d)(?:(?:0?[1-9]|[12]\d|3[01])\.\s?)?" + MONTH + r"\s(?:19|20)\d{2}(?!\d)"),
    re.compile(r"(?<![\d/.,])(?:0?[1-9]|1[0-2])/(?:19|20)\d{2}(?![\d/])"),
    re.compile(r"(?<![\d/.,])(?:0?[1-9]|[12]\d|3[01])/(?:0?[1-9]|1[0-2])/(?:(?:19|20)\d{2}|\d{2})(?![\d/°])"),
]
MMYY = re.compile(r"(?<![\d/.,])(?:0?[1-9]|1[0-2])/\d{2}(?![\d/,.]\d|\d)")
YEAR = re.compile(r"(?<![\d/.,-])(?:19[3-9]\d|20[0-2]\d)(?![\d/]|[.,]\d|\s?(?:ml|mg|µg|g\b|µl|/µl|/nl|U\b|IE|I\.E\.|kcal|m\b|mm|pg|ng|Hz|kg))")
# "NRS 3/10" or "Kraftgrad 4/5" is a scale, "pN1 (1/22)" a lymph node ratio, not a date (label sample, 19.11.2026)
SCALE = re.compile(r"(?<![A-Za-zÄÖÜäöüß])(?:NRS|VAS|Skala|Score|BORG|Borg|Kraftgrad|KG|Punkte|GCS|Apgar|APGAR|Visus|Stärke|Grad)\b[^\n]{0,25}$")
NOT_MMYY = re.compile(r"\bp?N\d[a-c]?\s*\($|Lymphknoten\w*\s*\(?$|LK\s*\(?$|[Ss]chmerz\w*[^\n]{0,30}$|Intensität[^\n]{0,15}$")
# ward, room and theatre numbers are IDs in the GeMTeX guideline, whatever the sheet says
WARD = re.compile(r"(?<!\w)(?:(?:Normal|Intensiv|Wach|Privat|Kinder|Peripher|Überwachungs|Aufnahme|Kurzlieger|Wochen|Palliativ|Infektions|Isolier|Säuglings|Frühgeborenen|Beleg|Tages)?[Ss]tation|Stat\.|(?:Isolier|Einzel|Doppel|Patienten|Mehrbett)?[Zz]immer|Zi\.|Zi|Zimmer-Nr\.|Raum|Bett|OP-Saal|OP)(?![A-Za-zÄÖÜäöüß])\s?:?\s?((?:[A-Z]{1,4}[- ]?)?\d{1,4}[A-Za-z]?|[A-Z](?![\w-])|[IVX]{1,4}(?![\w-]))(?![\w-]|\.\d)")
NUMBER_WORD = r"(?:ein|zwei|drei|vier|fünf|sechs|sieben|acht|neun|zehn|elf|zwölf|dreizehn|vierzehn|fünfzehn|sechzehn|siebzehn|achtzehn|neunzehn|zwanzig|dreißig|vierzig|fünfzig|sechzig|siebzig|achtzig|neunzig|(?:ein|zwei|drei|vier|fünf|sechs|sieben|acht|neun)und(?:zwanzig|dreißig|vierzig|fünfzig|sechzig|siebzig|achtzig|neunzig))"
AGE_WORD = re.compile(r"(?i)(?<![a-zäöüß])" + NUMBER_WORD + r"(?=[- ]?jährig)")
AGE = re.compile(r"(?<![\d,.])\d{1,3}(?=[- ]?(?:jährig|jähr\.|-j\.|J\.(?!\w)|Jahre alt|-Jährige|j\.(?!\w)|LJ))|(?<=im Alter von )\d{1,3}|(?<=\bmit )\d{1,3}(?= Jahren\b)")
DIRECT = {"NAME_PATIENT", "NAME_DOCTOR", "NAME_RELATIVE", "NAME_EXT", "NAME_USERNAME", "ID", "CONTACT_PHONE", "CONTACT_FAX", "CONTACT_EMAIL", "LOCATION_STREET"}
REJECT = [  # (reason, pattern); the judges caught each of these giving a synthetic letter away
    ("foreign script", re.compile(r"[Ѐ-ӿ一-鿿぀-ヿ؀-ۿ]")),
    ("markup", re.compile(r"<[/A-Za-zÄÖÜ]")),
    ("draft", re.compile(r"\[[^\]\n]{0,40}\]|\bXX+\b|Platzhalter|\bEntwurf\b|\bAnmerkung:|\(Name\)|\bTODO\b|\.\.\.\s*$", re.M)),
    ("Swiss", re.compile(r"\b(?:Strasse|gross|grosse[nmrs]?|grösser\w*|Grösse|gemäss|regelmässig\w*|schliesslich|Fuss|ausserdem|ausser|Spital|Austrittsbericht|Hausärztin/Hausarzt)\b")),
    ("ICD-10-CM", re.compile(r"\b[A-Z]\d{2}\.\d{0,3}X\w*|\b[A-Z]\d{2}\.\d{3,4}[A-Z]?\b|initial encounter")),
]


def people(case):
    w = case.get("weitere_personen") or {}
    out = [(case["patient"], "NAME_PATIENT")]
    out += [(p, "NAME_DOCTOR") for p in [*case["personal"], case.get("hausarzt"), w.get("vorbehandler")] if p]
    return out + [(p, "NAME_RELATIVE") for p in w.get("angehoerige") or []] + [(p, "NAME_EXT") for p in [w.get("sonstige")] if p]


def find_spans(case, text):
    """All spans code can prove from the case sheet or from patterns, in a fixed order (on ties the first one wins).
    Returns (spans, problems)."""
    spans, problems = [], []

    def each(value, label):
        if value:
            spans.extend((m.start(), m.end(), label) for m in re.finditer(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text))

    def inside(m):
        return any(s <= m.start() and m.end() <= e for s, e, _ in spans)

    for p, label in people(case):
        v, n = re.escape(p["vorname"]), re.escape(p["nachname"])
        spans.extend((m.start(), m.end(), label) for m in re.finditer(rf"(?<!\w)(?:{v}\s{n}|{n},\s?{v}|{v[0]}\.\s?{n})(?!\w)", text))
        spans.extend((m.start(), m.end(), label) for m in list(re.finditer(rf"(?<!\w){n}s?(?!\w)", text)) if not inside(m))
        # a first name alone (children are often called by first name) is certain unless it is also a lower-case word here
        common_word = re.search(rf"(?<!\w){re.escape(p['vorname'].lower())}(?!\w)", text)
        for m in re.finditer(rf"(?<!\w){v}s?(?!\w)", text):
            if inside(m):
                continue
            if common_word:
                problems.append(f"first name also a word: {text[max(0, m.start() - 15):m.end() + 5]!r}")
            else:
                spans.append((m.start(), m.end(), label))
    gp, clinic, ids = case.get("hausarzt") or {}, case["klinik"], case["kennungen"]
    prev = (case.get("weitere_personen") or {}).get("vorbehandler") or {}
    for a in (case["patient"], clinic, gp):
        each(a.get("strasse"), "LOCATION_STREET")
        each(a.get("plz"), "LOCATION_ZIP")
        each(a.get("ort"), "LOCATION_CITY")
    each(clinic["name"], "LOCATION_HOSPITAL")
    if clinic["name"].upper() != clinic["name"]:
        each(clinic["name"].upper(), "LOCATION_HOSPITAL")
    each(prev.get("klinik"), "LOCATION_HOSPITAL")
    each(prev.get("ort"), "LOCATION_CITY")
    for value, label in ((clinic.get("telefon"), "CONTACT_PHONE"), (gp.get("telefon"), "CONTACT_PHONE"), (case["patient"].get("telefon"), "CONTACT_PHONE"),
                         (gp.get("fax"), "CONTACT_FAX"), (clinic.get("fax"), "CONTACT_FAX"), (clinic.get("mail"), "CONTACT_EMAIL"), (clinic.get("url"), "CONTACT_URL")):
        each(value, label)
    for k, v in ids.items():
        if k not in ("station", "zimmer", "op_saal", "bett"):  # bed "1" to "4" matched every lone digit (705 false IDs in a label sample)
            each(v, "ID")
    spans.extend((m.start(1), m.end(1), "ID") for m in WARD.finditer(text))
    each(case["kuerzel"], "NAME_USERNAME")
    user = (case.get("benutzerkennung") or {}).get("wert")
    if user and len(user) >= 4:
        each(user, "NAME_USERNAME")
    job = case["patient"].get("beruf")  # writers often use the female form of the profession
    if job:
        forms = sorted({job, job.replace("Arzt", "Ärzt"), job.replace("arzt", "ärzt")}, key=len, reverse=True)
        rx = r"(?<!\w)((?:" + "|".join(re.escape(f) for f in forms) + r")(?:in|innen)?)(?!\w)"
        spans.extend((m.start(1), m.end(1), "PROFESSION") for m in re.finditer(rx, text))
    line = case.get("stationsangabe") or {}
    if line.get("zeile"):  # values of the copied ward line are IDs by their place in it, whatever the word before
        rx, n = "", 0
        for t in re.split(r"(\{[ezsb]\})", line["vorlage"]):
            if t in ("{s}", "{z}", "{b}"):
                rx += "(" + re.escape(line["werte"][{"{s}": "station", "{z}": "zimmer", "{b}": "bett"}[t]]) + ")"
                n += 1
            else:
                rx += re.escape(line["einheit"]) if t == "{e}" else re.escape(t).replace("\\t", "\\s+").replace("\\ ", "\\s+")
        spans.extend((m.start(g), m.end(g), "ID") for m in re.finditer(rx, text) for g in range(1, n + 1))
    ward = ids.get("station") or ""
    if re.search(r"[A-Za-z]", ward) and len(ward) >= 3:  # ward codes like "CH-1" or "ITS8" are unique enough to match anywhere
        each(ward, "ID")
    age = case["patient"].get("alter")
    if age is not None:
        spans.extend((m.start(), m.end(), "AGE") for m in re.finditer(rf"(?<![\d,.]){age}(?=\s?(?:Jahre\b|J\.|-?jährig|-j\.|-Jährig))", text))
    spans.extend((m.start(), m.end(), "NAME_TITLE") for m in TITLE.finditer(text))
    spans.extend((m.start(1), m.end(1), "NAME_TITLE") for m in TITLE_AFTER.finditer(text))
    for rx in DATES:
        spans.extend((m.start(), m.end(), "DATE") for m in rx.finditer(text)
                     if "/" not in m.group() or not SCALE.search(text[max(0, m.start() - 45):m.start()]))
    for rx in (MMYY, YEAR):  # with no model tags to lean on, every year-like number counts as a date
        for m in rx.finditer(text):
            before = text[max(0, m.start() - 45):m.start()]
            if rx is MMYY and (SCALE.search(before) or NOT_MMYY.search(before)):
                continue
            spans.append((m.start(), m.end(), "DATE"))
    spans.extend((m.start(), m.end(), "AGE") for rx in (AGE, AGE_WORD) for m in rx.finditer(text))
    return spans, problems


def finalize(rec):
    """Raw record from factory.py -> (corpus row or None, reasons for rejection)."""
    text, case = rec.get("text"), rec.get("case")
    if not text:
        return None, ["empty"]
    problems = [name for name, rx in REJECT if rx.search(text)]
    if len(text) < 500:
        problems.append("too short")
    spans, flags = find_spans(case, text)
    if problems or flags:
        return None, problems + flags
    # overlaps: the longer span wins; on a tie the earlier one (stable sort), identical spans collapse
    taken, final = bytearray(len(text)), []
    for s, e, label in sorted(dict.fromkeys(spans), key=lambda x: (-(x[1] - x[0]), x[0])):
        if e > s and not any(taken[s:e]):
            final.append((s, e, label))
            taken[s:e] = b"\x01" * (e - s)
    return {"doc_id": rec["doc_id"], "text": text,
            "spans": [{"start": s, "end": e, "label": l, "tier": "DIRECT" if l in DIRECT else "QUASI"} for s, e, l in sorted(final)],
            "metadata": {"model": rec["model"], "temperature": rec.get("temperature"), "document_type": case["dokumentart"],
                         "specialty": case["fachgebiet"], "main_diagnosis": case["hauptdiagnose"]}}, []

# -----------------------------------------------------------------------------
# filter 1: two LLM judges on the masked letter


JUDGE_PROMPT = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "judge_prompt.md"), encoding="utf-8").read()
# judges sometimes report department names or a lone month; neither is identifying under the guideline
GENERIC = re.compile(r"^(?:(?:Klinik|Abteilung|Zentrum|Institut|Praxis|Station) für [A-ZÄÖÜ][\w\- ,und]*|(?:Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember)|\[[A-Z_]+\][-\w.]*|(?:Herrn?|Frau|Dr\.|Prof\.)\s*\[[A-Z_]+\])$")


def mask(row):
    out, pos = [], 0
    for s in sorted(row["spans"], key=lambda s: s["start"]):
        out += [row["text"][pos:s["start"]], f"[{s['label']}]"]
        pos = s["end"]
    return "".join(out) + row["text"][pos:]


def judge(key, model, row):
    """Findings the judge lists that occur verbatim in the masked letter; None when the answer cannot be read."""
    from factory import call
    masked = mask(row)
    try:
        raw, _, _ = call(key, model, JUDGE_PROMPT, masked, temperature=0.0, max_tokens=4000)
        found = json.loads(re.search(r"\{.*\}", raw, re.S).group())["funde"]
    except Exception:  # noqa: BLE001, API error or unreadable answer: the letter does not pass
        return None
    found = [str(f.get("text", "")).strip() if isinstance(f, dict) else "" for f in found]
    return [t for t in found if len(t) >= 2 and t in masked and not re.fullmatch(r"\[[A-Z_]+\]", t) and not GENERIC.search(t)]

# -----------------------------------------------------------------------------
# filter 2: no near-copy of a test letter


STOP = {"mg", "ml", "µg", "g", "i", "e", "v", "p", "o", "s", "c", "h", "ie", "mmol", "l", "dl", "kg", "mmhg", "min"}


def grams(text, n=10):
    """Word 10-grams without numbers and units, so medication schemes ("Ramipril 5 mg 1-0-0") do not count as overlap."""
    w = [t for t in re.findall(r"[a-zäöüß0-9]+", text.lower()) if not t.isdigit() and t not in STOP]
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def near_copies(rows, test_texts, max_shared=3):
    """doc_ids of rows sharing max_shared or more 10-grams with one test letter. A 10-gram found in two or more
    test letters is a stock phrase and does not count."""
    index = collections.defaultdict(set)
    for k, t in enumerate(test_texts):
        for g in grams(t):
            index[g].add(k)
    index = {g: ks for g, ks in index.items() if len(ks) == 1}
    flagged = set()
    for r in rows:
        hits = collections.Counter(k for g in grams(r["text"]) for k in index.get(g, ()))
        if hits and hits.most_common(1)[0][1] >= max_shared:
            flagged.add(r["doc_id"])
    return flagged


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("raw", help="raw.jsonl from factory.py (several files comma-separated)")
    ap.add_argument("out")
    ap.add_argument("--judges", default="", help="comma-separated OpenRouter ids; a finding from any judge drops the letter")
    ap.add_argument("--test-sets", default="", help="comma-separated JSONL files with a 'text' field per test letter")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    rows, reasons = [], collections.Counter()
    for line in (l for path in a.raw.split(",") for l in open(path, encoding="utf-8")):
        rec = json.loads(line)
        row, why = finalize(rec) if "error" not in rec else (None, ["API error"])
        reasons.update(r.split(":")[0] for r in why)
        rows += [row] if row else []
    print(f"labelled {len(rows)}, rejected: {dict(reasons.most_common())}")
    for model in filter(None, a.judges.split(",")):
        key = os.environ["OPENROUTER_API_KEY"]
        with cf.ThreadPoolExecutor(a.workers) as ex:
            verdicts = list(ex.map(lambda r: judge(key, model, r), rows))
        rows = [r for r, v in zip(rows, verdicts) if v == []]
        print(f"after judge {model}: {len(rows)}")
    if a.test_sets:
        tests = [json.loads(l)["text"] for path in a.test_sets.split(",") for l in open(path, encoding="utf-8")]
        bad = near_copies(rows, tests)
        rows = [r for r in rows if r["doc_id"] not in bad]
        print(f"after duplicate check against {len(tests)} test letters: {len(rows)}")
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(f"labels: {dict(collections.Counter(s['label'] for r in rows for s in r['spans']).most_common())}")


if __name__ == "__main__":
    main()
