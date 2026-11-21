"""Write synthetic German clinical letters with an open-weight LLM.

Code draws a fake case sheet for every letter: patient, staff, relatives, clinic, addresses,
phone numbers, identifiers, dates, document type, diagnosis and a few style choices. An optional
first call lets an open model plan the clinical core (secondary diagnoses, medication, procedures).
The writer model then gets the sheet and writes the letter as plain text, without any markup.
labels.py later finds every identifying value in the text, because code knows the sheet.

Data in data/: hospital names and area codes from Wikidata (CC0), postcodes and places from
GeoNames (CC BY 4.0), first names, last names and streets from Faker (MIT), and a list of
diagnoses by department written by gpt-oss-120b (Apache 2.0). No text comes from real patients.

The shipped corpus used DeepSeek V4.1 Flash and GLM-5.3 (MIT licences, no conditions on outputs),
temperature 0.8, reasoning effort "low". The largest batch (seed 21, 1,000 sheets) was:

  python factory.py --models deepseek/deepseek-v4.1-flash-1017 --seed 21 --n 1000 --out runs/s21 \
      --core-model deepseek/deepseek-v4.1-flash-1017 --providers pearl-research/fp8,deepinfra/fp8,stepfun/fp8 --workers 32

With faker==40.5.4 this code redraws the 1,000 case sheets of seed 21 exactly. Earlier batches of
the corpus (seeds 7, 9, 14, 20) came from earlier revisions of the sheet and are not redrawn.
Needs OPENROUTER_API_KEY in the environment. Output: <out>/raw.jsonl, one record per letter.
"""
import argparse
import collections
import concurrent.futures as cf
import datetime as dt
import json
import os
import random
import re
import time
import unicodedata
import urllib.request

from faker import Faker

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
API = "https://openrouter.ai/api/v1/chat/completions"
# cheapest provider first, but no 4-bit quantised endpoints; never the model maker's own API (its terms were not read)
PROVIDER = {"sort": "price", "quantizations": ["fp8", "fp16", "bf16", "fp32", "unknown"]}
MAKER_API = {"z-ai/": ["Z.AI"], "deepseek/": ["DeepSeek"]}

# names are not only German; weights roughly follow the population of Germany by origin
LOCALES = [("de_DE", 70), ("tr_TR", 8), ("pl_PL", 5), ("it_IT", 3), ("hr_HR", 3), ("ro_RO", 3), ("cs_CZ", 2),
           ("en_US", 2), ("de_AT", 2), ("nl_NL", 1), ("fr_FR", 1)]
# rough share of inpatient cases per department in German hospitals (Destatis), estimated
DEPARTMENTS = {
    "Kardiologie": 14, "Gastroenterologie": 10, "Pneumologie": 6, "Nephrologie": 3, "Hämatologie und Onkologie": 6,
    "Endokrinologie und Diabetologie": 2, "Neurologie": 7, "Allgemein- und Viszeralchirurgie": 11,
    "Unfallchirurgie und Orthopädie": 12, "Gefäßchirurgie": 2, "Urologie": 6, "Gynäkologie und Geburtshilfe": 9,
    "Pädiatrie": 6, "Dermatologie": 1.5, "Hals-Nasen-Ohren-Heilkunde": 3, "Psychiatrie und Psychotherapie": 5,
    "Geriatrie": 3, "Neurochirurgie": 1.5, "Rheumatologie": 1, "Zentrale Notaufnahme": 4,
}
SURGICAL = {"Allgemein- und Viszeralchirurgie", "Unfallchirurgie und Orthopädie", "Gefäßchirurgie",
            "Urologie", "Gynäkologie und Geburtshilfe", "Hals-Nasen-Ohren-Heilkunde", "Neurochirurgie"}
# (document type, weight, stay in days as (min, max); (0, 0) means a single visit)
DOC_TYPES = [("Entlassbrief nach stationärem Aufenthalt", 40, (2, 21)), ("Ambulanzbrief", 18, (0, 0)),
             ("vorläufiger Kurzarztbrief bei Entlassung", 10, (1, 9)), ("Befundbericht einer Untersuchung", 10, (0, 0)),
             ("Verlegungsbrief an eine andere Klinik", 6, (1, 14)), ("Konsilbericht", 6, (0, 0)), ("Operationsbericht", 10, (0, 0))]
# German phrases of the case sheet (titles, style choices, hard negatives); the writer sees them verbatim
PHRASES = json.load(open(os.path.join(DATA, "sheet_phrases.json"), encoding="utf-8"))
ROLES = [("Chefarzt", "Chefärztin", PHRASES["titles_head"]), ("Oberarzt", "Oberärztin", PHRASES["titles_senior"]),
         ("Stationsarzt", "Stationsärztin", PHRASES["titles_junior"]),
         ("Assistenzarzt", "Assistenzärztin", PHRASES["titles_senior"] + PHRASES["titles_junior"])]
CLINIC_FIT = [
    (r"Kinder|Jugend", {"Pädiatrie"}), (r"Psychiatr|Seelische|Nerven|Psychotherap", {"Psychiatrie und Psychotherapie", "Neurologie"}),
    (r"Herz(?!og|berg)", {"Kardiologie", "Gefäßchirurgie"}), (r"Frauen", {"Gynäkologie und Geburtshilfe"}),
    (r"Ortho|Sport|Wirbel|Hand", {"Unfallchirurgie und Orthopädie"}), (r"Lunge", {"Pneumologie"}),
    (r"Neurolog", {"Neurologie", "Neurochirurgie"}), (r"Haut", {"Dermatologie"}), (r"Rheuma", {"Rheumatologie"}),
    (r"Geriatr", {"Geriatrie"}), (r"Augen|Reha|Sanatorium|Heilstätte|Heilanstalt|Krebsforschung|Fachklinik|Fachkrankenhaus", set()),
]
# historic buildings in the open list ("Heil- und Pflegeanstalt Ansbach: Sporthalle") gave letters away; "Fachklinik Schloss ..." stays
HISTORIC = re.compile(r"Spital|Kirche|Gebäude|Haus des|Sporthalle|Pflegeanstalt|Anstalt|Heilstätte|Sanatorium|[Ee]hemal|Lazarett|Siechen|"
                      r"^(?!.*(?:Klinik|Krankenhaus|Klinikum)).*(?:Stift\b|Kloster|Schloss)")
# Notes on some phrase lists, kept here because they explain odd choices:
# - "Ad 1)" was removed from the layouts on 19.11.2026: 0 % of real letters, 20 % of ours.
# - salutation, closing, contact line and opening sentence are drawn by code (19.11.2026): masked 5-grams shared between
#   letters were 2.4 % in real letters and 6.6 % in ours, led by "Telefon ... Fax ..." and "wir berichten über die stationäre".
# - lookalikes (eponyms, drug brands, place-like terms, numbers that are not IDs) and hard negatives were written for this
#   project; none is taken from a test letter.

# -----------------------------------------------------------------------------
# open lists


def load_lists():
    hospitals = [l.rstrip("\n").split("\t") for l in open(os.path.join(DATA, "hospitals.tsv"), encoding="utf-8")]
    hospitals = [k for k in hospitals if not re.search(r"Ehemalig|ehemalig|Lazarett|Hospital zum|Siechen|Armen|[(#\d]", k[0])]
    places = [l.rstrip("\n").split("\t") for l in open(os.path.join(DATA, "places.tsv"), encoding="utf-8")]
    diagnoses = [json.loads(l) for l in open(os.path.join(DATA, "diagnoses.jsonl"), encoding="utf-8")]
    area_codes = {}
    for line in open(os.path.join(DATA, "area_codes.tsv"), encoding="utf-8"):
        if not line.startswith("#"):
            k, v = line.rstrip("\n").split("\t")
            area_codes[k] = v.split(",")
    return hospitals, places, diagnoses, area_codes


_FAKERS = {}


def fake(locale, rng):
    fk = _FAKERS.setdefault(locale, Faker(locale))
    fk.seed_instance(rng.randint(0, 2**31))
    return fk


def person(rng, sex=None):
    sex = sex or rng.choice("mw")
    loc = rng.choices([l for l, _ in LOCALES], [w for _, w in LOCALES])[0]
    fk = fake(loc, rng)
    first = fk.first_name_male() if sex == "m" else fk.first_name_female()
    last = fk.last_name()
    if loc == "pl_PL" and sex == "w" and last.endswith("ski"):
        last = last[:-1] + "a"
    return {"vorname": first, "nachname": last, "geschlecht": sex}


def relative(rng, age):
    rel = rng.choice([("Mutter", "w"), ("Vater", "m")] if age < 25 else [("Ehefrau", "w"), ("Ehemann", "m"), ("Tochter", "w"), ("Sohn", "m"),
                     ("Schwester", "w"), ("Bruder", "m"), ("Lebensgefährtin", "w")] if age > 45 else
                     [("Ehefrau", "w"), ("Ehemann", "m"), ("Mutter", "w"), ("Schwester", "w"), ("Bruder", "m"), ("Lebensgefährte", "m")])
    return {"beziehung": rel[0], **person(rng, rel[1])}


def address(fk, rng, places):
    plz, ort, state = rng.choice(places)
    nr = str(rng.randint(1, 140)) + (rng.choice("abc") if rng.random() < 0.08 else "")
    return {"strasse": f"{fk.street_name()} {nr}", "plz": plz, "ort": ort, "bundesland": state}


def area_code(rng, plz, ort, places, area_codes):
    """Area code that fits the town: Wikidata, else the most common code among towns with the same postcode prefix."""
    if ort in area_codes:
        return rng.choice(area_codes[ort])
    for n in (3, 2):
        cand = collections.Counter(area_codes[o[1]][0] for o in places if o[0][:n] == plz[:n] and o[1] in area_codes)
        if cand:
            return cand.most_common(1)[0][0]
    return "0" + str(rng.randint(201, 9999))


def phone(rng, code, switchboard=None):
    """Phone numbers in many real formats. With a switchboard number, a direct line of the same switchboard."""
    base = switchboard or str(rng.randint(20, 9999) if len(code) > 4 else rng.randint(200, 99999))
    ext = rng.choice([str(rng.randint(0, 9)), str(rng.randint(10, 99)), str(rng.randint(100, 9999))])
    c0 = code[1:]
    return rng.choice([f"{code} {base}-{ext}", f"{code}/{base}-{ext}", f"{code} / {base} {ext}", f"+49 {c0} {base}-{ext}",
                       f"+49 (0){c0} {base}-{ext}", f"({code}) {base}-{ext}", f"{code}-{base}-{ext}", f"{code} {base}{ext}",
                       f"{code}.{base}.{ext}", f"+49-{c0}-{base}{ext}"]), base


def slugify(name):
    return re.sub(r"[^a-z]+", "-", name.lower().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss"))[:24].strip("-")


def ascii_fold(name):
    n = name.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("Ä", "Ae").replace("Ö", "Oe").replace("Ü", "Ue").replace("ß", "ss")
    return re.sub(r"[^A-Za-z]", "", unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode())

# -----------------------------------------------------------------------------
# the case sheet
#
# The sheet grew in four passes during development (a full sheet, then thinning to the density of real letters,
# then local area codes and more ID forms, then fixes found by LLM judges). Each pass draws from one random stream,
# and several values drawn early are replaced later. Those draws stay, so a seed gives the same sheets as in the
# shipped corpus. Additions made late in development use their own random streams for the same reason.


def make_case(i, seed, hospitals, places, diagnoses, area_codes):
    rng = random.Random(f"{seed}:{i}")
    fk = Faker("de_DE")
    fk.seed_instance(rng.randint(0, 2**31))
    dept = rng.choices(list(DEPARTMENTS), weights=list(DEPARTMENTS.values()))[0]
    d = rng.choice([x for x in diagnoses if x["fach"] == dept] or diagnoses)
    types = [t for t in DOC_TYPES if t[0] != "Operationsbericht" or d["fach"] in SURGICAL]
    icd = re.sub(r"^([A-Z]\d{2}\.\d{1,2})\d*$", r"\1", d["icd"])  # ICD-10-GM has at most two digits after the point
    if d["tage"][1] == 0:  # an outpatient diagnosis gets an outpatient document, not a one-day stay
        types = [t for t in types if t[2] == (0, 0)]
    doc_type, _, (lo, hi) = rng.choices(types, weights=[t[1] for t in types])[0]
    age = rng.randint(max(0, d["alter"][0]), max(d["alter"][0], min(99, d["alter"][1])))
    sex = d["geschlecht"] if d["geschlecht"] in ("m", "w") else rng.choice("mw")
    if lo:
        lo, hi = max(1, d["tage"][0]), max(1, d["tage"][0], d["tage"][1])
    admission = dt.date(2012, 1, 1) + dt.timedelta(days=rng.randint(0, 5400))
    discharge = admission + dt.timedelta(days=rng.randint(lo, hi))
    letter_date = discharge + dt.timedelta(days=rng.choice([0, 0, 0, 1, 2, 3, 6]))
    birth = admission - dt.timedelta(days=age * 365 + rng.randint(0, 364))

    # pass 1: a full sheet
    name, city = rng.choice(hospitals)
    clinic_addr = address(fk, rng, [o for o in places if o[1] == city] or places)
    rng.randint(201, 9999)  # unused area code, see the note above make_case
    staff = []
    for male, female, titles in ROLES:
        p = person(rng)
        staff.append({"rolle": female if p["geschlecht"] == "w" else male, "titel": rng.choice(titles), **p})
    gp = person(rng)
    other_clinic = rng.choice(hospitals)
    old_phone = lambda: (rng.randint(200, 9999), rng.randint(0, 9999), rng.choice(range(6)))  # noqa: E731, replaced in pass 3
    patient = {**person(rng, sex), "alter": age, "geburtsdatum": birth.isoformat(), **address(fk, rng, places),
               "beruf": fk.job() if 18 <= age < 67 and rng.random() < 0.5 else None}
    clinic = {"name": name, **clinic_addr, "telefon": old_phone(), "fax": old_phone(), "mail": rng.choice(range(4)), "url": None}
    initials = "/".join(rng.sample(["mk", "sr", "hb", "tl", "ws", "jg", "ek", "pf", "dn", "as", "bo", "ri"], 2))
    gp = {"titel": rng.choice(["Dr. med.", "Dr. med.", "Dr.", "", "Dipl.-Med."]), **gp, **address(fk, rng, places)}
    rng.randint(201, 9999), old_phone()
    others = {"angehoerige": [relative(rng, age)],
              "vorbehandler": {"titel": rng.choice(PHRASES["titles_senior"]), **person(rng), "klinik": other_clinic[0], "ort": other_clinic[1]},
              "sonstige": {"rolle": rng.choice(["gesetzlicher Betreuer", "Mitarbeiterin Sozialdienst", "Sachbearbeiter Krankenkasse"]), **person(rng)}}
    case_no = rng.choice([str(rng.randint(10**6, 10**9)), f"{rng.randint(2012, 2026)}{rng.randint(10**4, 10**6)}", f"F{rng.randint(10**5, 10**7)}"])
    patient_no = str(rng.randint(10**5, 10**8))
    insurance_no = rng.choice("ABCDEFGHKLMNPRSTVWXYZ") + str(rng.randint(10**8, 10**9 - 1))
    ward = rng.choice([str(rng.randint(1, 40)), f"{rng.choice('ABCDM')}{rng.randint(1, 9)}", f"{rng.choice('ABCH')}{rng.randint(10, 49)}",
                       f"{rng.randint(1, 9)}{rng.choice('abAB')}"])
    rng.randint(1, 420)
    pathology_no = f"{rng.choice('EHP')}{rng.randint(1000, 99999)}/{admission.year % 100:02d}"
    dates = {"aufnahme": admission.isoformat(), "entlassung": discharge.isoformat(), "briefdatum": letter_date.isoformat()}
    rng.choices(range(3), [3, 5, 3]), rng.choices(range(3), [5, 3, 2]), rng.choice(range(6)), rng.choice(range(3))
    style = {"umfang": None, "briefkopf": None, "datumsformat": None, "vorgeschichte": None,
             "ton": rng.choice(PHRASES["tones"])}
    if lo == hi == 0:
        dates = {"untersuchung_oder_vorstellung": admission.isoformat(), "briefdatum": letter_date.isoformat()}
    case = {"doc_id": f"v3-{seed}-{i:05d}", "dokumentart": doc_type, "fachgebiet": d["fach"], "hauptdiagnose": f"{d['text']} ({icd})",
            "patient": patient, "klinik": clinic, "personal": staff, "kuerzel": initials, "hausarzt": gp, "weitere_personen": others,
            "kennungen": None, "daten": dates, "stil": style}

    # pass 2: real letters carry far fewer identifiers than the full sheet (GraSCCo 5.6 % of characters, the full sheet 15.6 %)
    case["personal"] = staff = [staff[k] for k in sorted(rng.sample(range(4), rng.choices([1, 2, 3, 4], [3, 4, 3, 1])[0]))]
    for st in staff:
        if rng.random() < 0.06:
            st["titel"] = rng.choice(PHRASES["titles_rare"])
    others["angehoerige"] = others["angehoerige"] if rng.random() < 0.35 else []
    for k, pr in (("vorbehandler", 0.4), ("sonstige", 0.12)):
        if rng.random() > pr:
            others[k] = None
    if rng.random() > 0.7:
        case["hausarzt"] = gp = None
    if rng.random() < 0.5:
        patient["strasse"] = patient["plz"] = patient["ort"] = patient["bundesland"] = None
    for _ in ("telefon", "fax", "mail", "url"):  # contact data is drawn anew in pass 3
        rng.random()
    if rng.random() < 0.4:
        clinic["strasse"] = clinic["plz"] = None
    yy = admission.isoformat()[2:4]
    ids = {"fallnummer": case_no, "patientennummer": patient_no, "versichertennummer": insurance_no,
           "station": rng.choice([ward, f"{rng.choice(['PSY', 'K', 'KJP', 'N', 'G', 'IMC', 'ITS', 'U'])}{rng.randint(1, 20)}",
                                  f"{rng.choice('ABCDEFGH')}{rng.randint(1, 4)}{rng.randint(0, 9)}", f"{rng.choice(['KJPP', 'KIN', 'ONK', 'CH'])}-{rng.randint(1, 6)}",
                                  str(rng.randint(1, 12)) + rng.choice(["", "", " Süd", " Nord", " West", " Ost"])]),
           "zimmer": str(rng.randint(1, 420)),
           "op_saal": rng.choice(["I", "II", "III", "IV", "1", "2", "3", "5", "7"]),
           "labornummer": rng.choice([f"{rng.randint(10**7, 10**8)}", f"L{yy}-{rng.randint(10**4, 10**6)}", f"{rng.randint(1000, 9999)}-{rng.randint(100, 999)}"]),
           "auftragsnummer": rng.choice([f"A{rng.randint(10**5, 10**7)}", f"{rng.randint(10**5, 10**7)}/{yy}"]),
           "eingangsnummer_pathologie": pathology_no}
    keep = {"fallnummer": 0.55, "patientennummer": 0.35, "versichertennummer": 0.15, "station": 0.45, "zimmer": 0.3,
            "op_saal": 0.6 if "Operation" in doc_type else 0.0, "labornummer": 0.15, "auftragsnummer": 0.1, "eingangsnummer_pathologie": 0.2}
    case["kennungen"] = ids = {k: (v if rng.random() < keep[k] else None) for k, v in ids.items()}
    style["briefkopf"] = rng.choices(PHRASES["letterheads"], [4, 3, 2, 2, 1])[0]
    style["datumsformat"] = rng.choice(PHRASES["date_formats"])
    style["vorgeschichte"] = rng.choice(PHRASES["history_forms"])
    if age < 18:  # children and some psychiatric patients are called by first name in the running text
        style["anrede"] = rng.choice(PHRASES["naming_children"])
    elif dept == "Psychiatrie und Psychotherapie":
        style["anrede"] = rng.choice(PHRASES["naming_psychiatry"])
    style["angaben"] = "Nimm nur Angaben, die zu dieser Dokumentart passen. Ein echter Brief nennt Namen und Kennungen meist nur im Kopf, in der Anrede und in der Unterschrift."

    # pass 3: a clinic that fits the department, local area codes, more ID forms, a course of the case, wider length
    towns = {o[1] for o in places}
    fitting = []
    for name, town in (x for x in hospitals if x[1] in towns and not HISTORIC.search(x[0])):
        hit = [f for rx, f in CLINIC_FIT if re.search(rx, name)]
        if (not hit or any(dept in f for f in hit)) and (dept == "Pädiatrie" or not re.search(r"Kinder|Jugend", name)):
            fitting.append((name, town))
    if any(dept in f for rx, f in CLINIC_FIT if f) and rng.random() < 0.3:
        fitting = [k for k in fitting if any(dept in f for rx, f in CLINIC_FIT if re.search(rx, k[0]))] or fitting
    name, town = rng.choice(fitting)
    in_town = [o for o in places if o[1] == town]
    plz, ort, state = rng.choice(in_town)
    slug = slugify(name)
    code = area_code(rng, plz, ort, places, area_codes)
    tel, switchboard = phone(rng, code)
    fax, _ = phone(rng, code, switchboard)
    clinic.update({"name": name, "plz": plz if clinic.get("plz") else None, "ort": ort, "bundesland": state,
                   "telefon": tel if rng.random() < 0.85 else None, "fax": fax if rng.random() < 0.65 else None,
                   "mail": rng.choice([f"{rng.choice(['sekretariat', 'info', 'ambulanz', 'station', 'kardiologie', 'chirurgie', 'aufnahme'])}@{slug}.de",
                                       f"{staff[0]['vorname'][0].lower()}.{staff[0]['nachname'].lower()}@{slug}.de"]) if rng.random() < 0.4 else None,
                   "url": rng.choice([f"www.{slug}.de", f"https://www.{slug}.de", f"{slug}.de"]) if rng.random() < 0.3 else None})
    if clinic["mail"] and not re.fullmatch(r"[a-z.\-]+@[a-z\-]+\.de", clinic["mail"]):
        clinic["mail"] = f"info@{slug}.de"
    if gp:
        gp["telefon"] = phone(rng, area_code(rng, gp["plz"], gp["ort"], places, area_codes))[0] if rng.random() < 0.6 else None
        if rng.random() < 0.25:
            gp["fax"] = phone(rng, area_code(rng, gp["plz"], gp["ort"], places, area_codes))[0]
    if patient.get("ort") and rng.random() < 0.12:
        patient["telefon"] = phone(rng, area_code(rng, patient["plz"], patient["ort"], places, area_codes))[0]
    if ids.get("patientennummer") and rng.random() < 0.4:
        p = ids["patientennummer"]
        ids["patientennummer"] = rng.choice([p.zfill(10), f"{p[:4]} {p[4:]}".strip(), f"PID{p}"])
    if ids.get("fallnummer") and rng.random() < 0.3:
        f = ids["fallnummer"]
        ids["fallnummer"] = rng.choice([f"{f[:4]}.{f[4:]}" if len(f) > 6 else f, f"{f}-{rng.randint(1, 9)}", f.zfill(12)])
    if rng.random() < 0.15:
        ids["arztnummer_lanr"] = str(rng.randint(10**8, 10**9 - 1))
    r = rng.random()  # 60 % everyday cases, 25 % everyday with a twist, 15 % difficult courses
    if r < 0.60:
        course = "Alltag: typischer, unkomplizierter Verlauf"
    elif r < 0.85:
        course = "Alltag mit Wendung: " + rng.choice(PHRASES["twists"])
    else:
        course = ("schwieriger Verlauf: mehrere Probleme, die sich gegenseitig bedingen (Kaskade), "
                  + rng.choice(PHRASES["difficult_courses"]))
    case["verlauf"] = course
    # a full letterhead without an address on the sheet made writers invent one ("Musterstraße 1, 12345")
    if "voller Briefkopf" in style["briefkopf"]:
        if not clinic.get("strasse") or not clinic.get("plz"):
            clinic["strasse"], clinic["plz"] = address(fake("de_DE", rng), rng, in_town)["strasse"], plz
        if not gp:
            style["briefkopf"] = "voller Briefkopf mit Klinik, Anschrift und Kontakt, ohne Empfängeranschrift"
    style["anschriften"] = "Anschriften, Straßen, Postleitzahlen und Nummern nur aus dem Fallblatt. Was dort fehlt, lässt du weg."
    rng.lognormvariate(6.05, 0.5)
    style["aufbau"] = rng.choice(PHRASES["layouts"])
    style["altersangabe"] = "Form der Altersangabe wie " + rng.choice(PHRASES["age_forms"])
    negatives = [rng.choice(g) for g in PHRASES["lookalikes"]] + [rng.choice(PHRASES["lookalikes"][0])] if rng.random() < 0.12 else []

    # pass 4: fixes found by LLM judges and by error analysis
    # length: GraSCCo dev median is 470 words and writers deliver about 80 % of the target, so the target median is about 550 words
    words = int(min(1800, max(150, random.Random("laenge:" + case["doc_id"]).lognormvariate(6.31, 0.55))))
    style["umfang"] = f"etwa {round(words, -1)} Wörter"
    user_codes(case)
    phrasing(case)
    for st in staff:  # "MD", "PhD" before a name gave letters away
        if re.search(r"\b(?:MD|PhD)\b", st.get("titel") or ""):
            st["titel"] = rng.choice(["Dr. med.", "Dr.", "PD Dr. med."])
    # English glosses in the diagnosis list, e.g. "(Bell's palsy)", gave letters away
    case["hauptdiagnose"] = re.sub(r"\s*\((?=[^)]*(?:'s\b|\bpalsy\b|\bdisease\b|\bsyndrome\b|\bfailure\b|\binjury\b))[^)]*\)", "", case["hauptdiagnose"])
    for k in ("patient", "klinik", "hausarzt"):  # the federal state has no label in the GeMTeX scheme and real letters rarely name it
        if case.get(k):
            case[k].pop("bundesland", None)
    if d["icd"].startswith("P"):  # newborn codes: age in days, not "0 years"
        patient["alter"] = 0
        patient["alter_in_tagen"] = rng.randint(0, 27)
    # IDs from the GeMTeX guideline (section 4.5) that the sheet lacked
    yy = (dates.get("aufnahme") or dates.get("untersuchung_oder_vorstellung"))[2:4]
    letters = "ABCDEFGHKLMNPRSTUVWXZ"
    extra = {
        "befundnummer": (0.25, lambda: rng.choice([f"B{yy}-{rng.randint(1000, 99999)}", f"{rng.randint(10**6, 10**8)}", f"BEF-{rng.randint(10**4, 10**6)}"])),
        "roentgennummer": (0.12, lambda: rng.choice([f"R{rng.randint(10**5, 10**7)}", f"{rng.randint(10**5, 10**7)}/{yy}"])),
        "blocknummer_pathologie": (0.3 if dept in SURGICAL or "Onkologie" in dept else 0.05,
                                   lambda: f"{rng.choice('EHPB')}{rng.randint(10**4, 10**6)}-{rng.randint(1, 12)}"),
        "studien_pseudonym": (0.05 if "Onkologie" not in dept else 0.25,
                              lambda: f"{rng.choice('SPK')}{rng.randint(1, 99):02d}{''.join(rng.choice(letters) for _ in range(2))}-{''.join(rng.choice(letters + '0123456789') for _ in range(5))}"),
        "iban_fuer_sozialdienst": (0.02, lambda: "DE" + "".join(str(rng.randint(0, 9)) for _ in range(20))),
        "protokoll_oder_geraetenummer": (0.08, lambda: rng.choice([f"{rng.randint(10**5, 10**6)}-{rng.randint(1000, 9999)}", f"{rng.randint(10**6, 10**8)}"])),
    }
    for k, (pr, make) in extra.items():
        if rng.random() < pr:
            ids[k] = make()
    # short IDs leaked because our letters only had "Station X"; the guideline counts ward and room numbers as ID
    if rng.random() < 0.6:
        ward_line(case, rng)
    if rng.random() < 0.4:
        negatives += rng.sample(PHRASES["hard_negatives"], k=rng.randint(1, 3))
    if negatives:  # the judges spotted forced content, so these are optional: the writer uses only what fits
        style["gegenbeispiele"] = {"nur verwenden, wenn es zum Fall passt": negatives}
    return case


def user_codes(case):
    """User codes as hospital systems print them (surname letters plus first-name letters, e.g. MUELAN for Anna Müller).
    Before, the dictation sign came from twelve fixed pairs ("tl/bo"). Now it is derived from a staff name in several
    spellings, and 35 % of sheets carry a line with a system user code."""
    rng = random.Random("kuerzel:" + case["doc_id"])
    doc = rng.choice(case["personal"])
    v, n = ascii_fold(doc["vorname"]), ascii_fold(doc["nachname"])
    if len(v) < 2 or len(n) < 3:
        return
    a = (v[0] + n[0]).lower()
    b = "".join(rng.choice("abcdeghjklmnprstw") for _ in range(2))  # the typist, not named on the sheet
    case["kuerzel"] = rng.choice([f"{a}/{b}", f"{a}/{b}", f"{a.upper()}/{b}", f"{a.upper()}/{b.upper()}", f"{a}-{b}", f"{v[0].lower()}.{b}",
                                  f"{n[:2]}/{b}", f"{n[:3]}/{b}", f"{a}/{b}/{rng.choice('abcdeghjklmnprstw')}"])
    if rng.random() < 0.35:
        p = rng.choice(case["personal"])
        v, n = ascii_fold(p["vorname"]), ascii_fold(p["nachname"])
        if len(v) >= 2 and len(n) >= 4:
            k = rng.choice([(n[:4] + v[:2]).upper()] * 8 + [(n[:4] + v[:2]).lower()] * 3 + [(n[:3] + v[:1]).upper()] * 2
                           + [(v[:1] + n).lower()[:9]] * 3 + [(n[:7] + v[:1]).lower()] * 2 + [(n[:5] + v[:1]).upper() + str(rng.randint(1, 9))] * 2)
            case["benutzerkennung"] = {"zeile": rng.choice(PHRASES["user_lines"]).format(k=k), "wert": k}


def phrasing(case):
    rng = random.Random("formeln:" + case["doc_id"])
    draw = lambda options: rng.choices([x for x, _ in options], [w for _, w in options])[0]  # noqa: E731
    st = case["stil"]
    st["anrede_an_empfaenger"], st["grussformel"] = draw(PHRASES["salutations"]), draw(PHRASES["closings"])
    st["kontakt_im_kopf"], st["erster_satz"] = draw(PHRASES["contact_forms"]), draw(PHRASES["openings"])


def ward_line(case, rng):
    """A ward/room/bed line the writer copies verbatim; labels.py marks the values by their place in it."""
    value = rng.choice([lambda: rng.choice("ABCDEFGHK"), lambda: rng.choice(["I", "II", "III", "IV", "V", "VI"]), lambda: str(rng.randint(1, 20)),
                        lambda: f"{rng.choice('ABCDEFGHK')}{rng.randint(1, 49)}", lambda: f"{rng.randint(1, 9)}{rng.choice('abAB')}",
                        lambda: f"{rng.choice('ABCI')}{rng.randint(0, 9)}{rng.randint(0, 9)}", lambda: str(rng.randint(100, 9999))])()
    unit = rng.choice([e for e in PHRASES["ward_units"] if e != "Kinderstation" or case["patient"]["alter"] < 18])
    room, bed = str(rng.choice([rng.randint(1, 40), rng.randint(100, 450)])), str(rng.randint(1, 4))
    template = rng.choice(PHRASES["ward_lines"])
    values = {}
    for key, slot, v in (("station", "{s}", value), ("zimmer", "{z}", room), ("bett", "{b}", bed)):
        if slot in template:
            case["kennungen"][key] = values[key] = v
    case["stationsangabe"] = {"zeile": template.format(e=unit, s=value, z=room, b=bed), "vorlage": template, "einheit": unit, "werte": values}

# -----------------------------------------------------------------------------
# the LLM calls


def call(key, model, system, user, temperature=0.8, max_tokens=16000, reasoning="", providers=(), tries=6):
    ignore = [p for prefix, ps in MAKER_API.items() if model.startswith(prefix) for p in ps]
    body = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
            "provider": {"only": list(providers), "allow_fallbacks": True} if providers else {**PROVIDER, "ignore": ignore},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if reasoning:  # without a limit GLM-5.3 thinks until max_tokens
        body["reasoning"] = {"effort": reasoning}
    err = None
    for t in range(tries):
        try:
            req = urllib.request.Request(API, data=json.dumps(body).encode(), headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.loads(r.read())
            if "choices" not in d:
                raise RuntimeError(json.dumps(d)[:300])
            return d["choices"][0]["message"].get("content") or "", d.get("usage", {}), d.get("provider")
        except Exception as e:  # noqa: BLE001
            err = e  # throttled or overloaded: back off with jitter, so 32 parallel workers spread out
            time.sleep(min(60, 2 ** t * (4 if "429" in str(e) else 2)) + random.random() * 2)
    raise RuntimeError(f"{model}: {err}")


CORE_SYSTEM = open(os.path.join(DATA, "core_prompt.md"), encoding="utf-8").read()
ICD = re.compile(r"^[A-Z]\d{2}(\.\d{1,2})?[-+*!]?$")
OPS = re.compile(r"^\d-\d{2,3}(\.[\dxa-z]{1,3})?$")
ATC = re.compile(r"^[A-Z]\d{2}[A-Z]{2}\d{2}$")
DOSING = re.compile(r"^(?:\d+(?:[,.]\d+|/\d)?|½|¼)(?:-(?:\d+(?:[,.]\d+|/\d)?|½|¼)){2,3}$|^(?:bei Bedarf|b\. ?B\.|1x/Woche|1x wöchentlich|alle \d+ (?:Tage|Wochen)|einmalig|nach Schema|kontinuierlich|.*(?:Hub|IE|Tropfen|Trpf).*)$")


def check_core(k):
    """Problems of a planned clinical core; empty means usable."""
    p = [f"ICD {d.get('icd')}" for d in k.get("nebendiagnosen") or [] if not ICD.match(str(d.get("icd", "")))]
    p += [f"OPS {d.get('ops')}" for d in k.get("prozeduren") or [] if not OPS.match(str(d.get("ops", "")))]
    for drugs in ("medikation_aufnahme", "medikation_entlassung"):
        classes = {}
        for m in k.get(drugs) or []:
            if m.get("hinweis") == "abgesetzt":
                continue
            atc = str(m.get("atc", ""))
            if not ATC.match(atc):
                p.append(f"ATC {atc}")
                continue
            if not re.search(r"\d", str(m.get("staerke", ""))):
                p.append(f"no strength: {m.get('wirkstoff')}")
            if not DOSING.match(str(m.get("schema", "")).strip()):
                p.append(f"dosing: {m.get('schema')}")
            # the same ATC level-4 class twice (two ACE inhibitors); insulins and analgesics often come in pairs
            if atc[:5] in classes and atc[:5] not in ("A10AB", "A10AE", "N02AA", "N02BB", "B05BB", "A12BA"):
                p.append(f"class twice: {atc[:5]}")
            classes[atc[:5]] = m.get("wirkstoff")
    return p


def plan_core(key, case, model, reasoning, providers, tries=2):
    """An open model plans secondary diagnoses, medication and procedures before the letter is written. Two tries."""
    c, cost = case, 0.0
    user = json.dumps({"dokumentart": c["dokumentart"], "fachgebiet": c["fachgebiet"], "hauptdiagnose": c["hauptdiagnose"],
                       "alter": c["patient"]["alter"], "geschlecht": c["patient"]["geschlecht"], "verlauf": c.get("verlauf"),
                       "daten": c["daten"]}, ensure_ascii=False, indent=1)
    for _ in range(tries):
        raw, usage, _ = call(key, model, CORE_SYSTEM, user, temperature=0.7, max_tokens=8000, reasoning=reasoning, providers=providers)
        cost += usage.get("cost") or 0
        m = re.search(r"\{.*\}", raw, re.S)
        try:
            core = json.loads(m.group()) if m else None
        except json.JSONDecodeError:
            core = None
        problems = check_core(core) if isinstance(core, dict) else ["no JSON"]
        if not problems:
            break
    if not problems:  # real letters never show ATC codes: off the sheet, kept on the side
        core = {k: v for k, v in core.items() if v}
        atc = {d["wirkstoff"]: d.pop("atc") for drugs in ("medikation_aufnahme", "medikation_entlassung") for d in core.get(drugs) or [] if d.get("atc")}
        case["fallkern"], case["fallkern_atc"] = core, atc
    case["fallkern_meta"] = {"model": model, "probleme": problems, "cost": cost}


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items() if v not in (None, [], {}, "")}
    return [clean(v) for v in x] if isinstance(x, list) else x


def user_prompt(case):
    c = clean({k: v for k, v in case.items() if k not in ("doc_id", "fallkern_meta", "fallkern_atc", "stationsangabe", "benutzerkennung")})
    if case.get("stationsangabe"):
        c["stationsangabe"] = case["stationsangabe"]["zeile"]
    if case.get("benutzerkennung"):
        c["zeile_aus_dem_kliniksystem (wörtlich übernehmen, im Kopf, in der Fußzeile oder bei der Unterschrift)"] = case["benutzerkennung"]["zeile"]
    return ("Schreibe das folgende Dokument.\n\nFallblatt (Daten im Fallblatt stehen als JJJJ-MM-TT, im Dokument gilt das Datumsformat aus den Stilvorgaben):\n"
            + json.dumps(c, ensure_ascii=False, indent=1))


def tidy(raw):
    """Writers slip into Markdown now and then; real exports have none."""
    raw = re.sub(r"^```\w*\n|\n```\s*$", "", raw.strip())
    raw = re.sub(r"\*\*|^#{1,4} ", "", raw, flags=re.M)
    return re.sub(r"[ \t]+$", "", raw, flags=re.M).replace("`", "")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default="deepseek/deepseek-v4.1-flash-1017", help="comma-separated OpenRouter ids; every model writes every sheet")
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--core-model", default="", help="model that plans the clinical core first (empty: no core)")
    ap.add_argument("--reasoning", default="low", help="reasoning effort for writer and core model")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--providers", default="", help="comma-separated OpenRouter endpoint tags to pin, e.g. deepinfra/fp8")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    key = os.environ["OPENROUTER_API_KEY"]
    providers = [p for p in a.providers.split(",") if p]
    lists = load_lists()
    system = open(os.path.join(DATA, "system_prompt.md"), encoding="utf-8").read()
    cases = [make_case(i, a.seed, *lists) for i in range(a.start, a.start + a.n)]
    jobs = [(m, c) for m in a.models.split(",") for c in cases]

    def one(job):
        model, case = job
        t0 = time.time()
        if a.core_model and "fallkern" not in case:
            plan_core(key, case, a.core_model, a.reasoning, providers)
        try:
            raw, usage, provider = call(key, model, system, user_prompt(case), a.temperature, reasoning=a.reasoning, providers=providers)
            if not raw.strip():  # about 2 % of answers are empty (thinking used the whole budget): one more try
                raw, usage2, provider = call(key, model, system, user_prompt(case), a.temperature, reasoning=a.reasoning, providers=providers)
                usage = {k: (usage.get(k) or 0) + (usage2.get(k) or 0) for k in ("prompt_tokens", "completion_tokens", "cost")}
        except RuntimeError as e:
            return {"doc_id": case["doc_id"], "model": model, "error": str(e)[:300]}
        return {"doc_id": f"{case['doc_id']}-{model.split('/')[-1]}", "model": model, "provider": provider, "text": tidy(raw), "spans": [],
                "case": case, "raw": raw, "usage": {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "cost")},
                "seconds": round(time.time() - t0, 1), "prompt_version": "v3", "temperature": a.temperature, "reasoning": a.reasoning,
                "ohne_markierung": True}

    os.makedirs(a.out, exist_ok=True)
    path, cost = os.path.join(a.out, "raw.jsonl"), 0.0
    with cf.ThreadPoolExecutor(a.workers) as ex, open(path, "a", encoding="utf-8") as fh:
        for r in ex.map(one, jobs):
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            if "error" in r:
                print("ERROR", r["model"], r["doc_id"], r["error"], flush=True)
                continue
            cost += r["usage"].get("cost") or 0
            print(f"{r['doc_id']:48s} {len(r['text']):5d} chars {r['seconds']:5.1f}s", flush=True)
    print(f"done: {path}, cost {cost:.4f} $")


if __name__ == "__main__":
    main()
