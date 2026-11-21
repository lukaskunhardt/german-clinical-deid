# Hebel h (Schema v2, Bedingung c für v1.0): Regel

Festgelegt 20.11.2026, 14:26 UTC, vor dem Training (Jobs gestartet 14:27). Wird nicht geändert; Änderungen nur als neue datierte Datei.

**Umfang für v1.0:** Korpus v2 (`hebelh/korpus_v2_r2.jsonl` im Scratch-Repo, SHA-256 8706f655…) = dieselben 2.550 Briefe wie `hebelA/korpus_r2.jsonl`,
neu gelabelt mit `SCHEMA_V2=1 fabrik/labels.py` (Kennungsarten aus dem Fallblatt-Schlüssel nach `regeln/schema_h_20261120.json`), neu gerendert wie Hebel A.
Abweichung außer den Arten: die Label-Korrekturen vom 20.11. (41 von 2.550 Briefen; 39 falsche DATE-Spannen weg, je 6 ID/DATE dazu).
Die Kategorien ohne Beispiele im Korpus (COUNTRY, ORGANIZATION, OTHER, STATE, VEHICLE, SOCIALSECURITY, IP, BIOID, ACCOUNT) brauchen neue Briefe
und kommen in v1.1 (Zeitkasten; keine neuen Hebel vor v1.0).

**Lauf:** gelectra-large, `--typen-v2`, altes Rezept, Seeds 42, 43, 44 (`runs/hebelh/start.sh`, Modelle `modelle/ds3_large_ATv2/s4x`). Bank wie largeAT.

**Messung:** gepaart 3 gegen 3 gegen largeAT s42..s44, je Seed die Schwärz-Schwelle nach der bekannten Regel (`bank/schwelle_regel.py`), Art-Schwelle 0,5 mit
Spannen nach Art (Hebel S). b1 = strenges Span-und-Typ-F1 mikro über die 28 MDeID-Kategorien; Abbildung unserer Arten auf MDeID für **beide Seiten** nach
`auf_mdeid` in `regeln/schema_h_20261120.json` (Angehörige und Externe → PATIENT). Die alte Abbildung wird als Zusatz gezeigt.

**Zählt, wenn alles gilt:**
1. b1 MDeID dev-A: Mittel der 3 Seeds höher als largeAT, gepaartes Bootstrap-Intervall (95 %) ganz über 0.
2. b1 MDeID dev-B: Mittel nicht schlechter (≥ 0).
3. Schwärzung nicht schlechter: mdeid_de dev-B Lecks (alle und ohne Ketten) Mittel ≤ +1 Punkt; GraSCCo dev höchstens 1 Brief mehr mit Leck;
   SPARCLING dev Lecks Mittel ≤ +1 Punkt; Kurztexte-Fehlalarm ≤ 5 %.
Zählt h: Hauptmodell = bester Seed von Korpus v2 nach der Seed-Regel (`bank/seedwahl.py`, nur dev). Zählt h nicht: largeAT s44 bleibt, Bedingung c ist
damit entschieden (h nach Regel geprüft, verworfen), die Karte nennt es.
