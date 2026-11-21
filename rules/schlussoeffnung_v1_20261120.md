# Schlussöffnung v1.0 (Bedingung e): Regel

Festgelegt 20.11.2026, 15:00 UTC; überarbeitet 15:10 nach dem Gegenlesen durch einen Sub-Agenten (Punkte unten), vor jedem Test-Job dieser Öffnung. Wird nicht geändert; Änderungen nur als neue datierte Datei, und nur vor dem Start.
Ergänzt `runs/GO_TESTOEFFNUNG`. Die Karte nennt den Commit dieser Datei und den Commit von `regeln/schluss_lesungen_20261120.json`.

## Was geöffnet wird
- GraSCCo-Test (41 Briefe, `bench/split.py`) und MDeID Deutsch Test (402 Dokumente). Beide Sätze genau einmal in dieser Öffnung.
- Frühere Nutzung beider Sätze steht in `results/testnutzung.md` und `bank/zugriffe.jsonl`; die Karte fasst sie zusammen.

## Modelle und Lesungen (fest vor dem Start)
- **Unser Modell:** das Ergebnis der Hebel-h-Regel (`regeln/hebel_h_regel_20261120.md`): zählt h, der Seed nach `bank/seedwahl.py`, sonst largeAT s44.
  Schwärz-Schwelle nach `bank/schwelle_regel.py` auf dev (largeAT s44: 0,01). Arten: Spannen nach Art bei 0,5. GROSS-Lesung wie im Paket.
  Eingetragen in `regeln/schluss_lesungen_20261120.json` und committet, bevor ein Test-Job startet.
- **Zweite Zeile für mein Modell ohne GROSS-Lesung** (dieselben Gewichte, gleiche Bedingung wie die Gegner). Die Karte zeigt beide Zeilen.
- **Ausgelieferte Datei:** dieselben Gewichte als ONNX **fp16** (int8 zerstört gelectra-large: GraSCCo dev F1 0,111 mit int8 per Kanal und dynamisch,
  fp32 und fp16 0,970 wie PyTorch; Job `6b006186efaf5e14f4b36dab`, 20.11. 15:10; int8 in v1.1) (die Datei, die in `onnx/` des Modell-Repos liegt; das Job-Log zeigt ihre SHA-256, die mit der
  hochgeladenen Datei übereinstimmen muss), dieselbe Lesung, onnxruntime 1.32.0, eigener CPU-Job. Eigene Zeile.
- **Abbildung unserer Arten auf MDeID:** Feld `abbildung` in `regeln/schluss_lesungen_20261120.json` (v1, oder v2 wenn h zählt); `bank/schluss_tabelle.py`
  liest sie nur von dort.
- **Gegner, je zwei Zeilen** wie in `results/gegner_tabelle_dev.md`: (1) Standard argmax mit eigenen Arten; (2) Wahrscheinlichkeits-Lesung mit der auf dev
  gewählten Schwelle (Spalte "Schwelle" dort; meine Schwellenregel, sonst Ausweich = bestes mittleres F1 auf GraSCCo dev und MDeID dev-A).
  Die 15 Gegner mit rohen Logits aus `bank/gegner_labels_20261120.json`. GLiNER-Gegner (urchade/gliner_multi_pii-v1, knowledgator/gliner-pii-large-v1.0,
  fastino/gliner2-privacy-filter-PII-multi, syntriq/tarn-german-pii-gliner-base) eine Zeile mit eigener Standardschwelle 0,5, nur Spannen.
- Gegner bekommen keine GROSS-Lesung (wie auf dev); die Karte sagt das.

## Durchführung
- Ein Öffnungslauf, aufgeteilt in vier HF-Jobs, alle aus demselben Commit und zur selben Zeit gestartet (`runs/schluss.sh test haupt|onnx|pf|gliner`,
  Argumente nur aus `bank/schluss_args.py`): transformers 4.57.6 (mein Modell, 13 Gegner), ONNX auf CPU, 5.23.0 (die zwei privacy-filter), gliner
  (Installation wie auf dev). Grund: die Gegner brauchen verschiedene Bibliotheken; so liefen sie auch auf dev. Das ersetzt "alle Gegner im selben Job"
  in `runs/GO_TESTOEFFNUNG`.
- **Fertig** heißt: "TEST fertig" steht im Log. Ein Job ohne diese Zeile gilt als abgestürzt und darf neu starten. Zwischen zwei Starts darf sich nur
  Laden oder Schreiben ändern, nie Dekodierung oder Zählung; jede solche Änderung wird committet und in `bank/zugriffe.jsonl` vermerkt.
- Der Job speichert die Spannen je Dokument privat im Scratch-Repo (`bank/v1/preds_test/`), im Log nur Summen. Ich lese daraus nur Summen mit
  `bank/schluss_tabelle.py` (vorher committet). Kein Testdokument und keine einzelne Vorhersage wird angesehen.
- Vorher geprüft (Probe auf dev, je Gruppe): GraSCCo dev für mein Modell, Horizon small und urchade gleich der dev-Tabelle; MDeID dev b1 gleich
  `bank/arten.py`. Die übrigen Gruppen (MDeID dev im Job, pf, ONNX, knowledgator, tarn-GLiNER) laufen vor dem Start einmal auf dev; das Ergebnis steht in STAND.

## Maße (Bedingung b, wie in den Papers)
Je Satz: Zeichen-F1/P/R (= Köln-Zeichenmaß), Lecks mild (< 50 % verdeckt) und streng (ein Zeichen sichtbar), MDeID zusätzlich ohne Zufallsketten,
streng und locker mit Typ (MDeID-Kategorien, 28), Token binär, HIPAA-Recall (i2b2-2014-Liste, mit und ohne IDNUM), Bootstrap 95 % über Dokumente.

## Typ-Maße: was zählt
- MDeID: Gold aus den brat-Dateien (28 Kategorien). GraSCCo: nur Gold- und vorhergesagte Spannen mit MDeID-Gegenstück (ID, Titel, Angehörige fallen
  raus, sonst bevorzugte das Maß mein Schema).
- Gegner haben zwei Zeilen; für den Vergleich mit Typ zählt je Gegner die bessere der beiden Zeilen. Beide stehen in der Karte.
- Konfidenzintervall (Bootstrap 95 % über Dokumente) für die Typ-, Token- und HIPAA-Maße; Lecks mit Wilson-95 %.

## "Besser als"
Nur wenn gepaartes Bootstrap über Dokumente (2.000 Ziehungen, Seed 0) des Unterschieds zum besten Gegner im selben Maß ein 95-%-Intervall ganz über 0
(bei Lecks: ganz unter 0) ergibt. Sonst schreibt die Karte nur die Zahlen. (15:30: Bestätigung durch einen Sub-Agenten gestrichen, Lukas 20.11. 15:05;
ich prüfe den Vergleich selbst gegen Messcode und Ergebnisdateien.)

## MDeID-Test ist kein sauberer Haltesatz
Laut `results/testnutzung.md` hingen am MDeID-Test vor dem 19.11. 22:50 Entscheidungen über Korpus, Basismodell und eine Zählregel. Die Karte sagt das
und nennt GraSCCo-Test als den saubereren Satz.

## Was in die Karte kommt
Alle Zeilen beider Tabellen, auch wo mein Modell verliert. Hauptzahlen: MDeID Test Lecks mild (alle Angaben) und GraSCCo-Test Zeichen-F1 und Lecks,
dazu strenges Typ-F1 auf MDeID Test. Kein "besser als" ohne das gepaarte Bootstrap oben. Nach dem Ergebnis ändert sich nichts:
keine Regel, keine Schwelle, kein Modell.

## Nachtrag 20.11.2026, 16:30 UTC (nach dem Ende des haupt-Jobs, bevor ich eine Zahl der Tabelle gelesen habe)
`bank/schluss_tabelle.py` stürzte in `bank/lecks_streng.py` ab: Eine Gold-Spanne im MDeID-Test endet 21 Zeichen nach dem Ende ihres Textes (Fehler in den
Daten; gezählt ohne Text anzusehen). Fix: Zeichen außerhalb des Textes zählen beim strengen Leck nicht. Gilt für alle Zeilen gleich, ändert nichts an
gültigen Spannen. Kein Job neu gestartet.
