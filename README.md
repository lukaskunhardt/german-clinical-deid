# german-clinical-deid

Find and remove personal information in German clinical text: names, dates, ages, addresses, phone numbers, e-mail addresses and IDs.
This repository has the package, the code that made the synthetic training letters, and the code that trains and measures the model
[LukasKunhardt/german-clinical-deid](https://huggingface.co/LukasKunhardt/german-clinical-deid).

## Install

```bash
pip install "git+https://github.com/lukaskunhardt/german-clinical-deid"
```

Extras: `[onnx]` for the ONNX files, `[presidio]` for the Presidio recognizer, `[spacy]` for the spaCy component, `[train]` to rebuild data and model.

## Example

```python
from german_clinical_deid import Deidentifier

deid = Deidentifier("LukasKunhardt/german-clinical-deid")
text = "Herr Max Muster (geb. 03.02.1961) wurde am 12.05.2024 im Klinikum Nord aufgenommen."
print(deid.redact(text))
print([(e["text"], e["label"]) for e in deid.entities(text)])
```

`deid.spans(text)` gives the redaction spans as `(start, end, score)`. `deid.entities(text)` gives one span per identifier with its type.
`Deidentifier(..., threshold=0.5)` redacts less. `Deidentifier(..., onnx_file="onnx/model_fp16.onnx")` runs the ONNX file with onnxruntime.

## Reproduce the model

```bash
pip install -e ".[train]"
python rerender.py --download train.jsonl --n 2 --seed 1   # the 2,550 published letters, plus 2 copies with new invented values
python train.py --data train.jsonl --seed 44 --out model  # one NVIDIA A10G, about 45 minutes
python evaluate.py --model model --data grascco --split dev
python export_onnx.py --model model --check train.jsonl
python plot_pareto.py results/dev_comparison.json --metric mdeid_leak_rate --lower-is-better --out pareto
```

To make new letters, `factory.py` writes them through the OpenRouter API (`OPENROUTER_API_KEY`) and `labels.py` turns them into labelled records.

## Files

| File | Lines | What it does |
|---|---|---|
| `german_clinical_deid/deid.py` | 185 | the package: windows, scores, redaction spans, typed spans, ONNX runner |
| `german_clinical_deid/__init__.py` | 9 | package entry |
| `german_clinical_deid/presidio.py` | 30 | recognizer for Microsoft Presidio |
| `german_clinical_deid/spacy.py` | 28 | spaCy component |
| `factory.py` | 567 | draws fake case sheets and has open models write the letters |
| `labels.py` | 261 | labels the letters by code, rejects bad letters, judge filter, overlap check |
| `rerender.py` | 193 | writes each letter again with new invented values |
| `train.py` | 139 | trains the model |
| `evaluate.py` | 151 | measures a model on GraSCCo or MDeID |
| `export_onnx.py` | 70 | ONNX export (fp32, fp16), agreement check |
| `plot_pareto.py` | 102 | quality against size chart |
| `tests/test_equivalence.py` | 117 | package, training data and measures against the research code |
| `tests/test_labels.py` | 87 | labels.py against the research code |
| `tests/test_rerender.py` | 35 | rerender.py against the shipped training file |
| `tests/test_factory.py` | 39 | factory.py case sheets and prompts against the shipped ones |
| total | 2013 | |

## Tests

`tests/test_equivalence.py` checks that the package, `train.py` and `evaluate.py` give the same spans, labels and numbers as the research code that produced the released model.
The other files in `tests/` do the same for the data steps. The tests compare against the research code, so they need a copy of it; they are here to show how I checked the rewrite.

## License

MIT. See `LICENSE`.

Data files in `data/`: `places.tsv` comes from GeoNames (CC BY 4.0, geonames.org); `hospitals.tsv` and `area_codes.tsv` come from Wikidata (CC0); `diagnoses.jsonl` was written by gpt-oss-120b (Apache 2.0).

## Rules of the final test opening

The folder `rules/` holds the rules I wrote before I opened the test sets for the release, unchanged and with their SHA-256. `results/test_comparison.json` holds the test numbers of every model that `plot_pareto.py` draws.
