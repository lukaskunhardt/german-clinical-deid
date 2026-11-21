# german-clinical-deid

Find and remove personal information in German clinical text: names, dates, ages, addresses, phone numbers, e-mail addresses and IDs.
This repository has the Python package for the model
[LukasKunhardt/german-clinical-deid](https://huggingface.co/LukasKunhardt/german-clinical-deid),
the code that made its synthetic training letters, and the code that trains and measures it.

## How it works

The model is a token classifier (gelectra-large) with one class for "not personal" and 18 identifier types.
The package cuts a letter into overlapping windows of 512 tokens and gives every character the highest score of all tokens that cover it.
The score is the probability that the token is personal. Every character at or above the threshold stored with the model (0.01) is redacted.
The type of a span comes from the most likely identifier class. Text that is almost all in capitals is read a second time in lower case.

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

```
Herr [PHI] (geb. [PHI]) wurde am [PHI] im [PHI] aufgenommen.
[('Max Muster', 'NAME_PATIENT'), ('03.02.1961', 'DATE'), ('12.05.2024', 'DATE'), ('Klinikum Nord', 'LOCATION_HOSPITAL')]
```

`deid.spans(text)` gives the redaction spans as `(start, end, score)`. `deid.entities(text)` gives one span per identifier with its type.
`Deidentifier(..., threshold=0.5)` redacts less. `Deidentifier(..., onnx_file="onnx/model_fp16.onnx")` runs the ONNX file with onnxruntime.

## Reproduce the model

```bash
pip install -e ".[train]"
python rerender.py --download train.jsonl --n 2 --seed 1   # the 2,550 published letters, plus 2 copies with new invented values
python train.py --data train.jsonl --seed 44 --out model  # one NVIDIA A10G, about 45 minutes
python evaluate.py --model model --data grascco --split dev
python export_onnx.py --model model --check train.jsonl   # fp32, fp16 and int8 files, agreement with PyTorch
python plot_pareto.py results/test_comparison.json --metric mdeid_leak_rate --lower-is-better --percent --out pareto
```

To make new letters, `factory.py` writes them through the OpenRouter API (`OPENROUTER_API_KEY`) and `labels.py` turns them into labelled records.

## Files

| File | What it does |
|---|---|
| `german_clinical_deid/deid.py` | the package: windows, scores, redaction spans, typed spans, ONNX runner |
| `german_clinical_deid/presidio.py` | recognizer for Microsoft Presidio |
| `german_clinical_deid/spacy.py` | spaCy component |
| `factory.py` | draws fake case sheets and has open models write the letters |
| `labels.py` | labels the letters by code, rejects bad letters, judge filter, overlap check against test sets |
| `rerender.py` | writes each letter again with new invented values |
| `train.py` | trains the model |
| `evaluate.py` | measures a model on GraSCCo or MDeID |
| `export_onnx.py` | ONNX export (fp32, fp16, int8) and agreement check |
| `plot_pareto.py` | quality against size chart |
| `results/` | dev and test numbers of every model that `plot_pareto.py` draws |
| `rules/` | the rules I wrote before I opened the test sets for the release, unchanged, with their SHA-256 |
| `data/` | place lists, prompts and diagnoses that `factory.py` uses |

## Tests

`tests/test_equivalence.py` checks that the package, `train.py` and `evaluate.py` give the same spans, labels and numbers as the research code that produced the released model.
The other files in `tests/` do the same for the data steps. The tests compare against the research code, so they need a copy of it; they are here to show how I checked the rewrite.

## License

MIT. See `LICENSE`.

Data files in `data/`: `places.tsv` comes from GeoNames (CC BY 4.0, geonames.org); `hospitals.tsv` and `area_codes.tsv` come from Wikidata (CC0); `diagnoses.jsonl` was written by gpt-oss-120b (Apache 2.0).
