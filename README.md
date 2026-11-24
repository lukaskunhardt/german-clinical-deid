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
pip install german-clinical-deid
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
`Deidentifier(..., threshold=0.5)` redacts less. `Deidentifier(..., onnx_file="onnx/model.onnx")` runs the ONNX file with onnxruntime.

## Reproduce the model

```bash
pip install -e ".[train]"
python rerender.py --download train.jsonl --n 2 --seed 1   # the 2,550 published letters, plus 2 copies with new invented values
python train.py --data train.jsonl --seed 44 --out model  # one NVIDIA A10G, about 45 minutes
python evaluate.py --model model --data grascco --split dev
python export_onnx.py --model model --check train.jsonl   # model.onnx, model_quantized.onnx, model_fp16.onnx, agreement with PyTorch
python plot_pareto.py results/test_comparison.json --metric mdeid_leak_rate --lower-is-better --percent --out pareto
```

To make new letters, `factory.py` writes them through the OpenRouter API (`OPENROUTER_API_KEY`) and `labels.py` turns them into labelled records.

## Smaller model

[LukasKunhardt/german-clinical-deid-base](https://huggingface.co/LukasKunhardt/german-clinical-deid-base) is gbert-base (109M parameters) trained to copy the large model's probabilities as well as the gold labels. It leaks in about as many MDeID test notes as the large model, with a slightly lower F1, and is meant for German only. The package loads it the same way: `Deidentifier("LukasKunhardt/german-clinical-deid-base")`.

```bash
python rerender.py --download train_r10.jsonl --n 10 --seed 7   # 28,050 letters
python distill.py --data train_r10.jsonl --seed 42 --out model-base  # one NVIDIA A10G, about 40 minutes
```

## Measure it on your own documents

The model has only been tested on synthetic and openly available letters. If you have annotated clinical text that cannot leave your site, `partner_eval.py` measures the model there and writes one JSON file with sums only: no text, no file names, no per-document rows. It reads brat, INCEpTION (UIMA CAS JSON), GATE XML and a simple JSONL format, works with any label scheme and runs offline once the model folder is copied in.

```bash
pip install german-clinical-deid
python partner_eval.py --format brat --path my_notes/ --direct NAME,ID,PHONE,STREET --out counts.json
```

It reports word-level and character-level precision, recall and F1 for two ways of running the model (the helper package and the plain `pipeline()`), how many spans of each of your labels were redacted, and in how many documents a name, ID, phone number or street (your `--direct` labels) stayed readable.

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
| `distill.py` | trains the smaller model from the large model's probabilities |
| `evaluate.py` | measures a model on GraSCCo or MDeID |
| `partner_eval.py` | measures the model on your own annotated documents and writes sums only |
| `export_onnx.py` | ONNX export (fp32, int8 embeddings, fp16) and agreement with PyTorch |
| `plot_pareto.py` | quality against size chart |
| `results/` | dev and test numbers of every model that `plot_pareto.py` draws |
| `rules/` | the evaluation rules I fixed before the final test run |
| `data/` | place lists, prompts and diagnoses that `factory.py` uses |

## Tests

This code is a clean rewrite of my research code. The tests in `tests/` compared both versions and found the same spans, labels and scores. They need the old research code to run, so they are mainly a record of that check.

## License

MIT. See `LICENSE`.

Data files in `data/`: `places.tsv` comes from GeoNames (CC BY 4.0, geonames.org); `hospitals.tsv` and `area_codes.tsv` come from Wikidata (CC0); `diagnoses.jsonl` was written by gpt-oss-120b (Apache 2.0).
