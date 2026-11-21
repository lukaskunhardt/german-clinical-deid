"""Export the model to ONNX (fp32, fp16 and int8) and check that an ONNX file gives the same spans as PyTorch.

  python export_onnx.py --model model --out model/onnx --check letters.jsonl

The shipped files are fp16 (inputs and outputs kept in fp32) and int8. fp32 and fp16 gave the PyTorch numbers exactly on GraSCCo dev.
int8 is dynamic quantization from onnxruntime: its preprocessing step, int8 weights per channel, only MatMul layers, the classifier
head in fp32, and reduce_range=True. Without reduce_range (also with the preprocessing step) the int8 model marked almost every
character on GraSCCo dev (character F1 0.111) on 20 Nov 2026; with it, character F1 was 0.969 against 0.970 for fp32.
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer


def export(model_dir, out):
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).eval()
    ids = tok("Herr Max Muster", return_tensors="pt")
    fp32 = os.path.join(out, "model.onnx")
    torch.onnx.export(model, (ids["input_ids"], ids["attention_mask"]), fp32, input_names=["input_ids", "attention_mask"],
                      output_names=["logits"], opset_version=17, dynamo=False,
                      dynamic_axes={k: {0: "batch", 1: "tokens"} for k in ("input_ids", "attention_mask", "logits")})
    del model

    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    # keep_io_types: callers pass int64 ids and read fp32 logits, as with the fp32 file
    onnx.save(convert_float_to_float16(onnx.load(fp32), keep_io_types=True), os.path.join(out, "model_fp16.onnx"))
    for f in ("model.onnx", "model_fp16.onnx"):
        print(f, round(os.path.getsize(os.path.join(out, f)) / 1e6), "MB")


def quantize_int8(out):
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    pre = os.path.join(out, "model_pre.onnx")
    quant_pre_process(os.path.join(out, "model.onnx"), pre, skip_symbolic_shape=True)
    # the classifier head stays fp32: its 19 class scores are read at a threshold as low as 0.01
    head = [n.name for n in onnx.load(pre, load_external_data=False).graph.node
            if n.op_type in ("MatMul", "Gemm") and "classifier" in n.name]
    # reduce_range=True (7-bit weights; onnxruntime advises it for per-channel int8): F1 0.111 without it, 0.969 with it
    quantize_dynamic(pre, os.path.join(out, "model_int8.onnx"), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"],
                     per_channel=True, reduce_range=True, nodes_to_exclude=head)
    os.remove(pre)
    print("model_int8.onnx", round(os.path.getsize(os.path.join(out, "model_int8.onnx")) / 1e6), "MB")


def check(model_dir, onnx_file, path, n):
    """Share of texts with identical redaction spans and identical typed spans, PyTorch against ONNX."""
    import gc

    from german_clinical_deid import Deidentifier

    texts = [json.loads(line)["text"] for line in open(path, encoding="utf-8")][:n]
    ref = Deidentifier(model_dir)
    want = [(ref.spans(t), ref.entities(t)) for t in texts]
    del ref
    gc.collect()
    ox = Deidentifier(model_dir, onnx_file=onnx_file)
    got = [(ox.spans(t), ox.entities(t)) for t in texts]
    same = sum([(s, e) for s, e, _ in a[0]] == [(s, e) for s, e, _ in b[0]] for a, b in zip(want, got))
    typed = sum(a[1] == b[1] for a, b in zip(want, got))
    print(f"{onnx_file}: redaction spans identical in {same}/{len(texts)} texts, typed spans identical in {typed}/{len(texts)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--check", default="", help="jsonl with a 'text' field per line")
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    out = a.out or os.path.join(a.model, "onnx")
    if not all(os.path.exists(os.path.join(out, f)) for f in ("model.onnx", "model_fp16.onnx")):
        export(a.model, out)
    if not os.path.exists(os.path.join(out, "model_int8.onnx")):
        quantize_int8(out)
    if a.check:
        for f in ("model_fp16.onnx", "model_int8.onnx"):
            check(a.model, os.path.relpath(os.path.join(out, f), a.model), a.check, a.n)


if __name__ == "__main__":
    main()
