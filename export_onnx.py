"""Export the model to the ONNX files of the Hub repo and compare them with PyTorch.

  python export_onnx.py --model model --out model/onnx --check dev.jsonl

Writes onnx/model.onnx (fp32), onnx/model_quantized.onnx (int8 embedding table, all other weights fp32) and onnx/model_fp16.onnx
(fp16 weights, fp32 inputs and outputs). model.onnx and model_quantized.onnx are the names transformers.js looks for, so
`dtype: "q8"` there loads the int8 file. --check writes onnx/quantization_check.json.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

FILES = ("model.onnx", "model_quantized.onnx", "model_fp16.onnx")


def export(model_dir, out):
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).eval()
    ids = tok(["Herr Max Muster", "x"], padding=True, return_tensors="pt")
    with torch.no_grad():
        torch.onnx.export(model, (ids["input_ids"], ids["attention_mask"]), f"{out}/model.onnx", input_names=["input_ids", "attention_mask"],
                          output_names=["logits"], opset_version=17, dynamo=False,
                          dynamic_axes={k: {0: "batch", 1: "tokens"} for k in ("input_ids", "attention_mask", "logits")})


def to_fp16(out):
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    # keep_io_types: callers pass int64 ids and read fp32 logits, as with the fp32 file
    onnx.save(convert_float_to_float16(onnx.load(f"{out}/model.onnx"), keep_io_types=True), f"{out}/model_fp16.onnx")


def quantize(out):
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    # Only the embedding table (Gather) is int8. With int8 MatMul layers as well, the file shrank to 337 MB but only 97.4 % of
    # tokens kept their PyTorch label on development data (embeddings only: 99.85 %); the old MatMul-only file kept 97.3 %.
    pre = f"{out}/model_pre.onnx"
    quant_pre_process(f"{out}/model.onnx", pre, skip_symbolic_shape=True)
    # the classifier head stays fp32: its 19 class scores are read at a threshold as low as 0.01
    head = [n.name for n in onnx.load(pre, load_external_data=False).graph.node
            if n.op_type in ("MatMul", "Gemm") and "classifier" in n.name]
    # reduce_range=True (7-bit weights): an int8 file without it marked almost every character on GraSCCo dev (F1 0.111, 20 Nov 2026)
    quantize_dynamic(pre, f"{out}/model_quantized.onnx", weight_type=QuantType.QInt8, per_channel=True, reduce_range=True,
                     op_types_to_quantize=["Gather"], nodes_to_exclude=head)
    os.remove(pre)


def check(model_dir, out, path, n, threads, threshold=0.01):
    """Per file against PyTorch, first 512 tokens of each text: same arg-max label, same redaction decision at the threshold, CPU time."""
    import onnxruntime as ort

    texts = [json.loads(line)["text"] for line in open(path, encoding="utf-8")][:n]
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).eval()
    torch.set_num_threads(threads)
    enc = [tok(t, truncation=True, max_length=512, return_tensors="np") for t in texts]
    outside = model.config.label2id["O"]

    def read(logits):
        p = np.exp(logits - logits.max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        return logits.argmax(-1), 1 - p[:, outside] >= threshold

    with torch.no_grad():
        ref = [read(model(input_ids=torch.tensor(e["input_ids"]), attention_mask=torch.tensor(e["attention_mask"])).logits[0].numpy())
               for e in enc]
    res = {}
    for f in FILES:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        session = ort.InferenceSession(f"{out}/{f}", opts, providers=["CPUExecutionProvider"])
        same = red = n = 0
        t0 = time.time()
        for e, (lab, rd) in zip(enc, ref):
            z = session.run(None, {"input_ids": e["input_ids"].astype(np.int64), "attention_mask": e["attention_mask"].astype(np.int64)})[0][0]
            lab2, rd2 = read(z)
            same, red, n = same + int((lab2 == lab).sum()), red + int((rd2 == rd).sum()), n + len(lab)
        res[f"onnx/{f}"] = {"token_agreement": round(same / n, 5), "redact_agreement_t001": round(red / n, 5),
                            "ms_per_text": round(1000 * (time.time() - t0) / len(enc), 1), "mb": round(os.path.getsize(f"{out}/{f}") / 1e6, 1)}
        print(f, json.dumps(res[f"onnx/{f}"]), flush=True)
    return {"texts": len(texts), "files": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--check", default="", help="jsonl with a 'text' field per line")
    ap.add_argument("--n", type=int, default=242, help="number of texts to compare")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    out = a.out or os.path.join(a.model, "onnx")
    export(a.model, out)
    to_fp16(out)
    quantize(out)
    for f in FILES:
        print(f, round(os.path.getsize(f"{out}/{f}") / 1e6), "MB")
    if a.check:
        json.dump(check(a.model, out, a.check, a.n, a.threads), open(f"{out}/quantization_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
