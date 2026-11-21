# Rules written before the final test opening (original files, in German)

I wrote these files before I opened the test sets for this release, and I copy them here unchanged so that anyone can read them. My research repository is private; the commits there are named in the model card. SHA-256 of each file:

```
ef2b895d4b337c6371d87d3359af75594ff1d6e66410458d9c029e1b2e1a3d09  schlussoeffnung_v1_20261120.md
65e866201ac70ff3a3c468e3883ee8db4c8f9661b83e83b7ccb6e2dc4523556b  schluss_lesungen_20261120.json
0fd1e078ba227d790a48949196393d8a9de447c47ef733ddd6b83375e287dd6a  hebel_h_regel_20261120.md
```

- `schlussoeffnung_v1_20261120.md`: how the final opening runs, which measures count, when a claim like "fewer leaks than" is allowed. The last section is an addendum written after the jobs ended and before I read any number (a crash in the counting).
- `schluss_lesungen_20261120.json`: the fixed reading (threshold, decoding) of every model in the test tables.
- `hebel_h_regel_20261120.md`: the rule for a label-schema variant I trained on the same day. It did not pass (more leaks), so this release ships the earlier model.
