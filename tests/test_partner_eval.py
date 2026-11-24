import json

import partner_eval as pe


def test_brat_and_score(tmp_path):
    text = "Herr Max Muster, Tel. 0301234, kam am 03.02.2024."
    (tmp_path / "a.txt").write_text(text, encoding="utf-8")
    (tmp_path / "a.ann").write_text("T1\tNAME 5 15\tMax Muster\nT2\tPHONE 22 29\t0301234\nT3\tDATE 38 48\t03.02.2024\n", encoding="utf-8")
    docs = pe.load_brat(str(tmp_path))
    assert [s[2] for s in docs[0]["spans"]] == ["NAME", "PHONE", "DATE"]
    # name fully, date fully, phone missed
    m = pe.score(docs, [[(5, 15), (38, 48)]], {"NAME", "PHONE"})
    assert m["docs_with_direct"] == 1 and m["docs_missed_direct"] == 1
    assert m["per_label"]["PHONE"] == {"gold_spans": 1, "fully_redacted": 0, "at_least_half": 0, "not_redacted": 1}
    assert m["token_r"] == round(3 / 4, 4) and m["token_p"] == 1.0
    assert "Max" not in json.dumps(m)


def test_jsonl_and_inception(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text(json.dumps({"text": "Dr. Weber", "spans": [{"start": 4, "end": 9, "label": "DOCTOR"}]}) + "\n", encoding="utf-8")
    assert pe.load_jsonl(str(p))[0]["spans"] == [(4, 9, "DOCTOR")]
    cas = {"%FEATURE_STRUCTURES": [{"%TYPE": "uima.cas.Sofa", "sofaString": "Dr. Weber"},
                                   {"%TYPE": "webanno.custom.PHI", "begin": 4, "end": 9, "kind": "NAME_DOCTOR"}]}
    (tmp_path / "x.json").write_text(json.dumps(cas), encoding="utf-8")
    docs = pe.load_inception(str(tmp_path), "webanno.custom.PHI", "kind")
    assert docs[0]["spans"] == [(4, 9, "NAME_DOCTOR")]
    m = pe.score(docs, [[(4, 9)]], {"NAME_DOCTOR"})
    assert m["token_f1"] == 1.0 and m["docs_missed_direct"] == 0


def test_gate(tmp_path):
    xml = ('<?xml version="1.0" encoding="UTF-8"?><GateDocument version="3"><GateDocumentFeatures/>'
           '<TextWithNodes>Pat. <Node id="5"/>Hanna Meier<Node id="16"/>, Zürich<Node id="23"/></TextWithNodes>'
           '<AnnotationSet Name="phi-annotations"><Annotation Id="1" Type="Name" StartNode="5" EndNode="16"/></AnnotationSet>'
           '<AnnotationSet Name="other"><Annotation Id="2" Type="Location" StartNode="18" EndNode="23"/></AnnotationSet></GateDocument>')
    (tmp_path / "r.xml").write_text(xml, encoding="utf-8")
    docs = pe.load_gate(str(tmp_path), "phi-annotations")
    assert docs[0]["text"] == "Pat. Hanna Meier, Zürich"
    assert docs[0]["spans"] == [(5, 16, "Name")]
    assert docs[0]["text"][5:16] == "Hanna Meier"
