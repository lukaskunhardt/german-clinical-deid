"""spaCy component "german_clinical_deid": sets doc.ents from the redaction spans, labelled with their predicted type.

    import spacy, german_clinical_deid.spacy  # registers the component
    nlp = spacy.blank("de")
    nlp.add_pipe("german_clinical_deid", config={"model": "LukasKunhardt/german-clinical-deid"})
    [(e.text, e.label_) for e in nlp(text).ents]
"""
from typing import Optional

from spacy.language import Language
from spacy.util import filter_spans

from .deid import Deidentifier


@Language.factory("german_clinical_deid", default_config={"model": "LukasKunhardt/german-clinical-deid", "threshold": None})
def make_component(nlp, name, model: str, threshold: Optional[float]):
    deid = Deidentifier(model, threshold=threshold)

    def component(doc):
        scores = deid.read(doc.text)
        # "expand" snaps character spans to whole spaCy tokens, so nothing is cut in half
        ents = [doc.char_span(s, e, label=deid.span_type(scores[1], s, e), alignment_mode="expand")
                for s, e, _ in deid.spans(doc.text, scores=scores)]
        doc.ents = filter_spans([x for x in ents if x is not None] + list(doc.ents))
        return doc

    return component
