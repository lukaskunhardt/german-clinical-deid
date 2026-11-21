"""Recognizer for Microsoft Presidio. It returns the redaction spans; the entity type is the most frequent predicted type of the span.

    from presidio_analyzer import AnalyzerEngine
    from german_clinical_deid.presidio import GermanClinicalRecognizer
    engine = AnalyzerEngine(supported_languages=["de"], nlp_engine=...)  # any German NLP engine
    engine.registry.add_recognizer(GermanClinicalRecognizer("LukasKunhardt/german-clinical-deid"))
    engine.analyze(text, language="de")
"""
from presidio_analyzer import EntityRecognizer, RecognizerResult

from .deid import Deidentifier


class GermanClinicalRecognizer(EntityRecognizer):
    def __init__(self, model, threshold=None, supported_language="de"):
        self._model, self._threshold = model, threshold
        super().__init__(supported_entities=["PHI"], supported_language=supported_language, name="GermanClinicalRecognizer")

    def load(self):
        self.deid = Deidentifier(self._model, threshold=self._threshold)
        self.supported_entities = ["PHI"] + self.deid.labels[1:]

    def analyze(self, text, entities, nlp_artifacts=None):
        scores = self.deid.read(text)
        out = []
        for s, e, p in self.deid.spans(text, scores=scores):
            label = self.deid.span_type(scores[1], s, e)
            if not entities or label in entities or "PHI" in entities:
                out.append(RecognizerResult(entity_type=label, start=s, end=e, score=p))
        return out
