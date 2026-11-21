"""De-identification of German clinical text.

    from german_clinical_deid import Deidentifier
    deid = Deidentifier("LukasKunhardt/german-clinical-deid")
    deid.redact("Herr Max Muster wurde am 12.05.2024 aufgenommen.")
"""
from .deid import Deidentifier

__all__ = ["Deidentifier"]
