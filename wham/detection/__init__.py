"""
Coda detection module — Python port of the Project CETI MATLAB Coda-detector.

Provides end-to-end coda detection from raw audio recordings:
    raw audio → click detection → similarity analysis → coda clustering

Usage:
    from wham.detection import detect_codas, DetectorParams

    codas = detect_codas("recording.wav")
    for coda in codas:
        print(f"{coda.n_clicks} clicks, ICIs: {coda.icis}")
"""

from wham.detection.coda_detector import (
    Click,
    Coda,
    DetectorParams,
    codas_to_dict,
    detect_codas,
)

__all__ = [
    "Click",
    "Coda",
    "DetectorParams",
    "detect_codas",
    "codas_to_dict",
]
