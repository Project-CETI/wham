"""Tests for the coda detector module."""

import tempfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wavfile

from wham.detection import Click, Coda, DetectorParams, codas_to_dict, detect_codas
from wham.detection.coda_detector import (
    bandpass_filter,
    cluster_clicks_into_codas,
    compute_similarity_matrix,
    detect_clicks_tkeo,
    extract_waveforms,
    normalized_xcorr,
    remove_duplicate_codas,
    resample_to_target,
    teager_kaiser,
)


FS = 48000


def _make_click(t_sec: float, fs: int = FS, amplitude: float = 0.8) -> np.ndarray:
    """Generate a synthetic click (damped sinusoid) at a given time."""
    duration = 0.002
    n = int(duration * fs)
    t = np.arange(n) / fs
    click = amplitude * np.sin(2 * np.pi * 5000 * t) * np.exp(-t * 3000)
    return click


def _make_coda_audio(
    click_times: list, fs: int = FS, duration: float = 5.0
) -> np.ndarray:
    """Generate synthetic audio with clicks at specified times."""
    n_samples = int(duration * fs)
    audio = np.random.randn(n_samples) * 0.001
    click = _make_click(0, fs)
    for t in click_times:
        start = int(t * fs)
        end = min(start + len(click), n_samples)
        audio[start : end] += click[: end - start]
    return audio


def _save_wav(audio: np.ndarray, fs: int = FS) -> str:
    """Save audio to a temporary WAV file and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wavfile.write(tmp.name, fs, (audio * 32767).astype(np.int16))
    return tmp.name


class TestTeagerKaiser:
    def test_output_shape(self):
        sig = np.random.randn(1000)
        result = teager_kaiser(sig)
        assert result.shape == sig.shape

    def test_enhances_impulse(self):
        sig = np.zeros(1000)
        sig[500] = 1.0
        result = teager_kaiser(sig)
        assert result[500] > result[400]

    def test_zero_input(self):
        sig = np.zeros(100)
        result = teager_kaiser(sig)
        assert np.all(result == 0)


class TestBandpassFilter:
    def test_removes_dc(self):
        sig = np.ones(FS) + np.sin(2 * np.pi * 5000 * np.arange(FS) / FS)
        filtered = bandpass_filter(sig, FS, 2000, 24000)
        assert abs(np.mean(filtered)) < 0.01

    def test_passes_signal_in_band(self):
        t = np.arange(FS) / FS
        sig = np.sin(2 * np.pi * 5000 * t)
        filtered = bandpass_filter(sig, FS, 2000, 24000)
        assert np.max(np.abs(filtered)) > 0.5


class TestResample:
    def test_same_rate(self):
        data = np.random.randn(1000)
        result, fs = resample_to_target(data, FS, FS)
        assert len(result) == len(data)
        assert fs == FS

    def test_downsample(self):
        data = np.random.randn(48000)
        result, fs = resample_to_target(data, 48000, 24000)
        assert fs == 24000
        assert len(result) == 24000


class TestClickDetection:
    def test_detects_synthetic_clicks(self):
        audio = _make_coda_audio([1.0, 1.2, 1.4])
        filtered = bandpass_filter(audio, FS, 2000, 24000)
        params = DetectorParams()
        clicks = detect_clicks_tkeo(filtered, FS, params)
        assert len(clicks) >= 2

    def test_no_clicks_in_silence(self):
        audio = np.zeros(FS * 3)
        params = DetectorParams()
        clicks = detect_clicks_tkeo(audio, FS, params)
        assert len(clicks) == 0


class TestNormalizedXcorr:
    def test_identical_signals(self):
        a = np.random.randn(100)
        result = normalized_xcorr(a, a, 50)
        assert result > 0.99

    def test_zero_signal(self):
        a = np.random.randn(100)
        b = np.zeros(100)
        result = normalized_xcorr(a, b, 50)
        assert result == 0.0

    def test_orthogonal_signals(self):
        a = np.sin(2 * np.pi * 100 * np.arange(1000) / FS)
        b = np.sin(2 * np.pi * 10000 * np.arange(1000) / FS)
        result = normalized_xcorr(a, b, 50)
        assert result < 0.5


class TestCodaDataclass:
    def test_properties(self):
        clicks = [
            Click(time=1.0, peak_value=0.9, snr=20),
            Click(time=1.2, peak_value=0.8, snr=18),
            Click(time=1.4, peak_value=0.85, snr=19),
        ]
        coda = Coda(clicks=clicks, score=2.5)

        assert coda.n_clicks == 3
        assert len(coda.icis) == 2
        assert abs(coda.icis[0] - 0.2) < 0.001
        assert abs(coda.duration - 0.4) < 0.001
        assert coda.start_time == 1.0

    def test_empty_coda(self):
        coda = Coda()
        assert coda.n_clicks == 0
        assert coda.duration == 0.0
        assert coda.start_time == 0.0


class TestCodaToDicts:
    def test_serialization(self):
        clicks = [
            Click(time=1.0, peak_value=0.9, snr=20, ipi=3.5),
            Click(time=1.2, peak_value=0.8, snr=18, ipi=3.4),
            Click(time=1.4, peak_value=0.85, snr=19, ipi=3.6),
        ]
        codas = [Coda(clicks=clicks, score=2.5)]
        result = codas_to_dict(codas)

        assert len(result) == 1
        assert result[0]["coda_id"] == 1
        assert result[0]["n_clicks"] == 3
        assert "icis" in result[0]
        assert "mean_snr" in result[0]


class TestRemoveDuplicates:
    def test_removes_similar_codas(self):
        params = DetectorParams()
        c1 = Coda(
            clicks=[
                Click(time=1.0, peak_value=0.9, snr=20),
                Click(time=1.2, peak_value=0.8, snr=18),
                Click(time=1.4, peak_value=0.85, snr=19),
            ],
            score=3.0,
        )
        c2 = Coda(
            clicks=[
                Click(time=1.001, peak_value=0.88, snr=19),
                Click(time=1.201, peak_value=0.79, snr=17),
                Click(time=1.401, peak_value=0.84, snr=18),
            ],
            score=2.5,
        )
        result = remove_duplicate_codas([c1, c2], params)
        assert len(result) == 1
        assert result[0].score == 3.0

    def test_keeps_different_codas(self):
        params = DetectorParams()
        c1 = Coda(
            clicks=[
                Click(time=1.0, peak_value=0.9, snr=20),
                Click(time=1.2, peak_value=0.8, snr=18),
                Click(time=1.4, peak_value=0.85, snr=19),
            ],
            score=3.0,
        )
        c2 = Coda(
            clicks=[
                Click(time=3.0, peak_value=0.7, snr=15),
                Click(time=3.5, peak_value=0.6, snr=14),
                Click(time=4.0, peak_value=0.65, snr=16),
            ],
            score=2.0,
        )
        result = remove_duplicate_codas([c1, c2], params)
        assert len(result) == 2


class TestEndToEnd:
    def test_detect_codas_synthetic(self):
        """Full pipeline test with a synthetic coda."""
        click_times = [1.0, 1.15, 1.30, 1.45]
        audio = _make_coda_audio(click_times, duration=3.0)
        wav_path = _save_wav(audio)

        try:
            codas = detect_codas(wav_path)
            assert isinstance(codas, list)
            for coda in codas:
                assert isinstance(coda, Coda)
                assert coda.n_clicks >= 3
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_detect_codas_silence(self):
        """No codas in silence."""
        audio = np.zeros(FS * 3)
        wav_path = _save_wav(audio)

        try:
            codas = detect_codas(wav_path)
            assert len(codas) == 0
        finally:
            Path(wav_path).unlink(missing_ok=True)

    def test_progress_callback(self):
        """Progress callback is called."""
        audio = _make_coda_audio([1.0, 1.2, 1.4], duration=3.0)
        wav_path = _save_wav(audio)
        calls = []

        def cb(buf, total, t):
            calls.append((buf, total, t))

        try:
            detect_codas(wav_path, progress_callback=cb)
            assert len(calls) > 0
        finally:
            Path(wav_path).unlink(missing_ok=True)
