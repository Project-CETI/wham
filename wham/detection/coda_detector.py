"""
Coda Detector — Python port of the Project CETI MATLAB Coda-detector.

Detects sperm whale codas in audio recordings using:
1. Teager-Kaiser Energy Operator (TKEO) for click detection
2. SNR-based transient selection
3. Inter-Pulse Interval (IPI) estimation
4. Waveform cross-correlation similarity
5. Graph-based clustering to group clicks into codas

Based on: "Automatic detection and annotation of eastern Caribbean
sperm whale codas" — Project CETI

Original MATLAB code: https://github.com/Project-CETI/Coda-detector
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal

warnings.filterwarnings("ignore", category=wavfile.WavFileWarning)


@dataclass
class DetectorParams:
    """Detection parameters — defaults match the original MATLAB code."""

    fs_target: int = 48000
    f_low: float = 2000.0
    f_high: float = 24000.0
    buffer_length: float = 3.0
    buffer_step: float = 1.0
    detection_threshold: float = 0.3
    snr_threshold: float = 10.0
    snr_window: float = 0.1
    snr_window_narrow: float = 0.07
    crop_margin: float = 0.003
    max_transients_per_buffer: int = 20
    mps_window: float = 0.005
    min_peak_distance: float = 0.0018
    ipi_width: float = 0.00045
    rho_corr: float = 0.4
    rho_ipi: float = 0.3
    rho_amp: float = 0.3
    ici_min: float = 0.05
    ici_max: float = 0.6
    max_clicks_per_coda: int = 8
    min_coda_clicks: int = 3
    xcorr_lags: int = 150
    duplicate_ici_tol: float = 0.005
    duplicate_overlap_ratio: float = 0.7


@dataclass
class Click:
    """A detected click with its properties."""

    time: float
    peak_value: float
    snr: float
    ipi: float = 0.0
    waveform: Optional[np.ndarray] = None
    amplitude: float = 0.0


@dataclass
class Coda:
    """A detected coda (group of clicks)."""

    clicks: List[Click] = field(default_factory=list)
    score: float = 0.0

    @property
    def n_clicks(self) -> int:
        return len(self.clicks)

    @property
    def times(self) -> List[float]:
        return [c.time for c in self.clicks]

    @property
    def icis(self) -> List[float]:
        t = sorted(self.times)
        return [t[i + 1] - t[i] for i in range(len(t) - 1)]

    @property
    def duration(self) -> float:
        t = self.times
        return max(t) - min(t) if len(t) > 1 else 0.0

    @property
    def start_time(self) -> float:
        return min(self.times) if self.clicks else 0.0


def teager_kaiser(sig: np.ndarray) -> np.ndarray:
    """
    Teager-Kaiser Energy Operator.
    Psi[x[n]] = x^2[n] - x[n-1] * x[n+1]

    Enhances impulsive transients like whale clicks while
    suppressing smooth background noise.
    """
    x = np.asarray(sig, dtype=np.float64)
    ex = np.zeros_like(x)
    ex[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
    ex[0] = ex[1]
    ex[-1] = ex[-2]
    return ex


def bandpass_filter(
    data: np.ndarray, fs: int, f_low: float, f_high: float, order: int = 4
) -> np.ndarray:
    """Butterworth bandpass filter."""
    nyq = fs / 2.0
    low = max(f_low / nyq, 0.001)
    high = min(f_high / nyq, 0.999)
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, data)


def resample_to_target(
    data: np.ndarray, fs_orig: int, fs_target: int
) -> Tuple[np.ndarray, int]:
    """Resample audio to target sample rate."""
    if fs_orig == fs_target:
        return data, fs_target
    ratio = fs_target / fs_orig
    n_samples = int(len(data) * ratio)
    resampled = signal.resample(data, n_samples)
    return resampled, fs_target


def detect_clicks_tkeo(
    filtered_signal: np.ndarray, fs: int, params: DetectorParams
) -> List[Tuple[int, float]]:
    """
    Detect clicks using TKEO + peak finding.
    Returns list of (sample_index, peak_value) tuples.
    """
    tkeo = teager_kaiser(filtered_signal)
    tkeo = np.maximum(tkeo, 0)

    tkeo_max = np.max(tkeo)
    if tkeo_max == 0:
        return []
    tkeo_norm = tkeo / tkeo_max

    min_dist_samples = int(params.min_peak_distance * fs)
    peaks, properties = signal.find_peaks(
        tkeo_norm,
        height=params.detection_threshold,
        distance=min_dist_samples,
    )

    edge_margin = int(0.002 * fs)
    valid = (peaks > edge_margin) & (peaks < len(filtered_signal) - edge_margin)
    peaks = peaks[valid]
    heights = properties["peak_heights"][valid]

    return list(zip(peaks, heights))


def compute_snr(
    filtered_signal: np.ndarray, click_sample: int, fs: int, params: DetectorParams
) -> float:
    """Compute SNR for a click candidate."""
    half_win = int(params.snr_window * fs / 2)
    narrow_win = int(params.snr_window_narrow * fs / 2)

    start = max(0, click_sample - half_win)
    end = min(len(filtered_signal), click_sample + half_win)
    segment = filtered_signal[start:end]

    sig_start = max(0, click_sample - narrow_win)
    sig_end = min(len(filtered_signal), click_sample + narrow_win)
    click_segment = filtered_signal[sig_start:sig_end]

    if len(segment) < 10 or len(click_segment) < 5:
        return 0.0

    envelope = np.abs(segment)
    noise_level = np.median(envelope)
    click_peak = np.max(np.abs(click_segment))

    if noise_level <= 0:
        return 60.0

    snr_db = 20 * np.log10(click_peak / noise_level)
    return snr_db


def select_transients(
    filtered_signal: np.ndarray,
    click_candidates: List[Tuple[int, float]],
    fs: int,
    params: DetectorParams,
) -> List[Click]:
    """
    Filter click candidates by SNR threshold.
    Keep only the strongest transients per buffer.
    """
    clicks = []
    for sample_idx, peak_val in click_candidates:
        snr = compute_snr(filtered_signal, sample_idx, fs, params)
        if snr >= params.snr_threshold:
            time_s = sample_idx / fs
            clicks.append(
                Click(
                    time=time_s,
                    peak_value=peak_val,
                    snr=snr,
                )
            )

    clicks.sort(key=lambda c: c.peak_value, reverse=True)
    clicks = clicks[: params.max_transients_per_buffer]
    clicks.sort(key=lambda c: c.time)

    return clicks


def estimate_ipi(
    filtered_signal: np.ndarray, click_sample: int, fs: int, params: DetectorParams
) -> float:
    """
    Estimate Inter-Pulse Interval for a single click.

    The IPI comes from the multipulse structure of sperm whale clicks
    (reflections within the spermaceti organ). It is characteristic of
    individual whales and correlates with body size.
    """
    half_win = int(params.mps_window * fs)
    start = max(0, click_sample - half_win)
    end = min(len(filtered_signal), click_sample + half_win)
    segment = filtered_signal[start:end]

    if len(segment) < 20:
        return 0.0

    tkeo = teager_kaiser(segment)
    tkeo = np.maximum(tkeo, 0)
    tkeo_max = np.max(tkeo)
    if tkeo_max == 0:
        return 0.0
    tkeo_norm = tkeo / tkeo_max

    min_dist = int(params.min_peak_distance * fs)
    peaks, _ = signal.find_peaks(tkeo_norm, distance=min_dist)

    if len(peaks) < 2:
        return 0.0

    peak_vals = tkeo_norm[peaks]
    sorted_idx = np.argsort(peak_vals)[::-1]

    main_peak = peaks[sorted_idx[0]]
    second_peak = peaks[sorted_idx[1]]

    if len(sorted_idx) > 2:
        third_peak = peaks[sorted_idx[2]]
        pk2 = peak_vals[sorted_idx[1]]
        pk3 = peak_vals[sorted_idx[2]]
        if pk2 > 0:
            rp = (pk2 - pk3) / pk2
            if rp < 0.4:
                d2 = abs(main_peak - second_peak)
                d3 = abs(main_peak - third_peak)
                if d3 > d2:
                    second_peak = third_peak

    ipi_samples = abs(main_peak - second_peak)
    ipi_ms = 1000.0 * ipi_samples / fs

    if ipi_ms < 1.8:
        return 0.0

    return ipi_ms


def extract_waveforms(
    filtered_signal: np.ndarray,
    clicks: List[Click],
    fs: int,
    seg_ms: float = 16.0,
    asymmetry: float = 0.12,
) -> None:
    """Extract click waveforms for cross-correlation."""
    seg_samples = int(seg_ms * fs / 1000)
    pre = int(seg_samples * (0.5 - asymmetry))
    post = seg_samples - pre

    for click in clicks:
        sample = int(click.time * fs)
        start = max(0, sample - pre)
        end = min(len(filtered_signal), sample + post)
        waveform = filtered_signal[start:end]

        if len(waveform) < seg_samples:
            waveform = np.pad(waveform, (0, seg_samples - len(waveform)))

        click.waveform = waveform
        click.amplitude = np.max(np.abs(waveform))


def normalized_xcorr(a: np.ndarray, b: np.ndarray, max_lag: int) -> float:
    """Normalized cross-correlation (max absolute value)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0

    a_norm = a / na
    b_norm = b / nb

    corr = np.correlate(a_norm, b_norm, mode="full")
    center = len(a_norm) - 1
    start = max(0, center - max_lag)
    end = min(len(corr), center + max_lag + 1)

    return float(np.max(np.abs(corr[start:end])))


def compute_similarity_matrix(
    clicks: List[Click], params: DetectorParams
) -> np.ndarray:
    """
    Build similarity matrix from cross-correlation, amplitude, and IPI.

    sim(i, j) = rho_corr * xcorr + rho_amp * amp_sim + rho_ipi * ipi_sim

    This weighted combination captures both acoustic similarity (waveform shape)
    and bioacoustic identity (IPI as a proxy for individual identity).
    """
    n = len(clicks)
    sim = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            xcorr_val = normalized_xcorr(
                clicks[i].waveform, clicks[j].waveform, params.xcorr_lags
            )

            amp_max = max(clicks[i].amplitude, clicks[j].amplitude)
            if amp_max > 0:
                amp_sim = (
                    1.0 - abs(clicks[i].amplitude - clicks[j].amplitude) / amp_max
                )
            else:
                amp_sim = 0.0

            ipi_max = max(clicks[i].ipi, clicks[j].ipi)
            if ipi_max > 0:
                ipi_sim = 1.0 - abs(clicks[i].ipi - clicks[j].ipi) / ipi_max
            else:
                ipi_sim = 1.0

            s = (
                params.rho_corr * xcorr_val
                + params.rho_amp * amp_sim
                + params.rho_ipi * ipi_sim
            )

            sim[i, j] = s
            sim[j, i] = s

    np.fill_diagonal(sim, 1.0)
    return sim


def _enumerate_codas(
    times: np.ndarray,
    sim_matrix: np.ndarray,
    size: int,
    params: DetectorParams,
    candidates: list,
) -> None:
    """
    Enumerate all valid coda candidates of a given size.

    A valid candidate has all ICIs within [ici_min, ici_max] and
    reasonable ICI consistency (CV < 2.0).
    """
    n = len(times)
    sorted_indices = np.argsort(times)

    def backtrack(current: list, start: int) -> None:
        if len(current) == size:
            t = times[current]
            t_sorted = np.sort(t)
            icis = np.diff(t_sorted)

            if np.all(icis >= params.ici_min) and np.all(icis <= params.ici_max):
                if len(icis) > 1:
                    ici_std = np.std(icis)
                    ici_mean = np.mean(icis)
                    if ici_mean > 0 and ici_std / ici_mean > 2.0:
                        return

                pair_sims = []
                for a in range(len(current)):
                    for b in range(a + 1, len(current)):
                        pair_sims.append(sim_matrix[current[a], current[b]])

                avg_sim = np.mean(pair_sims) if pair_sims else 0
                score = avg_sim * len(current)

                if avg_sim > 0.3:
                    candidates.append((list(current), score))
            return

        for i in range(start, n):
            idx = sorted_indices[i]
            if current:
                dt = times[idx] - times[current[-1]]
                if dt > params.ici_max * size:
                    break
            current.append(idx)
            backtrack(current, i + 1)
            current.pop()

    backtrack([], 0)


def cluster_clicks_into_codas(
    clicks: List[Click], sim_matrix: np.ndarray, params: DetectorParams
) -> List[Coda]:
    """
    Graph-based clustering: enumerate candidate codas (subsets of clicks)
    that satisfy ICI constraints, score them by similarity, and greedily
    select the best non-overlapping codas.
    """
    n = len(clicks)
    if n < params.min_coda_clicks:
        return []

    times = np.array([c.time for c in clicks])
    candidates: list = []

    for size in range(params.min_coda_clicks, min(n, params.max_clicks_per_coda) + 1):
        _enumerate_codas(times, sim_matrix, size, params, candidates)

    candidates.sort(key=lambda x: x[1], reverse=True)

    used: set = set()
    codas: List[Coda] = []
    for indices, score in candidates:
        if any(i in used for i in indices):
            continue
        coda = Coda(
            clicks=[clicks[i] for i in indices],
            score=score,
        )
        codas.append(coda)
        used.update(indices)

    return codas


def remove_duplicate_codas(codas: List[Coda], params: DetectorParams) -> List[Coda]:
    """Remove duplicate codas that share very similar ICI patterns."""
    if len(codas) <= 1:
        return codas

    to_remove: set = set()
    for i in range(len(codas)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(codas)):
            if j in to_remove:
                continue
            icis_i = codas[i].icis
            icis_j = codas[j].icis
            if len(icis_i) != len(icis_j):
                continue
            if not icis_i:
                continue
            matches = sum(
                1
                for a, b in zip(icis_i, icis_j)
                if abs(a - b) < params.duplicate_ici_tol
            )
            ratio = matches / len(icis_i)
            if ratio >= params.duplicate_overlap_ratio:
                if codas[j].score <= codas[i].score:
                    to_remove.add(j)
                else:
                    to_remove.add(i)

    return [c for i, c in enumerate(codas) if i not in to_remove]


def detect_codas(
    audio_path: str,
    params: Optional[DetectorParams] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
) -> List[Coda]:
    """
    Main detection pipeline.

    Takes an audio file path, returns a list of detected Coda objects.
    Supports any sample rate (automatically resampled to 48 kHz).

    Args:
        audio_path: Path to a WAV audio file.
        params: Detection parameters. Uses defaults if None.
        progress_callback: Optional callback(buffer_idx, total_buffers, time_offset).

    Returns:
        List of detected Coda objects, sorted by start time.
    """
    if params is None:
        params = DetectorParams()

    fs_orig, data = wavfile.read(audio_path)

    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float64)

    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val

    data, fs = resample_to_target(data, fs_orig, params.fs_target)
    filtered = bandpass_filter(data, fs, params.f_low, params.f_high)

    total_samples = len(filtered)
    buffer_samples = int(params.buffer_length * fs)
    step_samples = int(params.buffer_step * fs)

    all_codas: List[Coda] = []
    n_buffers = max(1, (total_samples - buffer_samples) // step_samples + 1)

    for buf_idx in range(n_buffers):
        start = buf_idx * step_samples
        end = min(start + buffer_samples, total_samples)
        buf = filtered[start:end]
        time_offset = start / fs

        if progress_callback:
            progress_callback(buf_idx + 1, n_buffers, time_offset)

        click_candidates = detect_clicks_tkeo(buf, fs, params)
        if not click_candidates:
            continue

        clicks = select_transients(buf, click_candidates, fs, params)
        if len(clicks) < params.min_coda_clicks:
            continue

        for click in clicks:
            sample = int(click.time * fs)
            click.ipi = estimate_ipi(buf, sample, fs, params)
            click.time += time_offset

        extract_waveforms(filtered, clicks, fs)

        sim_matrix = compute_similarity_matrix(clicks, params)
        buffer_codas = cluster_clicks_into_codas(clicks, sim_matrix, params)
        all_codas.extend(buffer_codas)

    all_codas = remove_duplicate_codas(all_codas, params)
    all_codas.sort(key=lambda c: c.start_time)

    return all_codas


def codas_to_dict(codas: List[Coda]) -> List[dict]:
    """Convert codas to a list of dicts for easy serialization."""
    results = []
    for i, coda in enumerate(codas):
        results.append(
            {
                "coda_id": i + 1,
                "n_clicks": coda.n_clicks,
                "start_time": round(coda.start_time, 4),
                "duration": round(coda.duration, 4),
                "icis": [round(ici, 4) for ici in coda.icis],
                "mean_ici": round(np.mean(coda.icis), 4) if coda.icis else 0,
                "click_times": [round(t, 4) for t in sorted(coda.times)],
                "mean_snr": round(np.mean([c.snr for c in coda.clicks]), 1),
                "mean_ipi": round(np.mean([c.ipi for c in coda.clicks]), 2),
                "score": round(coda.score, 4),
            }
        )
    return results


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m wham.detection.coda_detector <audio_file.wav>")
        sys.exit(1)

    audio_file = sys.argv[1]
    print(f"Detecting codas in: {audio_file}")

    def progress(buf: int, total: int, time: float) -> None:
        print(f"  Buffer {buf}/{total} (t={time:.1f}s)")

    detected = detect_codas(audio_file, progress_callback=progress)
    output = codas_to_dict(detected)

    print(f"\nDetected {len(detected)} codas:")
    for r in output:
        icis_str = ", ".join(f"{ici * 1000:.0f}ms" for ici in r["icis"])
        print(
            f"  Coda {r['coda_id']}: {r['n_clicks']} clicks at "
            f"t={r['start_time']:.2f}s "
            f"(duration={r['duration'] * 1000:.0f}ms, ICIs=[{icis_str}])"
        )

    output_path = Path(audio_file).stem + "_codas.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")
