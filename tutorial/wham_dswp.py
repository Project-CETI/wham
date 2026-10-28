"""Reproducible DSWP ingestion and WhAM embedding helpers.

The module keeps the expensive model import lazy. This makes dataset
inspection, metadata checks and unit tests runnable without downloading the
WhAM checkpoints.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np


DSWP_REPO_ID = "orrp/DSWP"
DSWP_REVISION = "a2e5d6dd02fc60343e1288c33314e14e8b7aa5be"
WHAM_SAMPLE_RATE = 16_000
WHAM_EMBEDDING_DIM = 1_280


def download_dswp_audio(
    file_names: Iterable[str],
    cache_dir: Path,
    repo_id: str = DSWP_REPO_ID,
) -> list[Path]:
    """Download named public DSWP WAV files into ``cache_dir``.

    The Hugging Face dataset is a soundfolder with numeric WAV names. This
    helper intentionally downloads only named files so a tutorial run does
    not silently fetch the whole 1,501-file release.
    """

    from huggingface_hub import hf_hub_download

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for file_name in file_names:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=Path(file_name).name,
            repo_type="dataset",
            revision=DSWP_REVISION,
            local_dir=str(cache_dir),
        )
        paths.append(Path(path))
    return paths


def audio_manifest(audio_paths: Iterable[Path]):
    """Return verified container metadata for local WAV files."""

    import pandas as pd
    import soundfile as sf

    rows = []
    for path in audio_paths:
        info = sf.info(str(path))
        rows.append(
            {
                "file": Path(path).name,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "duration_s": round(info.duration, 6),
                "frames": info.frames,
                "format": info.format,
                "subtype": info.subtype,
            }
        )
    return pd.DataFrame(rows)


def build_wham_interface(
    wham_root: Path,
    coarse_checkpoint: Path,
    codec_checkpoint: Path,
    device: str = "cpu",
):
    """Build the official VampNet interface for embedding extraction.

    The public tutorial needs only the coarse model and codec. It does not
    load the optional coarse-to-fine or WaveBeat checkpoints.
    """

    wham_root = Path(wham_root).resolve()
    vampnet_root = wham_root / "vampnet"
    if str(vampnet_root) not in sys.path:
        sys.path.insert(0, str(vampnet_root))

    from vampnet.interface import Interface

    return Interface(
        coarse_ckpt=str(Path(coarse_checkpoint).resolve()),
        codec_ckpt=str(Path(codec_checkpoint).resolve()),
        coarse2fine_ckpt=None,
        wavebeat_ckpt=None,
        device=device,
    )


def extract_layerwise_embedding(audio_path: Path, interface) -> np.ndarray:
    """Return mean-pooled WhAM activations with shape ``[layers, 1280]``.

    This follows the repository's ``vampnet_embed`` path. The returned array
    retains the layer axis so a researcher can compare layer 10 with other
    layers without rerunning the model.
    """

    import torch
    from audiotools import AudioSignal

    signal = AudioSignal(str(audio_path))
    with torch.inference_mode():
        signal = interface.preprocess(signal)
        vampnet = interface.coarse
        codes = interface.encode(signal)[:, : vampnet.n_codebooks, :]
        latents = vampnet.embedding.from_codes(codes, interface.codec)
        _, activations = vampnet(latents, return_activations=True)

    if activations.ndim != 4:
        raise ValueError(
            "Expected WhAM activations with shape [layers, batch, time, dim]"
        )
    if activations.shape[1] != 1:
        raise ValueError(f"Expected one audio item, got batch {activations.shape[1]}")
    pooled = activations[:, 0].mean(dim=1)
    result = pooled.detach().cpu().numpy().astype(np.float32)
    validate_embeddings(result[None, ...])
    return result


def extract_embeddings(
    audio_paths: Iterable[Path],
    interface,
    layer: int = 10,
) -> np.ndarray:
    """Return one mean-pooled WhAM vector per audio path."""

    layerwise = [extract_layerwise_embedding(Path(path), interface) for path in audio_paths]
    if not layerwise:
        raise ValueError("At least one audio path is required")
    stacked = np.stack(layerwise)
    if layer < 0 or layer >= stacked.shape[1]:
        raise ValueError(f"Layer {layer} is unavailable in shape {stacked.shape}")
    return stacked[:, layer, :]


def validate_embeddings(embeddings: np.ndarray) -> None:
    """Validate the shape expected by the downstream tutorial."""

    if embeddings.ndim != 3:
        raise ValueError("Expected embeddings with shape [files, layers, dimensions]")
    if embeddings.shape[-1] != WHAM_EMBEDDING_DIM:
        raise ValueError(
            f"Expected WhAM dimension {WHAM_EMBEDDING_DIM}, got {embeddings.shape[-1]}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain non-finite values")


def save_embedding_artifact(
    path: Path,
    audio_paths: Iterable[Path],
    embeddings: np.ndarray,
    layer: int,
) -> None:
    """Save paths and vectors in a portable compressed NumPy artifact."""

    validate_embeddings(embeddings)
    audio_paths = list(audio_paths)
    if len(audio_paths) != embeddings.shape[0]:
        raise ValueError("Audio path and embedding counts differ")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        audio_paths=np.asarray([str(Path(item)) for item in audio_paths]),
        embeddings=embeddings.astype(np.float32),
        layer=np.asarray(layer, dtype=np.int64),
    )


def load_embedding_artifact(path: Path) -> dict[str, object]:
    """Load and validate a saved embedding artifact."""

    with np.load(Path(path), allow_pickle=False) as data:
        embeddings = np.asarray(data["embeddings"])
        audio_paths = [Path(item) for item in data["audio_paths"].tolist()]
        layer = int(np.asarray(data["layer"]).item())
    validate_embeddings(embeddings)
    if len(audio_paths) != embeddings.shape[0]:
        raise ValueError("Audio path and embedding counts differ")
    return {
        "audio_paths": audio_paths,
        "embeddings": embeddings,
        "layer": layer,
    }


def reduce_embeddings(
    embeddings: np.ndarray,
    method: str = "pca",
    random_state: int = 42,
) -> np.ndarray:
    """Project vectors to two dimensions with a deterministic small-data setup."""

    from sklearn.decomposition import PCA

    if embeddings.ndim != 2:
        raise ValueError("Expected a matrix with shape [files, dimensions]")
    if embeddings.shape[0] < 2:
        raise ValueError("At least two files are required for a projection")

    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
    elif method == "umap":
        if embeddings.shape[0] < 4:
            raise ValueError("UMAP requires at least four files")
        from umap import UMAP

        reducer = UMAP(
            n_components=2,
            n_neighbors=min(15, embeddings.shape[0] - 1),
            random_state=random_state,
            n_jobs=1,
        )
    elif method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(30, max(1, (embeddings.shape[0] - 1) // 3))
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )
    else:
        raise ValueError(f"Unknown projection method: {method}")
    return reducer.fit_transform(embeddings)


def cluster_embeddings(
    embeddings: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
) -> dict[str, object]:
    """Run a transparent KMeans baseline and report silhouette when defined."""

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    if embeddings.ndim != 2:
        raise ValueError("Expected a matrix with shape [files, dimensions]")
    if n_clusters < 2 or n_clusters >= embeddings.shape[0]:
        raise ValueError("n_clusters must be at least 2 and smaller than file count")

    scaled = StandardScaler().fit_transform(embeddings)
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = model.fit_predict(scaled)
    unique_labels = np.unique(labels)
    silhouette = None
    if 1 < len(unique_labels) < len(labels):
        silhouette = float(silhouette_score(scaled, labels))
    return {
        "labels": labels,
        "model": model,
        "silhouette": silhouette,
    }


def load_annotation_labels(
    annotation_csv: Path,
    audio_paths: Iterable[Path],
    file_column: str = "file",
    label_column: str = "Unit",
) -> tuple[list[Path], np.ndarray]:
    """Join optional annotations to audio paths without inventing labels."""

    import pandas as pd

    table = pd.read_csv(annotation_csv)
    missing = {file_column, label_column} - set(table.columns)
    if missing:
        raise ValueError(f"Annotation CSV is missing columns: {sorted(missing)}")
    names = table[file_column].dropna().map(lambda value: Path(str(value)).name)
    if names.duplicated().any():
        raise ValueError("Annotation filenames must be unique")
    by_name = {
        Path(str(row[file_column])).name: str(row[label_column])
        for _, row in table.iterrows()
        if pd.notna(row[label_column])
    }
    matched_paths = []
    labels = []
    for path in audio_paths:
        name = Path(path).name
        if name in by_name:
            matched_paths.append(Path(path))
            labels.append(by_name[name])
    if not matched_paths:
        raise ValueError("No annotation rows matched the downloaded audio files")
    return matched_paths, np.asarray(labels)


def fit_label_baseline(embeddings: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> dict[str, object]:
    """Evaluate a baseline with recording groups held apart across the split."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.pipeline import make_pipeline

    if embeddings.shape[0] != len(labels):
        raise ValueError("Embedding and label counts differ")
    if len(groups) != len(labels) or any(not str(group).strip() for group in groups):
        raise ValueError("A nonempty recording group is required for each label")
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(labels)
    if len(encoder.classes_) < 2:
        raise ValueError("At least two social-unit classes are required")
    train_indices, test_indices = next(
        GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
            embeddings, encoded, groups
        )
    )
    train_x, test_x = embeddings[train_indices], embeddings[test_indices]
    train_y, test_y = encoded[train_indices], encoded[test_indices]
    if len(np.unique(train_y)) < 2:
        raise ValueError("Training groups must contain at least two classes")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2_000, random_state=42),
    )
    model.fit(train_x, train_y)
    predicted = model.predict(test_x)
    report = classification_report(
        test_y,
        predicted,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0,
        output_dict=True,
    )
    return {
        "model": model,
        "classes": encoder.classes_,
        "report": report,
        "train_indices": train_indices,
        "test_indices": test_indices,
    }


def fit_social_unit_baseline(
    embeddings: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
) -> dict[str, object]:
    """Fit the optional social-unit baseline when verified Unit labels exist."""

    return fit_label_baseline(embeddings, labels, groups)
