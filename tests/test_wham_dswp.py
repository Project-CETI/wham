from pathlib import Path

import numpy as np
import pytest

from tutorial.wham_dswp import (
    WHAM_EMBEDDING_DIM,
    cluster_embeddings,
    fit_label_baseline,
    load_annotation_labels,
    load_embedding_artifact,
    reduce_embeddings,
    save_embedding_artifact,
    validate_embeddings,
)


def test_validate_embeddings_accepts_wham_shape():
    embeddings = np.zeros((4, 3, WHAM_EMBEDDING_DIM), dtype=np.float32)
    validate_embeddings(embeddings)


def test_validate_embeddings_rejects_wrong_dimension():
    embeddings = np.zeros((4, 3, 16), dtype=np.float32)
    with pytest.raises(ValueError, match="Expected WhAM dimension"):
        validate_embeddings(embeddings)


def test_pca_and_kmeans_are_reproducible():
    rng = np.random.default_rng(42)
    embeddings = rng.normal(size=(12, WHAM_EMBEDDING_DIM)).astype(np.float32)
    first_projection = reduce_embeddings(embeddings, method="pca")
    second_projection = reduce_embeddings(embeddings, method="pca")
    assert np.allclose(first_projection, second_projection)

    result = cluster_embeddings(embeddings, n_clusters=3)
    assert result["labels"].shape == (12,)
    assert -1.0 <= result["silhouette"] <= 1.0


def test_annotation_join_requires_explicit_columns(tmp_path: Path):
    annotation_path = tmp_path / "annotations.csv"
    annotation_path.write_text("file,Unit\n1.wav,unit-a\n", encoding="utf-8")
    paths, labels = load_annotation_labels(annotation_path, [tmp_path / "1.wav"])
    assert paths == [tmp_path / "1.wav"]
    assert labels.tolist() == ["unit-a"]


def test_annotation_join_rejects_duplicate_names(tmp_path: Path):
    annotation_path = tmp_path / "annotations.csv"
    annotation_path.write_text("file,Unit\n1.wav,a\n1.wav,b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_annotation_labels(annotation_path, [tmp_path / "1.wav"])


def test_group_split_keeps_source_recordings_apart():
    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(40, 8))
    groups = np.repeat(np.arange(10), 4)
    labels = np.tile(["a", "b", "a", "b"], 10)
    result = fit_label_baseline(embeddings, labels, groups)
    train_groups = set(groups[result["train_indices"]])
    test_groups = set(groups[result["test_indices"]])
    assert train_groups.isdisjoint(test_groups)
    assert train_groups and test_groups


def test_artifact_rejects_mismatched_path_count(tmp_path: Path):
    embeddings = np.zeros((2, 1, WHAM_EMBEDDING_DIM), dtype=np.float32)
    path = tmp_path / "vectors.npz"
    with pytest.raises(ValueError, match="counts differ"):
        save_embedding_artifact(path, [Path("1.wav")], embeddings, 10)
    np.savez_compressed(path, embeddings=embeddings, audio_paths=["1.wav"], layer=10)
    with pytest.raises(ValueError, match="counts differ"):
        load_embedding_artifact(path)


def test_cluster_reports_undefined_silhouette_for_identical_vectors():
    result = cluster_embeddings(np.zeros((4, 3)), n_clusters=2)
    assert result["silhouette"] is None


def test_small_sample_projection_methods():
    embeddings = np.random.default_rng(9).normal(size=(2, 8))
    assert reduce_embeddings(embeddings, method="tsne").shape == (2, 2)
    with pytest.raises(ValueError, match="at least four files"):
        reduce_embeddings(embeddings, method="umap")
