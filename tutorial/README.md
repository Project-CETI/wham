# DSWP to WhAM embeddings tutorial

`wham_dswp_embeddings.ipynb` is a small, reproducible starter pipeline:

1. Download selected public DSWP WAV files from Hugging Face
2. Inspect their audio metadata
3. Extract mean-pooled layer embeddings with the official VampNet interface
4. Run PCA, UMAP or t-SNE and a KMeans baseline
5. Run optional coda or social-unit classifiers only when verified annotation labels are supplied

The public DSWP release has 1,501 audio files and a single `audio` feature. Its card says that there is no per-file metadata and that the recordings come from one vocal clan. The notebook does not invent labels or source recording identifiers. Each supervised section requires a separate verified CSV with `file`, `recording_group`, and the relevant `Unit` or `coda_label` column. Groups must identify independent source recordings or a larger appropriate unit. A group split reduces one source of leakage but cannot establish independence without verified provenance. Coda detection also requires positive and negative examples.

## Run

Follow the repository root installation instructions first, including the WhAM and VampNet packages and ffmpeg. Then, from the repository root:

```bash
python -m pip install -r tutorial/requirements.txt
jupyter notebook tutorial/wham_dswp_embeddings.ipynb
```

For real WhAM embeddings, download the official `coarse.pth` and `codec.pth` into `vampnet/models`. The notebook keeps model extraction disabled until both files are present. These files are large, so the default first run only validates public DSWP ingestion and the pipeline plumbing.

The sample dataset revision is pinned in `wham_dswp.py`. The dependencies are not locked, and a clean installation has not yet been verified. A CPU run is intended for a tiny smoke sample. A larger embedding sweep should use a compatible GPU environment.

The first real CPU smoke run is recorded in [VALIDATION.md](VALIDATION.md). It loaded the official coarse and codec weights, extracted a finite layer-10 vector with shape `(1, 1280)` from public DSWP audio, and executed the notebook on three files.

## Sources

- [WhAM repository](https://github.com/Project-CETI/wham)
- [WhAM paper](https://arxiv.org/abs/2512.02206)
- [Public DSWP dataset](https://huggingface.co/datasets/orrp/DSWP)
- [WhAM model weights](https://doi.org/10.5281/zenodo.17633708)
