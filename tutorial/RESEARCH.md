# WhAM tutorial research notes

Checked on 2026-10-25 against the official repository, paper, weights record and public dataset.

## Verified facts

- The official code obtains full VampNet/WhAM activations with dimension `1280`. Its downstream classifier mean-pools the time axis. Codec-only is `1024` and AVES is `768`.
- The official `Interface` resamples audio to the codec sample rate, converts it to mono and normalizes it before encoding.
- The official weights record contains `coarse.pth`, `c2f.pth`, `codec.pth` and optional `wavebeat.pth`. The tutorial needs only the coarse model and codec.
- The public DSWP release contains 1,501 WAV files, about 45 minutes of audio and a single public `audio` feature. The dataset card says the release has no per-file metadata and comes from one vocal clan and one geographic region.
- The public release does not expose the `Unit` labels used by the official downstream classifier and does not provide the negative examples needed for coda detection. Both supervised tasks are therefore optional and require a separately verified annotation CSV.

## Interpretation boundary

PCA, UMAP, t-SNE and KMeans on public DSWP embeddings are exploratory representation analysis. They do not establish social-unit separation. A supervised score is only reported after explicit labels are joined from an external annotation table.

## Sources

- [WhAM README](https://github.com/Project-CETI/wham/blob/main/README.md)
- [WhAM embedding models](https://github.com/Project-CETI/wham/blob/main/wham/embedding/models.py)
- [VampNet interface](https://github.com/Project-CETI/wham/blob/main/vampnet/vampnet/interface.py)
- [WhAM paper](https://arxiv.org/abs/2512.02206)
- [DSWP dataset card](https://huggingface.co/datasets/orrp/DSWP)
- [DSWP dataset README](https://huggingface.co/datasets/orrp/DSWP/raw/main/README.md)
- [WhAM model weights](https://doi.org/10.5281/zenodo.17633708)
