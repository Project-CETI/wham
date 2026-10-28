# Validation record

Date: 2026-10-25

## Environment

- Repository commit: `5802f42` before this validation note
- Python: 3.12
- PyTorch: 2.9.1+cpu
- Device: CPU
- Input: public DSWP `1.wav`, plus `10.wav` and `100.wav` for the notebook run
- Model path: official VampNet `Interface` with `coarse.pth` and `codec.pth`
- Optional `c2f.pth` and `wavebeat.pth`: not loaded

## Checkpoint integrity

The downloaded files were checked against the Zenodo record:

| File | Bytes | MD5 |
| --- | ---: | --- |
| `coarse.pth` | 1,343,719,071 | `0c1194ac517cd969aee295a362f60761` |
| `codec.pth` | 600,996,465 | `478875644e1591ea771903c18e8b71c7` |

## Results

- Official interface load: 9.3 seconds on CPU
- One-file extraction: finite embedding with shape `(1, 1280)` from layer 10
- One-file artifact: `work/artifacts/real_dswp_1_wham_layer10.npz`
- Full notebook execution with three public DSWP files: passed
- Public-label social-unit and coda classification: intentionally skipped because the DSWP release has no matching annotation table

## Follow-up changes

The tutorial now pins the DSWP dataset revision, rejects duplicate annotation filenames and separates source recording groups in the optional classifier. Synthetic tests check the group split and artifact integrity. Verified recording groups and labels were not available, so neither supervised task was evaluated on real annotations. Dependencies are not locked and the root installation procedure has not been tested in a clean environment.

On 2026-10-28, the updated notebook was executed with the same local Python 3.12 environment and CPU checkpoints. It produced three layer-10 vectors with shape `(3, 1280)` and ran PCA and KMeans. Eight tutorial tests passed. This is a smoke test, not an evaluation of cluster quality or a clean-install test.

The artifact and downloaded weights stay outside the commit. The notebook downloads or reuses them locally and keeps the supervised claims gated on explicit annotations.

Sources: [WhAM](https://github.com/Project-CETI/wham), [paper](https://arxiv.org/abs/2512.02206), [DSWP](https://huggingface.co/datasets/orrp/DSWP), [weights](https://doi.org/10.5281/zenodo.17633708).
