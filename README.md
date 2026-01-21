# WhAM: a Whale Acoustics Model
[![arXiv](https://img.shields.io/badge/arXiv-2512.02206-b31b1b.svg)](https://arxiv.org/abs/2512.02206)
[![Model Weights](https://img.shields.io/badge/Zenodo-Model%20Weights-blue.svg)](https://doi.org/10.5281/zenodo.17633708)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DSWP%20Dataset-yellow)](https://huggingface.co/datasets/orrp/DSWP)
![WhAM](assets/inference.png "WhAM")
WhAM is a transformer-based audio-to-audio model designed to synthesize and analyze sperm whale codas. Based on [VampNet](https://github.com/hugofloresgarcia/vampnet), WhAM uses masked acoustic token modeling to capture temporal and spectral features of whale communication. WhAM generates codas from a given audio context, enabling three core capabilities:

 - Acoustic Translation: The ability to style-transfer arbitrary audio prompts (e.g., human speech, noise) into the acoustic texture of sperm whale codas.

 - Synthesizing novel "pseudocodas".

 - Providing audio embeddings for downstream tasks such as social unit and spectral feature ("vowel") classification.

See our [NeurIPS 2025](https://openreview.net/pdf?id=IL1wvzOgqD) publication for more details.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Project-CETI/wham.git
    cd wham
    ```

2.  **Set up the environment:**
    ```bash
    conda create -n wham python=3.9
    conda activate wham
    ```

3.  **Install dependencies:**
    ```bash
    # Install the wham package
    pip install -e .

    # Install VampNet
    pip install -e ./vampnet

    # Install madmom
    pip install --no-build-isolation madmom

    # Install ffmpeg
    conda install -c conda-forge ffmpeg
    ```

4.  **Download model weights:**
    Download the [weights](https://zenodo.org/records/17633708) and extract to `vampnet/models/`.

## Generation

To run WhAM locally and prompt it in your browser:

```bash
python vampnet/app.py --args.load conf/interface.yml --Interface.device cuda
```

This will provide you with a Gradio link to test WhAM on inputs of your choice.

## Training Data

![Training](assets/training.png "Training")

You only need to follow these to fine-tune your own version of WhAM. First, obtain the original VampNet weights by following the instructions in the ![original repo](https://github.com/hugofloresgarcia/vampnet/tree/ismir-2023). Download 
c2f.pth and codec.pth and replace the weights you previously downloaded in `vampnet/models`.

Second, obtain data:

1.  **Domain adaptation data:**
   
    - Download audio samples from the [WMMS 'Best Of' Cut](https://whoicf2.whoi.edu/science/B/whalesounds/index.cfm). Save them under `vampnet/training_data/domain_adaptation`.
    
    - Download audio samples from the [BirdSet Dataset](https://huggingface.co/datasets/DBD-research-group/BirdSet). Save these under the same directory

    - Finally, download all samples from the [AudioSet Dataset](https://research.google.com/audioset/ontology/index.html) with the label `Animal` and once again save these into the directory

3.  **Species-specific finetuning:** Finetuning can be performed on the openly available **[Dominica Sperm Whale Project (DSWP)](https://huggingface.co/datasets/orrp/DSWP)** dataset, available on Hugging Face.


With data in hand, navigate into `vampnet` and perform Domain Adaptation:
```bash
python vampnet/scripts/exp/fine_tune.py "training_data/domain_adaptation" domain_adapted && python vampnet/scripts/exp/train.py --args.load conf/generated/domain_adapted/coarse.yml && python vampnet/scripts/exp/train.py --args.load conf/generated/domain_adapted/c2f.yml
```

Then fine-tune the domain-adapted model. Create the config file with the command:

```bash
python vampnet/scripts/exp/fine_tune.py "training_data/species_specific_finetuning" fine-tuned
```

To select which weights you want to use as a checkpoint, change `fine_tune_checkpoint` in `conf/generated/fine-tuned/[c2f/coarse].yml` to `./runs/domain_adaptation/[coarse/c2f]/[checkpoint]/vampnets/weights.pth`. `[checkpoint]` can be `latest` in order to use the last saved checkpoint from the previous run, though it is recommended to manually verify the quality of generations over various checkpoints as overtraining can often cause degradation in audio quality, especially with smaller datasets. After making that change, run the command:

```bash
python vampnet/scripts/exp/train.py --args.load conf/generated/fine-tuned/coarse.yml && python vampnet/scripts/exp/train.py --args.load conf/generated/fine-tuned/c2f.yml
```

After following these steps, you should be able to generate audio via the browser by running:
```bash
python app.py --args.load vampnet/conf/generated/fine-tuned/interface.yml
```

**Note**: The coarse and fine weights can be trained separately if compute allows. In this case, you would call the two scripts:

```bash
python vampnet/scripts/exp/train.py --args.load conf/generated/[fine-tuned/domain_adaptated]/coarse.yml
```

```bash
python vampnet/scripts/exp/train.py --args.load conf/generated/[fine-tuned/domain_adaptated]/c2f.yml
```

After both are finished running, ensure that both resulting weights are copied into the same copy of WhAM.



## Testing Data

1.  **Marine Mammel Data:**
    Download audio samples from the [WMMS 'Best Of' Cut](https://whoicf2.whoi.edu/science/B/whalesounds/index.cfm). Save them under `data/testing_data/marine_mammals/data/[SPECIES_NAME]`.
    * `[SPECIES_NAME]` must match the species names found in `wham/generation/prompt_configs.py`.
      
2.  **Sperm Whale Codas:**
    To evaluate on sperm whale codas, you can use the openly available [DSWP](https://huggingface.co/datasets/orrp/DSWP) dataset.

3. Generate artifical beeps for experiments. `data/generate_beeps.sh`


## Reproducing Paper Results
Note: Access to the DSWP+CETI annotated is required to reproduce all results; as of time of publication, only part of this data is publicly available. Still, we include the following code as it may be useful for researchers who may benefit from our evaluation pipeline.

### 1. Downstream Classification Tasks
To reproduce **Table 1** (Classification Accuracies) and **Figure 7** (Ablation Study):

**Table 1 Results:**
```bash
cd wham/embedding
./downstream_tasks.sh
```
* Runs all downstream classification tasks.
* **Baselines:** Run once.
* **Models (AVES, VampNet):** Run over 3 random seeds; reports mean and standard deviation.

**Figure 7 Results (Ablation):**
```bash
cd wham/embedding
./downstream_ablation.sh
```
* Outputs accuracy scores for ablation variants (averaged across 3 seeds with error bars).

### 2. Generative Metrics

**Figure 12: Frechet Audio Distance (FAD) Scores**
Calculate the distance between WhAM's generated results and real codas:
```bash
# Calculate for all species
bash wham/generation/eval/calculate_FAD.sh

# Calculate for a single species
bash wham/generation/eval/calculate_FAD.sh [species_name]
```
* *Runtime:* ~3 hours on an NVIDIA A10 GPU.

**Figure 3: FAD with Custom/BirdNET Embeddings**
To compare against other embeddings:
1.  Convert your `.wav` files to `.npy` embeddings.
2.  Place raw coda embeddings in: `data/testing_data/coda_embeddings`
3.  Place comparison embeddings in subfolders within: `data/testing_data/comparison_embeddings`
4.  Run:
    ```bash
    python wham/generation/eval/calculate_custom_fad.py
    ```
*For BirdNET embeddings, refer to the [official repo](https://github.com/BirdNET-Team/BirdNET-Analyzer).*

**Table 2: Embedding Type Ablation**
Calculate distances between raw codas, denoised versions, and noise profiles:
```bash
bash wham/generation/eval/FAD_ablation.sh
```
* *Prerequisites:* Ensure `data/testing_data/ablation/noise` and `data/testing_data/ablation/denoised` are populated.
* *Runtime:* ~1.5 hours on an NVIDIA A10 GPU.

**Figure 13: Tokenizer Reconstruction**
Test the mean squared reconstruction error:
```bash
bash wham/generation/eval/evaluate_tokenizer.sh
```

---

## Citation

Please use the following citation if you use this code, model or data.

```bibtex
@inproceedings{wham2025,
  title={Towards A Translative Model of Sperm Whale Vocalization},
  author={Orr Paradise, Pranav Muralikrishnan, Liangyuan Chen, Hugo Flores Garcia, Bryan Pardo, Roee Diamant, David F. Gruber, Shane Gero, Shafi Goldwasser},
  booktitle={Advances in Neural Information Processing Systems 39: Annual Conference
                  on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego, CA, USA},
  year={2025}
}
```
