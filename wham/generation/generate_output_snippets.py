from pathlib import Path
from typing import Tuple
import uuid
from dataclasses import dataclass, asdict

import sys
import numpy as np
import audiotools as at
import argbind

from vampnet.interface import Interface
from vampnet import mask as pmask
from prompt_configs import CONFIGS
from wam.utils import move_wav_files
import torch

from wam import GENERATED_MARINE_PATH, GENERATED_MARINE_PATH_TEMP, MARINE_TEST_DIR, VAMPNET_DIR

import os

#Change working directory to vampnet folder so that it works with vampnet code
os.chdir(VAMPNET_DIR)




conf = argbind.parse_args()


from torch_pitch_shift import pitch_shift, get_fast_shifts

def shift_pitch(signal, interval: int):
    signal.samples = pitch_shift(
        signal.samples, 
        shift=interval, 
        sample_rate=signal.sample_rate
    )
    return signal







def _vamp(data, OUT_DIR, out_name, return_mask=False):
    use_coarse2fine = True
    seed = 1
    
    if seed != 0:
        at.util.seed(data["seed"])

    out_dir = OUT_DIR / str(uuid.uuid4())
    out_dir.mkdir()

    sig = at.AudioSignal(data["input_audio"])
    sig = interface.preprocess(sig)

    loudness = sig.loudness()
    print(f"input loudness is {loudness}")

    if data["pitch_shift_amt"] != 0:
        sig = shift_pitch(sig, data["pitch_shift_amt"])

    z = interface.encode(sig)

    ncc = data["n_conditioning_codebooks"]

    # build the mask
    mask = pmask.linear_random(z, data["rand_mask_intensity"])
    mask = pmask.mask_and(
        mask, pmask.inpaint(
            z,
            interface.s2t(data["prefix_s"]),
            interface.s2t(data["suffix_s"])
        )
    )
    mask = pmask.mask_and(
        mask, pmask.periodic_mask(
            z,
            data["periodic_p"],
            data["periodic_w"],
            random_roll=True
        )
    )
    if data["onset_mask_width"] > 0:
        mask = pmask.mask_or(
            mask, pmask.onset_mask(sig, z, interface, width=data["onset_mask_width"])
        )
    if data["beat_mask_width"] > 0:
        beat_mask = interface.make_beat_mask(
            sig,
            after_beat_s=(data["beat_mask_width"]/1000), 
            mask_upbeats=not data["beat_mask_downbeats"],
        )
        mask = pmask.mask_and(mask, beat_mask)

    # these should be the last two mask ops
    mask = pmask.dropout(mask, data["dropout"])
    mask = pmask.codebook_unmask(mask, ncc)
    mask = pmask.codebook_mask(mask, int(data["n_mask_codebooks"]))



    
    
    _top_p = data["top_p"] if data["top_p"] > 0 else None
    # save the mask as a txt file
    np.savetxt(out_dir / "mask.txt", mask[:,0,:].long().cpu().numpy())

    _seed = data["seed"] if data["seed"] > 0 else None
    zv, mask_z = interface.coarse_vamp(
        z, 
        mask=mask,
        sampling_steps=data["num_steps"],
        mask_temperature=data["masktemp"]*10,
        sampling_temperature=data["sampletemp"],
        return_mask=True, 
        typical_filtering=data["typical_filtering"], 
        typical_mass=data["typical_mass"], 
        typical_min_tokens=data["typical_min_tokens"], 
        top_p=_top_p,
        gen_fn=interface.coarse.generate,
        seed=_seed,
        sample_cutoff=data["sample_cutoff"],
    )

    if use_coarse2fine: 
        zv = interface.coarse_to_fine(
            zv, 
            mask_temperature=data["masktemp"]*10, 
            sampling_temperature=data["sampletemp"],
            mask=mask,
            sampling_steps=data["num_steps"],
            sample_cutoff=data["sample_cutoff"], 
            seed=_seed,
        )

    sig = interface.to_signal(zv).cpu()

    sig = sig.normalize(loudness)    

    sig.write(out_dir / out_name)

    if return_mask:
        mask = interface.to_signal(mask_z).cpu()
        mask.write(out_dir / "mask.wav")
        return sig.path_to_file, mask.path_to_file
    else:
        return sig.path_to_file
    

def generate_batch(out_dir: str, input_dir: str, config: dict) -> None:
    """Generate samples for a batch of audio files.
    
    Parameters:
        out_dir: Output directory path
        input_dir: Input directory containing audio files
        config: Generation parameters
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    print(input_dir)
    for i, fname in enumerate(Path(input_dir).glob("*.wav")):
        if i >= 1000:
            break
            
        out_name = f"{fname.stem}b.wav"
        config["input_audio"] = str(fname)
        print(f"Processing {fname}")

        try:
            audio = _vamp(config, out_dir, out_name)
        except Exception as e:
            print(f"{out_name} FAILED: {str(e)}")
            continue

@argbind.bind()
def main(species: str = None):
    if species:
        if species not in CONFIGS:
            print(f"Error: species '{species}' not found in CONFIG.")
            return
        else:
            dataset = species
            config = CONFIGS[dataset]
            generate_batch(
                GENERATED_MARINE_PATH_TEMP / dataset,
                MARINE_TEST_DIR / dataset,
                config
            )
            move_wav_files(GENERATED_MARINE_PATH_TEMP / dataset, GENERATED_MARINE_PATH / dataset)
    else:
        # Process each dataset
        for dataset, config in CONFIGS.items():
            print(f"\nProcessing {dataset}...")
            generate_batch(
                GENERATED_MARINE_PATH_TEMP / dataset,
                MARINE_TEST_DIR / dataset,
                config
            )
            move_wav_files(GENERATED_MARINE_PATH_TEMP / dataset, GENERATED_MARINE_PATH / dataset)


Interface = argbind.bind(Interface)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    conf = argbind.parse_args()
    with argbind.scope(conf):
        interface = Interface().to(device)
        main()
