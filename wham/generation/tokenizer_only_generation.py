from pathlib import Path
from typing import Tuple
import uuid
from dataclasses import dataclass, asdict
import torch

import sys
import numpy as np
import audiotools as at
import argbind
import argparse

from vampnet.interface import Interface
from vampnet import mask as pmask
from prompt_configs import CONFIGS
from wam.utils import move_wav_files

from wam import GENERATED_MARINE_PATH_TEMP, MARINE_TEST_DIR, VAMPNET_DIR, REGEN_CODA_DIR

import os

#Change working directory to vampnet folder so that it works with vampnet code
os.chdir(VAMPNET_DIR)


"""
Modified version of generate output snippets. This version of the code takes an audio file, encodes it using the
DAC codec, then decodes it, skipping the generation portion of WhAM. 
"""


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
    seed = 1
    print(data["input_audio"])
    if seed != 0:
        at.util.seed(data["seed"])

    out_dir = OUT_DIR / str(uuid.uuid4())
    out_dir.mkdir()

    sig = at.AudioSignal(str(data["input_audio"]))
    sig = interface.preprocess(sig)

    loudness = sig.loudness()
    print(f"input loudness is {loudness}")

    if data["pitch_shift_amt"] != 0:
        sig = shift_pitch(sig, data["pitch_shift_amt"])

    z = interface.encode(sig)


    sig = interface.to_signal(z).cpu()

    sig = sig.normalize(loudness)    

    sig.write(out_dir / out_name)

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


    for i, fname in enumerate(Path(input_dir).glob("*.wav")):
        if i >= 1000:
            break
            
        out_name = f"{fname.stem}gen.wav"
        config["input_audio"] = str(fname)
        print(f"Processing {fname}")

        
        
        audio = _vamp(config, out_dir, out_name)
          
@argbind.bind()
def main(species: str = None):
    generate_batch(
        GENERATED_MARINE_PATH_TEMP / "tokenizer_only_codas",
        MARINE_TEST_DIR / "codas",
        CONFIGS['codas']
    )
    move_wav_files(GENERATED_MARINE_PATH_TEMP / "tokenizer_only_codas", REGEN_CODA_DIR)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Interface = argbind.bind(Interface)

if __name__ == "__main__":
    with argbind.scope(conf):
        interface = Interface().to(device)
        main()
