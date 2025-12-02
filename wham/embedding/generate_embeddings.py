from collections import Counter
from pathlib import Path
import pandas as pd
from wam import VAMPNET_DIR, EMBEDDING_PLOTS_DIR, SNIPPETS_DIR, DENOISED_DATA_DIR, CSV_PATH
from wam.utils import pathify
import argparse
import logging
from shutil import rmtree

import os


def make_labeling_dir(column_name):
    labeling_dir = DENOISED_DATA_DIR / pathify(column_name)
    logging.info(f"Labeling according to {column_name}")
    if labeling_dir.exists():
        rmtree(labeling_dir)
        logging.info(f"Cleared {labeling_dir}")
    return labeling_dir


def label_sliced_codas(column_name, labeling_dir, all_codas):
    if column_name not in all_codas.columns:
        raise ValueError(f"{column_name} not a column of {CSV_PATH}!")

    counter = Counter()
    for p in SNIPPETS_DIR.glob("*"):
        id = int(p.stem) - 1
        label = all_codas.loc[all_codas.index[id], column_name]
        if pd.isna(label) or label == " ":
            counter["!!unlabeled"] += 1
            continue
        label = pathify(str(label))
        dir = labeling_dir / label
        dir.mkdir(parents=True, exist_ok=True)
        (dir / p.name).symlink_to(p)
        logging.debug(f"{p} --> {dir / p.name}")
        counter[label] += 1
    logging.info(dict(counter))
    logging.info(f"Total: {sum(v for v in counter.values())}")


if __name__ == "__main__":
    if not SNIPPETS_DIR.exists() or not SNIPPETS_DIR.is_dir():
        raise FileNotFoundError(f"Expected audio snippets at {SNIPPETS_DIR}!")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--column_name",
        "-n",
        help="Column name in the CSV according to which codas are labels",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--interface",
        "-i",
        help="Path to VampNet model interface.yml (absolute, or relative to vampnet/)",
        type=str,
        default="conf/interface.yml",
    )
    parser.add_argument(
        "--clear_cache",
        "-cc",
        help="Clear the cache of embeddings before generating new ones",
        action="store_true",
    )
    parser.add_argument(
        "--method",
        "-m",
        help="Method to use for embedding visualization",
        type=str,
        default="tsne",
    )
    parser.add_argument(
        "--n_components",
        "-nc",
        help="Number of components to use for embedding visualization",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--output_suffix",
        "-os",
        help="Suffix to append to output directory",
        type=str,
        default="",
    )
    parser.add_argument(
        "--layer",
        "-l",
        help="Layer to use for embedding visualization (blank for [1, 3, ..., 19])",
        type=int,
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    all_codas = pd.read_csv(CSV_PATH, low_memory=False)
    labeling_dir = make_labeling_dir(args.column_name)
    label_sliced_codas(args.column_name, labeling_dir, all_codas)

    plot_dir = EMBEDDING_PLOTS_DIR / f"{args.column_name}_{args.method}{args.n_components}_{args.output_suffix}"
    embeddings_script = f"python scripts/utils/visualize_embeddings.py"
    embeddings_args_string = f"--args.load {args.interface} --Interface.device cuda --path_to_audio {labeling_dir} --output_dir {plot_dir} --method {args.method} --n_components {args.n_components}"
    if args.layer is not None:
        embeddings_args_string += f" --layer {args.layer}"
    embeddings_command = f"{embeddings_script} {embeddings_args_string}"
    logging.debug(f"cd {VAMPNET_DIR}")
    os.chdir(VAMPNET_DIR)
    if args.clear_cache:
        logging.info(f"Clearing embeddings cache")
        os.system("rm -rf ./.emb_cache")
    logging.debug(f"Running {embeddings_command}")
    status = os.system(embeddings_command)
    if os.WEXITSTATUS(status) != 0:
        raise RuntimeError(f"Embeddings command failed with status {status}")
    logging.info(f"Embeddings saved to {plot_dir}")
