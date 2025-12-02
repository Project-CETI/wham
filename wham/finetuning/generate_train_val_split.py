from wham import TRAIN_DIR, VAL_DIR, TRAIN_VAL_DIR
from codatools.codatools import SNIPPETS_DIR
import argparse
import logging
from shutil import rmtree

from sklearn.model_selection import train_test_split


def make_dirs():
    logging.info(f"Train/val split in {TRAIN_VAL_DIR}")
    if TRAIN_VAL_DIR.exists():
        rmtree(TRAIN_VAL_DIR)
        logging.info(f"Cleared {TRAIN_VAL_DIR}")
    TRAIN_DIR.mkdir(parents=True)
    VAL_DIR.mkdir(parents=True)


def make_train_val_split(n_train, n_val):
    all_sliced = [p for p in SNIPPETS_DIR.glob("*")]
    train, val = train_test_split(all_sliced, test_size=n_val, train_size=n_train)
    for p in train:
        (TRAIN_DIR / p.name).symlink_to(p)
        logging.debug(f"{p} --> {TRAIN_DIR / p.name}")
    for p in val:
        (VAL_DIR / p.name).symlink_to(p)
        logging.debug(f"{p} --> {VAL_DIR / p.name}")


if __name__ == "__main__":
    if not SNIPPETS_DIR.exists() or not SNIPPETS_DIR.is_dir():
        raise FileNotFoundError(f"Expected audio snippets at {SNIPPETS_DIR}!")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_train", help="Number of training samples", type=int, required=True
    )
    parser.add_argument(
        "--n_val", help="Number of validation samples", type=int, required=True
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    make_dirs()
    make_train_val_split(args.n_train, args.n_val)

    # embeddings_script = f"python scripts/utils/gtzan_embeddings.py"
    # embeddings_args_string = f"--args.load conf/interface.yml --Interface.device cuda --path_to_gtzan {labeling_dir} --output_dir {plot_dir}"
    # embeddings_command = f"{embeddings_script} {embeddings_args_string}"
    # logging.debug(f"cd {VAMPNET_DIR}")
    # os.chdir(VAMPNET_DIR)
    # logging.debug(f"Running {embeddings_command}")
    # os.system(embeddings_command)
    # logging.info(f"Embeddings saved to {plot_dir}")
