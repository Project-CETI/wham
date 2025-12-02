import os
import numpy as np
from scipy.linalg import sqrtm
from wam import CODA_EMBEDDING_DIR, EVALUATE_EMBEDDING_DIR
import argparse

"""
In order to calculate the FAD using custom embeddings, it was necessary to create our own script to calculate
the FAD score. Standard packages either don't support, or don't make it simple to calculate the FAD with pre-calculated embeddings
"""

def load_embeddings(directory):
    # Load all .npy files in a directory and stack them
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".npy")]
    all_embeds = [np.load(f) for f in files]
    if not all_embeds:
        return None
    return np.vstack(all_embeds)

def compute_stats(embeddings):
    #Given the embeddings, calculate mean and standard deviation according to the FAD algorithm
    mu = np.mean(embeddings, axis=0)
    sigma = np.cov(embeddings, rowvar=False)
    return mu, sigma

def calculate_fad(mu1, sigma1, mu2, sigma2):
    """
    Takes the statistics from the two groups of embeddings and calculates the FAD score between them

    Parameters:
        mu1 (float): Mean of the first set of embeddings
        sigma1 (str): Standard deviation of the first set of embeddings
        mu2 (float): Mean of the second set of embeddings
        sigma2 (str): Standard deviation of the second set of embeddings
    """
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2 * covmean)



def main(Eval_Dir):
    """
    Parameters:
        Eval_Dir (str): Directory containing embeddings of each audio file in the evaluation set, every embedding
            should be saved as a .npy file
    """
    # Load coda embeddings
    coda_embeddings = load_embeddings(CODA_EMBEDDING_DIR)
    if coda_embeddings is None:
        raise RuntimeError("Coda embeddings not found.")
    mu1, sigma1 = compute_stats(coda_embeddings)

    # Go through each subdirectory in gen_embeddings
    for subfolder in os.listdir(EVALUATE_EMBEDDING_DIR):
        subdir_path = os.path.join(EVALUATE_EMBEDDING_DIR, subfolder)
        if not os.path.isdir(subdir_path):
            continue

        test_embeds = load_embeddings(subdir_path)
        if test_embeds is None:
            print(f"[{subfolder}] No .npy files found, skipping.")
            continue

        mu2, sigma2 = compute_stats(test_embeds)
        fad_score = calculate_fad(mu1, sigma1, mu2, sigma2)
        print(f"[{subfolder}] FAD: {fad_score:.4f}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation_dir", type=str, default=EVALUATE_EMBEDDING_DIR, help="Path to the the directory containing embeddings for the evaluation set")
    args = parser.parse_args()
    main(args.evaluation_dir)