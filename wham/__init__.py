from pathlib import Path

WHAM_DIR = Path(__file__).resolve().parent.parent

WHAM_DATA_DIR = WHAM_DIR / "data"
PLOTS_DIR = WHAM_DATA_DIR / "plots"
EMBEDDING_PLOTS_DIR = PLOTS_DIR / "embeddings"

WHAM_AUDIO_DIR = WHAM_DATA_DIR / "audio"
SPLITS_DIR = WHAM_AUDIO_DIR / "splits"
TRAIN_VAL_DIR = SPLITS_DIR / "train_val"
TRAIN_DIR = TRAIN_VAL_DIR / "train"
VAL_DIR = TRAIN_VAL_DIR / "val"


VAMPNET_DIR = WHAM_DIR / "vampnet"

#Generation Testing Paths
MARINE_TEST_DIR = WHAM_DATA_DIR / "testing_data" / "marine_mammels" / "data"

#Generation Training paths
DENOISED_DATA_DIR = WHAM_DATA_DIR / "testing_data" / "codas_denoised"
CODA_DIR = MARINE_TEST_DIR / "codas"
REGEN_CODA_DIR = WHAM_DATA_DIR / "testing_data" / "regenerated_codas"
CSV_PATH = WHAM_DATA_DIR / "training_data" / "allcodas.csv"

#Embeddings Directory
CODA_EMBEDDING_DIR = WHAM_DATA_DIR / "testing_data" / "coda_embeddings"
EVALUATE_EMBEDDING_DIR = WHAM_DATA_DIR / "testing_data" / "comparison_embeddings"

GENERATED_MARINE_PATH_TEMP = WHAM_DATA_DIR / "testing_data" / "temp"
GENERATED_MARINE_PATH = WHAM_DATA_DIR / "testing_data" / "generated_marine_mammels"

IMPULSE_DIR = GENERATED_MARINE_PATH / "impulses"