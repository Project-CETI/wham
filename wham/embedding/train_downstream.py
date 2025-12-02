from wham.embedding.models import *
import torch
import pandas as pd
import numpy as np
from wham.embedding import VAMPNET_MDL_PATH, VAMPNET_SNIP2K_PATH, MODEL_WEIGHTS_PATH, CSV_PATH, CODA_DIR
import os
from preprocessing import preprocess_audio
import argparse
import wandb
import datetime
import random
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LabelType:
    CODA_DETECTOR = 0
    CODA_CLASSIFIER = 1
    SOCIAL_UNIT = 2

def get_vampnet_ckpt_path(backbone : str):
    if backbone == "vampnet-best":
        return "best_vampnet_coarse/vampnet/weights.pth", "best_vampnet_c2f/vampnet/weights.pth", "best_vampnet_coarse/lora.pth", "best_vampnet_c2f/lora.pth"
    coarse_base = VAMPNET_MDL_PATH / "vampnet" / "coarse.pth"
    c2f_base = VAMPNET_MDL_PATH / "vampnet" / "c2f.pth"
    if backbone == "vampnet-base":
        return (coarse_base, c2f_base, None, None)
    number = backbone.split("-")[1]
    if number in ["1", "5", "10", "20", "50"]:
        return (VAMPNET_DIR / "runs" / "1" / "coarse" / f"{number}" / "vampnet" / "weights.pth",
                None,
                VAMPNET_DIR / "runs" / "1" / "coarse" / f"{number}" / "lora.pth",
                None)
    elif number in ["100", "300", "500", "700", "1000", "1500"]:
        return (VAMPNET_SNIP2K_PATH / "coarse" / f"{number}" / "vampnet" / "weights.pth",
                VAMPNET_SNIP2K_PATH / "c2f" / f"{number}" / "vampnet" / "weights.pth",
                VAMPNET_SNIP2K_PATH / "coarse" / f"{number}" / "lora.pth",
                VAMPNET_SNIP2K_PATH / "c2f" / f"{number}" / "lora.pth")
    else:
        return (VAMPNET_MDL_PATH / "finetune_runs" / "snip" / "coarse" / f"{number}" / "vampnet" / "weights.pth",
                VAMPNET_MDL_PATH / "finetune_runs" / "snip" / "c2f" / f"{number}" / "vampnet" / "weights.pth",
                VAMPNET_MDL_PATH / "finetune_runs" / "snip" / "coarse" / f"{number}" / "lora.pth",
                VAMPNET_MDL_PATH / "finetune_runs" / "snip" / "c2f" / f"{number}" / "lora.pth")

# Ensure deterministic results
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define collate function to pad variable length sequences
def collate_fn(batch):
    # Sort batch by sequence length
    batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    # Pad sequences
    x, y = zip(*batch)
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True)
    y = torch.LongTensor(y)
    return x, y


def prepare_dataloader(label_type, csv_path, dir_to_process, column_name, batch_size=32, top_classes=None):
    '''
        Prepares dataloader for training
        Should return train dataloader, validation dataloader, and labels
    '''

    # Strip away suffix
    snippet_ids = sorted([int(snippet[:-4]) for snippet in os.listdir(dir_to_process)])

    labels = pd.read_csv(csv_path, index_col="coda_num")
    labels = labels[column_name]
    labels = labels.dropna()

    # Only keep ones that are in the snippets, and Only keep snippets that have labels
    labels = labels[labels.index.isin(snippet_ids)]
    snippets = np.array([snippet for snippet in snippet_ids if snippet in labels.index])
    labels = labels.to_numpy()

    if label_type == LabelType.CODA_DETECTOR:
        # Convert labels to binary
        label_dict = {label: 0 if "negative" in label else 1 for _, label in enumerate(sorted(np.unique(labels)))}
    elif label_type == LabelType.CODA_CLASSIFIER:
        # Count label frequencies with a Counter
        label_freq = Counter(labels)
        # Get the top classes
        top_classes = set([label for label, _ in label_freq.most_common(top_classes)])

        # Remove labels that are not in the top classes
        for i in range(len(labels) - 1, -1, -1):
            if labels[i] not in top_classes:
                labels = np.delete(labels, i)
                snippets = np.delete(snippets, i)

        # Prepare a dictionary to convert categorical strings to ints
        label_dict = {label: i for i, label in enumerate(sorted(np.unique(labels)))}

    # Load audio data
    audio_data_list = []
    print("Preprocessing audio data")
    for snippet in tqdm(snippets):
        audio_file = dir_to_process / f"{snippet}.wav"
        audio_data = preprocess_audio(audio_file)
        if audio_data.shape[0] == 1:
            audio_data = audio_data.squeeze(0)
        audio_data_list.append(audio_data)

    # Do a rough check on audio data size
    # If audio too long, remove from audio_data_list and snippets and labels
    for i in range(len(audio_data_list) - 1, -1, -1):
        if audio_data_list[i].shape[0] > 100000:
            audio_data_list.pop(i)
            snippets = np.delete(snippets, i)
            labels = np.delete(labels, i)

    labels = [label_dict[l] for l in labels]

    # Train-test split (80-20)
    train_audio, val_audio, train_labels, val_labels = train_test_split(audio_data_list, labels, test_size=0.2, random_state=42, stratify=labels)

    # Create two datasets
    train_dataset = CodaDataset(train_audio, train_labels)
    val_dataset = CodaDataset(val_audio, val_labels)

    # Create dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, train_dataset, val_dataset, labels, label_dict

def main():
    lr = 1e-4
    batch_size = 32
    epochs = 10

    dir_to_process = CODA_DIR
    column_name = args.column_name
    top_classes = args.top_classes
    tune_aves = args.tune_aves
    grad_accum_steps = args.grad_accum_steps
    codec_only = args.codec_only

    train_loader, val_loader, train_dataset, val_dataset, labels, label_dict = prepare_dataloader(
        LabelType.CODA_DETECTOR if args.binary else LabelType.CODA_CLASSIFIER, 
        args.csv_path,
        dir_to_process,
        column_name,
        batch_size=batch_size // grad_accum_steps,
        top_classes=top_classes
    )

    if args.backbone == "aves":
        extractor = AvesTorchaudioWrapper(
            config_path=MODEL_WEIGHTS_PATH / "aves-base-bio.torchaudio.model_config.json",
            model_path=MODEL_WEIGHTS_PATH / "aves-base-bio.torchaudio.pt",
            learnable=tune_aves,
            random_init=args.random_init
        ).to(device)
        model = DownstreamClassifier(
            extractor,
            num_classes=len(np.unique(list(label_dict.values()))),
            embeddings_dim=extractor.get_embedding_dims(),
            hidden=128,
            multi_label=False
        ).to(device)
    elif "vampnet" in args.backbone:
        ckpt_path = get_vampnet_ckpt_path(args.backbone)
        extractor = VampNetWrapper(device, *ckpt_path, codec_only=codec_only).to(device)
        model = DownstreamClassifier(
            extractor,
            num_classes=len(np.unique(list(label_dict.values()))),
            embeddings_dim=extractor.get_embedding_dims(),
            hidden=128,
            multi_label=False
        ).to(device)
    else:
        raise ValueError("Invalid backbone model")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    trainer = Trainer(model, optimizer, device, grad_accum_steps)
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    run_name = f"{column_name}{'_binary' if args.binary else ''}_{'tune' if tune_aves else 'freeze'}_{args.backbone}{'_codec_only' if codec_only else ''}_{'random' if args.random_init else 'pretrained'}_{top_classes if top_classes else 'all'}class_seed{args.seed}"

    # This is the path to store models trained by WhAM (as opposed to the checkpoints of the base models that we train on)
    checkpoint_path = MODEL_WEIGHTS_PATH / f"{run_name}.pt"

    if args.retrain:
        wandb.init(
            # set the wandb project where this run will be logged
            project="WHAM",

            # track hyperparameters and run metadata
            config={
                "learning_rate": lr,
                "architecture": args.backbone,
                "epochs": epochs,
                "top_classes": top_classes,
                "effective_batch_size": batch_size,
                "column_name": column_name,
                "tune_aves": tune_aves,
                "base_ckpt_dir": args.ckpt_dir if args.backbone == "vampnet" else "aves-base-bio.torchaudio.pt",
                "model_path": checkpoint_path,
                "timestamp": timestamp
            },
            name=run_name
        )
         # Train (This saves the best model every epoch)
        trainer.train(train_loader, val_loader, epochs=epochs, model_path=checkpoint_path)
    else:
        trainer.load_model(checkpoint_path)
        loss, acc = trainer.evaluate(val_loader)
        print(f"Loss: {loss}, Accuracy: {acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, help="Backbone model to use")
    parser.add_argument("--random_init", action="store_true", help="Use a random initialization for the backbone model")
    parser.add_argument("--tune_aves", action="store_true", help="Tune AVES model (Vampnet backbones cannot be tuned)")
    parser.add_argument("--binary", action="store_true", help="Use binary labels")
    parser.add_argument("--csv_path", type=str, default=CSV_PATH, help="Path to the CSV file")
    parser.add_argument("--column_name", "-c", type=str, required=True, help="Column name in the CSV file, to be used as labels for training")
    parser.add_argument("--top_classes", type=int, default=None, help="Number of top classes to consider (default is all classes)")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--codec_only", action="store_true", help="Use codec only")
    parser.add_argument("--retrain", action="store_true", help="Rerun the training loop. If this is not picked, the model by default will be loaded from the checkpoint and evaluated only without training.")
    args = parser.parse_args()
    set_seed(args.seed)
    main()