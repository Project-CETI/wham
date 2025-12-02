# This file contains the model definitions for the audio embedding models used by WhAM.
from torchaudio.models import wav2vec2_model
import torch.nn as nn
import torch
import json
from tqdm import tqdm
import pickle
import os
from preprocessing import preprocess_audio
from audiotools import AudioSignal
from vampnet.interface import Interface
from wam import VAMPNET_DIR
import wandb
from pathlib import Path

class AvesTorchaudioWrapper(nn.Module):
    '''
    This class is the wrapper around the AVES model and provides a forward method to generate embeddings given any audio using the AVES architecture.
    '''

    def __init__(self, config_path, model_path, learnable=False, random_init=True):

        super().__init__()

        # reference: https://pytorch.org/audio/stable/_modules/torchaudio/models/wav2vec2/utils/import_fairseq.html

        self.config = self.load_config(config_path)
        self.model = wav2vec2_model(**self.config, aux_num_out=None)
        if not random_init:
            self.model.load_state_dict(torch.load(model_path))
        self.model.feature_extractor.requires_grad_(learnable)

    def load_config(self, config_path):
        with open(config_path, 'r') as ff:
            obj = json.load(ff)

        return obj

    def forward(self, sig):
        # extract_feature in the sorchaudio version will output all 12 layers' output, -1 to select the final one
        out = self.model.extract_features(sig)[0][-1]

        return out
    
    def get_embedding_dims(self):
        return 768
    
class VampNetWrapper(nn.Module):
    '''
    This class is the wrapper around the VampNet model and provides a forward method to generate embeddings given any audio using the VampNet architecture.
    '''

    def __init__(self, device, coarse, c2f, coarse_lora, c2f_lora, codec_only=False):
        super().__init__()
        self.device = device
        self.interface = Interface(
            codec_ckpt=VAMPNET_DIR / "models" / "codec.pth",
            coarse_ckpt=coarse,
            coarse2fine_ckpt=c2f,
            coarse_lora_ckpt=coarse_lora,
            coarse2fine_lora_ckpt=c2f_lora,
            device=device
        )
        self.codec_only = codec_only
    
    def forward(self, x):
        # Remember that the audio is 16khz sample rate
        signals = [AudioSignal(xx, sample_rate=16000, device=self.device) for xx in x]
        audio_signal = AudioSignal.batch(signals)
        if self.codec_only:
            with torch.inference_mode():
                signal = self.interface.preprocess(audio_signal).to(self.device)
                embedding = self.interface.codec.encode(signal.samples, signal.sample_rate)["z"]
                embedding = embedding.permute(0, 2, 1)
        else:
            embedding = self.vampnet_embed(audio_signal)
        return embedding

    # per JukeMIR, we want the emebddings from the middle layer?
    def vampnet_embed(self, sig: AudioSignal, layer=10):
        with torch.inference_mode():
            # preprocess the signal
            sig = self.interface.preprocess(sig)

            # get the coarse vampnet model
            vampnet = self.interface.coarse

            # get the tokens
            z =  self.interface.encode(sig)[:, : vampnet.n_codebooks, :]
            z_latents = vampnet.embedding.from_codes(z,  self.interface.codec)

            # do a forward pass through the model, get the embeddings
            _z, embeddings = vampnet(z_latents, return_activations=True)

            num_layers = embeddings.shape[0]
            assert (
                layer < num_layers
            ), f"layer {layer} is out of bounds for model with {num_layers} layers"

            # return the embeddings
            return embeddings[layer]
        
    def get_embedding_dims(self):
        return 1024 if self.codec_only else 1280


class DownstreamClassifier(nn.Module):
    '''
    This is a simple 2 layer MLP classifier that takes the embeddings from any backbone model and classifies them into the desired number of classes.
    '''
    def __init__(self, model, num_classes, embeddings_dim=768, hidden=128, multi_label=False):

        super().__init__()

        self.model = model
        self.head = nn.Sequential(
            nn.Linear(in_features=embeddings_dim, out_features=hidden),
            nn.ReLU(),
            nn.Linear(in_features=hidden, out_features=num_classes)
        )
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        out = self.model(x)
        out = out.mean(dim=1) # Note: it's essential that the embeddings of the backbone are of shape [bsize, seq_len, emb_dim]
        logits = self.head(out)
        loss = None
        if y is not None:
            loss = self.loss_func(logits, y)

        return loss, logits

class CodaDataset(torch.utils.data.Dataset):
    '''
    This class is a simple PyTorch dataset that takes in a list of audio data and their corresponding labels and returns them in the __getitem__ method.
    '''
    def __init__(self, audio_data_list, labels):
        super().__init__()
        self.audio_data_list = audio_data_list
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio_data = self.audio_data_list[idx]
        label = self.labels[idx]
        return audio_data, label

    
class Trainer():
    '''
    This class is a simple PyTorch trainer that takes in a model, optimizer, device, and gradient accumulation steps and trains the model on the given data.
    '''
    def __init__(self, model, optimizer, device, grad_accum_steps=1):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.grad_accum_steps = grad_accum_steps

    def train(self, train_loader, val_loader, epochs, model_path):
        # start a new wandb run to track this script

        grad_accum_counter = 0

        # Evaluate once at the beginning
        loss, accuracy = self.evaluate(val_loader)
        wandb.log({"val loss": loss, "val accuracy": accuracy})

        best_accuracy = 0
        
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            self.model.train()
            for i, (x, y) in enumerate(tqdm(train_loader)):
                x, y = x.to(self.device), y.to(self.device)
                
                grad_accum_counter += 1
                if grad_accum_counter % self.grad_accum_steps == 0:
                    self.optimizer.zero_grad()

                loss, _ = self.model(x, y)
                loss.backward()

                if grad_accum_counter % self.grad_accum_steps == 0:
                    self.optimizer.step()
                wandb.log({"train loss": loss.item()})

            # Evaluate accuracy after each epoch
            loss, accuracy = self.evaluate(val_loader)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save_model(model_path)
            wandb.log({"val loss": loss, "val accuracy": accuracy})

        wandb.log({"best val accuracy": best_accuracy})
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                loss, logits = self.model(x, y)
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

        return total_loss / len(dataloader), correct / total
    
    def load_model(self, path):
        if isinstance(self.model.model, VampNetWrapper):
            state_dict = torch.load(path)
            new_state_dict = {}
            new_state_dict['head.0.weight'] = state_dict['head.0.weight']
            new_state_dict['head.0.bias'] = state_dict['head.0.bias']
            new_state_dict['head.2.weight'] = state_dict['head.2.weight']
            new_state_dict['head.2.bias'] = state_dict['head.2.bias']
            self.model.load_state_dict(new_state_dict, strict=False)
        else:
            self.model.load_state_dict(torch.load(path))

    def save_model(self, path):
        if isinstance(self.model.model, VampNetWrapper):
            state_dict = self.model.state_dict()
            new_state_dict = {}
            new_state_dict['head.0.weight'] = state_dict['head.0.weight']
            new_state_dict['head.0.bias'] = state_dict['head.0.bias']
            new_state_dict['head.2.weight'] = state_dict['head.2.weight']
            new_state_dict['head.2.bias'] = state_dict['head.2.bias']
            torch.save(new_state_dict, path)
        else:
            torch.save(self.model.state_dict(), path)
