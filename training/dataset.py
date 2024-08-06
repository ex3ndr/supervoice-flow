import torch
import torchaudio
from supervoice_flow.config import config
from supervoice_flow.audio import load_mono_audio, spectogram
from .audio import do_reverbrate
from pathlib import Path
import random
import requests

def create_sampler(datasets, duration, return_source = False):

    # Target duration
    samples = int(duration * config.audio.sample_rate)

    # Load the datasets
    files = []
    if isinstance(datasets, str):
        if datasets.startswith("https:") or datasets.startswith("http:"):
            dataset_files = requests.get(datasets + "files_all.txt").text.splitlines()
        else:
            with open(datasets + "files_all.txt", 'r') as file:
                dataset_files = file.read().splitlines()
        dataset_files = [datasets + p + ".flac" for p in dataset_files]
    else:
        dataset_files = []
        for dataset in datasets:
            dataset_files += list(Path(dataset).rglob("*.wav")) + list(Path(dataset).rglob("*.flac"))
        dataset_files = [str(p) for p in dataset_files]
    files += dataset_files
    print(f"Loaded {len(files)} files")

    # Sample a single item
    def sample_item():
        # Load random audio
        f = random.choice(files)
        audio = load_mono_audio(f, config.audio.sample_rate)

        # Pad or trim audio
        if audio.shape[0] < samples:
            padding = samples - audio.shape[0]
            padding_left = random.randint(0, padding)
            padding_right = padding - padding_left
            audio = torch.nn.functional.pad(audio, (padding_left, padding_right), value=0)
        else:
            start = random.randint(0, audio.shape[0] - samples)
            audio = audio[start:start + samples]

        # Spectogram
        spec = spectogram(audio, 
            n_fft = config.audio.n_fft, 
            n_mels = config.audio.n_mels, 
            n_hop = config.audio.hop_size, 
            n_window = config.audio.win_size,  
            mel_norm = config.audio.mel_norm, 
            mel_scale = config.audio.mel_scale, 
            sample_rate = config.audio.sample_rate
        ).transpose(0, 1).to(torch.float16)

        # Return result
        if return_source:
            return spec, audio
        else:
            return spec

    def sample_item_retry():
        while True:
            try:
                return sample_item()
            except Exception as e:
                print(f"Error: {e}")

    return sample_item_retry

def create_loader(datasets, duration, batch_size, num_workers, return_source = False):

    # Load sampler
    sampler = create_sampler(datasets, duration, return_source)

    # Load dataset
    class DistortedDataset(torch.utils.data.IterableDataset):
        def __init__(self, sampler):
            self.sampler = sampler
        def generate(self):
            while True:
                yield self.sampler()
        def __iter__(self):
            return iter(self.generate())
    dataset = DistortedDataset(sampler)

    # Load loader
    loader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, num_workers = num_workers, pin_memory = True, shuffle=False)

    return loader