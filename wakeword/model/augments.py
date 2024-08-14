import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
import random

class PadTrim(nn.Module):
    def __init__(self, max_length = 10000):
        """
        Initialize the PadTrim transformation.

        Parameters:
        - max_length: The maximum length of the audio recordings.
        """
        super(PadTrim, self).__init__()
        self.max_length = max_length

    def forward(self, waveform):
        """
        Pad or trim the input waveform to the maximum length.

        Parameters:
        - waveform: The input audio waveform tensor.

        Returns:
        - waveform: The padded or trimmed audio waveform tensor.
        """
        if waveform.size(1) < self.max_length:
            waveform = nn.functional.pad(waveform, (0, self.max_length - waveform.size(1)))
        elif waveform.size(1) > self.max_length:
            waveform = waveform[:, :self.max_length]
        return waveform

class AddNoise(nn.Module):
    def __init__(self, noise_dir, min_scale=0.1, max_scale=0.5):
        """
        Initialize the AddNoise transformation.

        Parameters:
        - noise_dir: Directory containing noise files.
        - min_scale: Minimum scaling factor for the noise.
        - max_scale: Maximum scaling factor for the noise.
        """
        super(AddNoise, self).__init__()
        self.noise_dir = noise_dir
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.noise_files = os.listdir(noise_dir)

    def forward(self, waveform):
        """
        Add random noise to the waveform at a random scale.

        Parameters:
        - waveform: The input audio waveform tensor.

        Returns:
        - waveform: The audio waveform with added noise.
        """
        # Randomly select a noise file
        noise_file = random.choice(self.noise_files)
        noise_path = os.path.join(self.noise_dir, noise_file)

        # Load the noise file
        noise_waveform, _ = torchaudio.load(noise_path)

        # Randomly scale the noise
        scale = random.uniform(self.min_scale, self.max_scale)

        # Add the noise to the waveform
        waveform = waveform + scale * noise_waveform

        return waveform

class MelSpectrograms(nn.Module):
    def __init__(self, sample_rate=8000, n_mels=13, win_length=512, hop_length=256, f_min=0):
        """
        Initialize the MelSpectrograms transformation.

        Parameters:
        - sample_rate: The sample rate of the input audio.
        - n_mels: The number of mel bins.
        - win_length: The length of the STFT window.
        - hop_length: The hop length of the STFT window.
        - f_min: The minimum frequency of the mel bins.
        - f_max: The maximum frequency of the mel bins.
        """
        super(MelSpectrograms, self).__init__()
        self.transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=sample_rate//2
        )

    def forward(self, waveform):
        """
        Compute the mel spectrogram of the input waveform.

        Parameters:
        - waveform: The input audio waveform tensor.

        Returns:
        - spectrogram: The computed mel spectrogram.
        """
        spectrogram = self.transform(waveform)

        # Min-max scale the spectrogram
        spectrogram_min = spectrogram.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        spectrogram_max = spectrogram.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        spectrogram = (spectrogram - spectrogram_min) / (spectrogram_max - spectrogram_min + 1e-6)

        return spectrogram
    
class SpecAugment(nn.Module):
    def _init__(self, rate=0.5, freq_mask_param=6, time_mask_param=2):
        """
        Initialize the SpecAugment transformation.

        Parameters:
        - rate: The probability of applying the transformation.
        - freq_mask_param: The maximum frequency mask length.
        - time_mask_param: The maximum time mask length.
        """
        super(SpecAugment, self).__init__()
        self.rate = rate
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)

    def forward(self, spectrogram):
        """
        Apply the SpecAugment transformation to the input spectrogram.

        Parameters:
        - spectrogram: The input mel spectrogram tensor.

        Returns:
        - spectrogram: The augmented mel spectrogram.
        """
        if random.random() < self.rate:
            spectrogram = self.freq_mask(spectrogram)
            spectrogram = self.time_mask(spectrogram)
        return spectrogram
        
class WakeWordDataset(Dataset):

    def __init__(self, file_paths, noise_dir, sample_rate=8000, max_length=10000):
        """
        Initialize the WakeWordDataset.

        Parameters:
        - file_paths: List of file paths for the wake word recordings.
        - noise_dir: Directory containing noise files.
        - sample_rate: The sample rate of the audio recordings.
        - max_length: The length of the audio recordings. Used for padding and trimming.
        """
        self.file_paths = file_paths
        self.add_noise = AddNoise(noise_dir)
        self.mel_spectrogram = MelSpectrograms(sample_rate)
        self.max_length = max_length
        self.spec_augment = SpecAugment()

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load the audio file
        waveform, _ = torchaudio.load(self.file_paths[idx])
        
        # Apply the PadTrim transformation
        waveform = PadTrim(self.max_length)(waveform)
        
        # Apply the AddNoise transformation
        waveform = self.add_noise(waveform)
        
        # Apply the MelSpectrograms transformation
        spectrogram = self.mel_spectrogram(waveform)
        
        # Apply the SpecAugment transformation
        spectrogram = self.spec_augment(spectrogram)
        
        return spectrogram