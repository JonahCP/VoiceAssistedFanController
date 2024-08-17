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
    
class NormalizeAudio(nn.Module):
    def forward(self, waveform):
        return waveform / waveform.abs().max()
    
class ResampleAudio(nn.Module):
    def __init__(self, target_freq=8000):
        """
        Initialize the ResampleAudio transformation.

        Parameters:
        - target_freq: The new sample rate to resample the audio recordings to.
        """
        super(ResampleAudio, self).__init__()
        self.target_freq = target_freq

    def forward(self, waveform, orig_freq):
        """
        Resample the input waveform to the new sample rate.

        Parameters:
        - waveform: The input audio waveform tensor.
        - orig_freq: The original sample rate of the audio waveform.

        Returns:
        - waveform: The resampled audio waveform tensor.
        """
        waveform = torchaudio.transforms.Resample(orig_freq, self.target_freq)(waveform)
        return waveform

class GrabNoise(nn.Module):
    def __init__(self, noise_dir, min_scale=0.1, max_scale=0.5):
        """
        Initialize the AddNoise transformation.

        Parameters:
        - noise_dir: Directory containing noise files.
        - min_scale: Minimum scaling factor for the noise.
        - max_scale: Maximum scaling factor for the noise.
        """
        super(GrabNoise, self).__init__()
        self.noise_dir = noise_dir
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.noise_files = os.listdir(noise_dir)

    def forward(self):
        """
        Grab a random noise file and scale it.

        Returns:
        - noise: The scaled noise waveform.
        - sample_rate: The sample rate of the noise waveform.
        """

        # Randomly select a noise file
        noise_file = random.choice(self.noise_files)
        noise_path = os.path.join(self.noise_dir, noise_file)

        # Load the noise file
        noise_waveform, sample_rate = torchaudio.load(noise_path)

        # Randomly scale the noise
        scale = random.uniform(self.min_scale, self.max_scale)

        return scale * noise_waveform, sample_rate

class MelSpectrograms(nn.Module):
    def __init__(self, sample_rate=8000, n_mels=13, n_fft=512, win_length=512, hop_length=256, f_min=0):
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
            n_fft=n_fft,
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
    def __init__(self, rate=0.5, freq_mask_param=6, time_mask_param=2):
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

    def __init__(self, file_paths, labels, noise_dir, sample_rate=8000, max_length=10000):
        """
        Initialize the WakeWordDataset.

        Parameters:
        - file_paths: List of file paths for the wake word recordings.
        - noise_dir: Directory containing noise files.
        - sample_rate: The sample rate of the audio recordings.
        - max_length: The length of the audio recordings. Used for padding and trimming.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.add_noise =GrabNoise(noise_dir)
        self.mel_spectrogram = MelSpectrograms(sample_rate)
        self.max_length = max_length
        self.spec_augment = SpecAugment()
        self.normalize = NormalizeAudio()
        self.resample = ResampleAudio()

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load the audio file
        waveform, sample_rate = torchaudio.load(self.file_paths[idx])
        
        # Apply the ResampleAudio transformation
        waveform = self.resample(waveform, sample_rate)

        # Apply the PadTrim transformation
        waveform = PadTrim(self.max_length)(waveform)

        # Apply the NormalizeAudio transformation
        waveform = self.normalize(waveform)
        
        # Apply the GrabNoise transformation
        noise, noise_sr = self.add_noise()

        # Apply the ResampleAudio transformation
        noise = self.resample(noise, noise_sr)

        # Apply the PadTrim transformation
        noise = PadTrim(self.max_length)(noise)

        # Apply normalization
        noise = self.normalize(noise)
        
        # Add the noise to the waveform
        waveform = waveform + noise

        # Apply the MelSpectrograms transformation
        spectrogram = self.mel_spectrogram(waveform)
        
        # Apply the SpecAugment transformation
        spectrogram = self.spec_augment(spectrogram)

        # Get the label
        label = self.labels[idx]
        
        return (spectrogram, label)