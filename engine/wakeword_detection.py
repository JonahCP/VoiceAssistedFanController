import numpy as np
import pyaudio
import torch
from collections import deque
from wakeword.model.augments import MelSpectrograms
from wakeword.model.model import CNNGRU

class AudioListener:
    def __init__(self, sample_rate=8000, num_seconds=1.25, chunk_size=4000):
        """
        Initialize audio listener.

        Parameters:
        - sample_rate: The sample rate of the audio.
        - num_seconds: The number of seconds to record.
        """
        self.sample_rate = sample_rate
        self.num_seconds = num_seconds
        self.chunk_size = chunk_size
        self.buffer_size = int(sample_rate * num_seconds)
        self.buffer = deque(maxlen=self.buffer_size)
        self.p = pyaudio.PyAudio()
        self.callback = None

    def start(self):
        """
        Start recording audio.
        """
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.sample_rate,
                                  input=True,
                                  input_device_index=None,
                                  frames_per_buffer=self.chunk_size,
                                  stream_callback=self.stream_callback)
        self.stream.start_stream()

    def stream_callback(self, in_data, frame_count, time_info, status):
        """
        Stream callback function.
        """

        # Convert the input data to a numpy array
        data = np.frombuffer(in_data, dtype=np.int16)

        self.buffer.extend(data)

        # Call the callback function
        if self.callback and (len(self.buffer) == self.buffer_size):
            # Grab the FULL buffer
            full_buffer = np.array(self.buffer)
            self.callback(full_buffer)

        return (None, pyaudio.paContinue)
    
    def set_callback(self, callback):
        """
        Set the callback function.
        """
        print('Setting callback')
        self.callback = callback

    def stop(self):
        """
        Stop recording audio.
        """
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

class WakeWordDetector:
    def __init__(self, model_class, model_path, sample_rate=8000, threshold=0.6):
        """
        Initialize the wake word detector.

        Parameters:
        - threshold: The threshold for detecting the wake word.
        """
        self.model = model_class  # Initialize the model
        state_dict = torch.load(model_path, weights_only=True, map_location=torch.device('cpu'))  # Load the state dictionary
        self.model.load_state_dict(state_dict)  # Load the state dict into the model
        self.model.eval()
        self.mel_spectrogram_transform = MelSpectrograms(sample_rate=sample_rate)
        self.threshold = threshold
    
    def detect_wake_word(self, audio_data):
        """
        Detect the wake word in the audio data.

        Parameters:
        - audio_data: The audio data to analyze.

        Returns:
        - True if the wake word is detected, False otherwise.
        """
        audio_data = torch.tensor(audio_data, dtype=torch.float32)
        print(audio_data.shape)
        
        audio_data = audio_data.unsqueeze(0)  

        with torch.no_grad(): 
            mel_spectrogram = self.mel_spectrogram_transform(audio_data)
            print(mel_spectrogram.shape)
            mel_spectrogram = mel_spectrogram.unsqueeze(0)

            output = self.model(mel_spectrogram)
            probabilities = torch.softmax(output, dim=1)
            print('Probabilities:', probabilities)

        return probabilities[0][1] > self.threshold

class TranscriptionAndCommands:
    def __init__(self, model):
        self.model = model

    def transcribe(self, audio_date):
        segments, _ = self.model.transcribe(audio_date, beam_size=5, language='en')

        text = ''
        for segment in segments:
            text = segment.text
        
        return text

# Example usage
def main():

    listener = AudioListener()
    detector = WakeWordDetector(CNNGRU(num_classes=2), 'checkpoints/model.pth')

    def callback(data):
        if detector.detect_wake_word(data):
            print('Wake word detected!')

    listener.set_callback(callback)
    listener.start()

    while True:
        pass


if __name__ == "__main__":
    main()