import numpy as np
import pyaudio
import torch
from collections import deque
from wakeword.model.augments import MelSpectrograms
from wakeword.model.model import CNNGRU
from faster_whisper import WhisperModel
import time

class AudioListener:
    def __init__(self, wakeword_sample_rate=8000, transcription_sample_rate=16000, num_seconds=1.25, chunk_size=4000):
        self.wakeword_sample_rate = wakeword_sample_rate
        self.transcription_sample_rate = transcription_sample_rate
        self.num_seconds = num_seconds
        self.chunk_size = chunk_size
        self.buffer_size = int(wakeword_sample_rate * num_seconds)
        self.wakeword_buffer = deque(maxlen=self.buffer_size)
        self.p = pyaudio.PyAudio()
        self.callback = None
        self.mode = 'wake_word'
        self.transcription_duration = 3
        self.transcription_buffer = deque(maxlen=int(transcription_sample_rate * self.transcription_duration))
        self.stream = None
        self.should_stop_stream = False  # Flag to stop stream

    def start_wakeword_stream(self):
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.wakeword_sample_rate,
                                  input=True,
                                  frames_per_buffer=self.chunk_size,
                                  stream_callback=self.stream_callback)
        self.stream.start_stream()

    def start_transcription_stream(self):
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.transcription_sample_rate,
                                  input=True,
                                  frames_per_buffer=self.chunk_size,
                                  stream_callback=self.stream_callback)
        self.stream.start_stream()

    def stop_stream(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def stream_callback(self, in_data, frame_count, time_info, status):
        if self.mode == 'wake_word':
            data = np.frombuffer(in_data, dtype=np.int16)
            self.wakeword_buffer.extend(data)

            if self.callback and (len(self.wakeword_buffer) == self.buffer_size):
                full_wake_buffer = np.array(self.wakeword_buffer)
                if self.callback(full_wake_buffer):  # If wakeword detected, set flag to stop stream
                    self.should_stop_stream = True
                    self.wakeword_buffer.clear()
                    self.transcription_buffer.clear()


        elif self.mode == 'transcription':
            data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32)
            data = data / np.iinfo(np.int16).max  # Normalize for Whisper
            self.transcription_buffer.extend(data)

            if len(self.transcription_buffer) == self.transcription_buffer.maxlen:
                full_transcription_buffer = np.array(self.transcription_buffer)
                self.callback(full_transcription_buffer) # type: ignore
                self.should_stop_stream = True
                self.wakeword_buffer.clear()
                self.transcription_buffer.clear()

        return (None, pyaudio.paContinue)

    def set_callback(self, callback):
        self.callback = callback

    def stop(self):
        self.stop_stream()
        self.p.terminate()

class WakeWordDetector:
    def __init__(self, model_class, model_path, sample_rate=8000, threshold=0.999):
        self.model = model_class
        state_dict = torch.load(model_path, weights_only=True, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.mel_spectrogram_transform = MelSpectrograms(sample_rate=sample_rate)
        self.threshold = threshold
    
    def detect_wake_word(self, audio_data):
        audio_data = torch.tensor(audio_data, dtype=torch.float32)
        audio_data = audio_data.unsqueeze(0)

        with torch.no_grad(): 
            mel_spectrogram = self.mel_spectrogram_transform(audio_data)
            mel_spectrogram = mel_spectrogram.unsqueeze(0)

            print('Mel spectrogram:', mel_spectrogram)

            output = self.model(mel_spectrogram)
            probabilities = torch.softmax(output, dim=1)
            print('Probabilities:', probabilities)

        return probabilities[0][1] > self.threshold

class TranscriptionAndCommands:
    def __init__(self, model):
        self.model = model

    def transcribe(self, audio_data):
        segments, _ = self.model.transcribe(audio_data, beam_size=5, language='en')
        text = ''
        for segment in segments:
            text += segment.text
        return text

    def execute_command(self, command):
        command = command.lower()
        if 'turn' in command and 'on' in command and 'fan' in command:
            print('Turning on fan')
        elif 'turn' in command and 'off' in command and 'fan' in command:
            print('Turning off fan')
        elif 'speed' in command and 'up' in command and 'fan' in command:
            print('Speeding up fan')
        elif 'slow' in command and 'down' in command and 'fan' in command:
            print('Slowing down fan')
        else:
            print('Command not recognized')

def main():
    listener = AudioListener()
    detector = WakeWordDetector(CNNGRU(num_classes=2), 'checkpoints/model.pth')
    transcriber = TranscriptionAndCommands(WhisperModel('small', device='cpu', compute_type='float32'))

    def callback(data):
        if listener.mode == 'wake_word':
            if detector.detect_wake_word(data):
                print('Wake word detected')
                listener.mode = 'transcription'  
                return True
            return False
        elif listener.mode == 'transcription':
            print('Transcribing...')
            print(data.shape)
            transcription = transcriber.transcribe(data)
            print('Transcription:', transcription)
            transcriber.execute_command(transcription)
            listener.mode = 'wake_word'  

    listener.set_callback(callback)
    listener.start_wakeword_stream()

    while True:
        time.sleep(0.1)
        if listener.should_stop_stream:
            listener.stop_stream()
            listener.should_stop_stream = False  # Reset the flag
            if listener.mode == 'transcription':
                listener.start_transcription_stream()
            elif listener.mode == 'wake_word':
                listener.start_wakeword_stream()

if __name__ == "__main__":
    main()
