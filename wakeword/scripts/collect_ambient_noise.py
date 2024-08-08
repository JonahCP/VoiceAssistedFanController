import wave
import sys
import pyaudio
import os

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1 
RATE = 44100
RECORD_SECONDS = 300

output_dir = 'my_data'
output_file = os.path.join(output_dir, 'ambient_noise.wav')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with wave.open(output_file, 'wb') as wf:
    p = pyaudio.PyAudio()
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)

    stream = p.open(format=FORMAT, channels=1, rate=RATE, input=True, input_device_index=1)

    print('Recording...')
    for _ in range(0, RATE // CHUNK * RECORD_SECONDS):
        wf.writeframes(stream.read(CHUNK))
    print('Done')

    stream.close()
    p.terminate()