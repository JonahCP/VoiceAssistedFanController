import wave
import sys
import pyaudio
import os
import keyboard

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1 
RATE = 44100
RECORD_SECONDS = 1.25 

output_dir = 'my_data'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

counter = 0

for i in range(100):        
    # Press space to record a new wakeup word                                                                                      
    keyboard.wait('space')
    output_file = os.path.join(output_dir, f'wakeup_word_{counter}.wav')

    with wave.open(output_file, 'wb') as wf:
        p = pyaudio.PyAudio()
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)

        stream = p.open(format=FORMAT, channels=1, rate=RATE, input=True, input_device_index=1)

        print('Recording...')
        for _ in range(0, int(RATE // CHUNK * RECORD_SECONDS)):
            wf.writeframes(stream.read(CHUNK))
        print('Done')

        stream.close()
        p.terminate()
        counter += 1