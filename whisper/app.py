import threading
import queue
import numpy as np
import pyaudio
import wave
from faster_whisper import WhisperModel
import warnings
from pynput import keyboard
import time
import os

warnings.filterwarnings("ignore")

# Shared variables and objects
recording = False
file_queue = queue.Queue()
output_dir = "recordings"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Function to toggle the recording variable
def toggle_recording():
    global recording

    def on_press(key):
        global recording
        try:
            if key.char == 'r':
                recording = not recording
                print(f"Recording {'started' if recording else 'stopped'}")
        except AttributeError:
            pass

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

# Function to record audio
def record_audio():
    global recording

    p = pyaudio.PyAudio()
    samplerate = 16000  # Sample rate for recording
    stream = None
    filename = None
    frames = []
    CHUNK = 1024

    while True:
        if recording and stream is None:
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=samplerate, input=True, frames_per_buffer=CHUNK)
            frames = []
            filename = os.path.join(output_dir, f"recording_{int(time.time())}.wav")
            print("Recording...")

        if recording and stream:
            data = stream.read(CHUNK)
            frames.append(data)

        if not recording and stream:
            print("Saving recording...")
            stream.stop_stream()
            stream.close()
            stream = None

            # Save the recording to a .wav file
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(samplerate)
                wf.writeframes(b''.join(frames))
            print(f"Recording saved to {filename}")
            file_queue.put(filename)


# Function to perform calculations when a .wav file is saved
def calculate_on_save():
    model = WhisperModel("turbo", device="cuda", compute_type="float16")
    print("Press 'R' to start/stop recording. Press Ctrl+C to exit.")
    while True:
        filename = file_queue.get()
        segments, _ = model.transcribe(filename, vad_filter=True)
        print(f"Transcription for {filename}:\n")
        print('*' * 20)
        for segment in segments:
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        print('*' * 20)

# Create and start threads
threads = [
    threading.Thread(target=toggle_recording, daemon=True),
    threading.Thread(target=record_audio, daemon=True),
    threading.Thread(target=calculate_on_save, daemon=True)
]

for thread in threads:
    thread.start()

try:
    while True:
        time.sleep(1)  # Keep the main thread alive
except KeyboardInterrupt:
    print("Exiting program.")

    # Clean the recording directory if user wants to
    clean = input("Do you want to clean the recording directory? (y/n): ")
    if clean.lower() == "y":
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))
        print("Recording directory cleaned.")
    else:
        print("Recording directory not cleaned.")
