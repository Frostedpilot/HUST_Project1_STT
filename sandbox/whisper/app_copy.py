import threading
import queue
import numpy as np
import pyaudio
import wave
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, WhisperTokenizer, pipeline
import warnings
from pynput import keyboard
from pydub import AudioSegment
import time
import os

warnings.filterwarnings("ignore")


# Shared variables and objects
recording = False
file_queue = queue.Queue()
output_dir = "recordings"
# Load the Whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16
model_id = "openai/whisper-base"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)
tokenizer = WhisperTokenizer.from_pretrained(model_id)

pipe = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=tokenizer,
    feature_extractor=processor.feature_extractor,
    # torch_dtype=torch_dtype,
    device=device,
    # chunk_length_s=10,
)


# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def preprocess_audio(filename):
    audio = AudioSegment.from_file(filename)
    audio = audio.set_channels(1).set_frame_rate(16000)
    processed_filename = f"{filename}_processed.wav"
    audio.export(processed_filename, format="wav")
    return processed_filename

# Function to toggle the recording variable
recording_lock = threading.Lock()
def toggle_recording():
    global recording

    def on_press(key):
        global recording
        try:
            if key.char == 'r':
                with recording_lock:
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
    print("Press 'R' to start/stop recording. Press Ctrl+C to exit.")
    while True:
        filename = file_queue.get()
        try:
            processed_filename = preprocess_audio(filename)
            print(f"Transcribing {processed_filename}...")
            result = pipe(processed_filename)
            print(f"Transcription for {processed_filename}:\n")
            print('*' * 20)
            print(result['text'])
            print('*' * 20)
        except Exception as e:
            print(f"Error transcribing {filename}: {e}")

# Create and start threads
threads = [
    threading.Thread(target=toggle_recording, daemon=True),
    threading.Thread(target=record_audio, daemon=True),
]

for thread in threads:
    thread.start()

try:
    calculate_on_save()
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
