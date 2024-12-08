from faster_whisper import WhisperModel
import pyaudio
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(filename='transcript.log', encoding='utf-8', level=logging.INFO)


# Load Whisper model
model = WhisperModel('turbo', device='cuda', compute_type='float16')

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper works best with 16kHz audio
CHUNK = int(RATE * 7)  # 5-second chunks

# Initialize PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("Listening... (Press Ctrl+C to stop)")

try:
    while True:
        # Read audio chunk from the microphone
        audio_chunk = stream.read(CHUNK)
        
        # Convert to NumPy array
        audio_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Transcribe audio chunk
        segments, _ = model.transcribe(audio_data, language='en', condition_on_previous_text=False, vad_filter=True, vad_parameters=dict(onset=0.5, min_speech_duration_ms=200, max_speech_duration_s = 4, min_silence_duration_ms=50))
        start = time.time()
        for segment in segments:
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
        end = time.time()
        logging.info(f"Time taken: {end - start:.2f}s")
except KeyboardInterrupt:
    print("Stopping transcription...")
    stream.stop_stream()
    stream.close()
    audio.terminate()
