import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import time
import warnings
import logging
import pyaudio
import numpy as np

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logging.basicConfig(filename='transcript.log', encoding='utf-8', level=logging.INFO)

# Load Whisper model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

generate_kwargs = {
    "condition_on_prev_tokens": False,
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.2,
    "return_timestamps": False,
}

LENGTH = 10

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    generate_kwargs=generate_kwargs,
    chunk_length_s=LENGTH,
)


# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper works best with 16kHz audio
CHUNK = int(RATE * LENGTH)  # LENGTH-second chunks

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
        start = time.time()
        result = pipe(audio_data)
        end = time.time()
        print(result['text'])
        logging.info("Transcription:\n")
        logging.info(result['text'])
        logging.info(f"Time taken: {end - start:.2f}s")
        logging.info("\n")
except KeyboardInterrupt:
    print("Stopping transcription...")
    stream.stop_stream()
    stream.close()
    audio.terminate()
