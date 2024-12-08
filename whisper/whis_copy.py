import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import time
import warnings

warnings.filterwarnings("ignore")

# Load the Whisper model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-medium.en"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

generate_kwargs = {
    "condition_on_prev_tokens": True,
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "return_timestamps": True,
}

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
    generate_kwargs=generate_kwargs,
    chunk_length_s=30,
)

print("Model loaded successfully\n")
print("Transcribing audio...\n")

while True:
    try:
        # Transcribe the audio
        start = time.time()
        result = pipe("bruh.mp3")
        end = time.time()
        print("Transcription:\n")
        print(result['text'])
        print("\n")
        print(f"Time taken: {end - start:.2f}s")
        print("\n")
        input("Press Enter to transcribe again\n")
    except Exception as e:
        print(f"Error: {e}")