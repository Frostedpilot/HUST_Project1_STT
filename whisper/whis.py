from faster_whisper import WhisperModel
import time
import warnings

warnings.filterwarnings("ignore")

# Load the Whisper model
MODEL_SIZE = 'turbo'
model = WhisperModel(MODEL_SIZE, device='cuda', compute_type='float16')

print("Model loaded successfully\n")
print("Transcribing audio...\n")

while True:
    try:
        # Transcribe the audio
        segments, info = model.transcribe('bruh.mp3', vad_filter=True, language='vi')
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        print("Transcription:\n")
        start = time.time()
        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        end = time.time()
        print("\n")
        print(f"Time taken: {end - start:.2f}s")
        print("\n")
        input("Press Enter to transcribe again\n")
    except Exception as e:
        print(f"Error: {e}")