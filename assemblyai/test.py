import assemblyai as aai
import json

with open("../secret.json", "r") as file:
    secret = json.load(file)
    api_key = secret["ASSEMBLYAI_API_KEY"]

aai.settings.api_key = api_key
config = aai.TranscriptionConfig(language_code="vi")
transcriber = aai.Transcriber(config=config)

audio_file = "bruh.wav"

transcript = transcriber.transcribe(audio_file)

if transcript.status == aai.TranscriptStatus.error:
    print("Transcription failed: ", transcript.error)
else:
    print(transcript.text)