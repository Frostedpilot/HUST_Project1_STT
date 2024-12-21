from silero_vad import (
    load_silero_vad,
    read_audio,
    get_speech_timestamps,
    save_audio,
    collect_chunks,
)

model = load_silero_vad()
wav = read_audio("bruh.wav")
speech_timestamp = get_speech_timestamps(wav, model)

# Get how many seconds of non-speech silero vad has detected
silero_vad_speech = 0
for i in range(len(speech_timestamp)):
    silero_vad_speech += speech_timestamp[i]["end"] - speech_timestamp[i]["start"]

silero_vad_speech /= 16000  # sample rate

second = silero_vad_speech % 60
_minute = silero_vad_speech // 60
minute = _minute % 60
hour = _minute // 60

print(
    f"Silero VAD detected {int(hour)} hours {int(minute)} minutes {int(second)} seconds of speech"
)

save_audio("speech.wav", collect_chunks(speech_timestamp, wav))
