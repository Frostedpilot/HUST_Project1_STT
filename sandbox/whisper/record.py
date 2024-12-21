import pyaudio
import wave

def record_audio(filename, duration, sample_rate=44100, chunk_size=1024, channels=2):
    audio = pyaudio.PyAudio()
    
    # Open a stream for recording
    stream = audio.open(format=pyaudio.paInt16, 
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)
    
    print("Recording...")
    frames = []

    for _ in range(0, int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)

    print("Recording finished.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recording as a .wav file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

# Record a 5-second audio file
record_audio("output.wav", duration=5)