# !pip install transformers, sentencepiece

from transformers import SpeechEncoderDecoderModel
from transformers import AutoFeatureExtractor, AutoTokenizer, GenerationConfig
import torchaudio
import torch
from pydub import AudioSegment
import glob
import os

model_path = 'nguyenvulebinh/wav2vec2-bartpho'
model = SpeechEncoderDecoderModel.from_pretrained(model_path).eval()
feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
if torch.cuda.is_available():
  model = model.cuda()


def decode_tokens(token_ids, skip_special_tokens=True, time_precision=0.02):
    timestamp_begin = tokenizer.vocab_size
    outputs = [[]]
    for token in token_ids:
        if token >= timestamp_begin:
            timestamp = f" |{(token - timestamp_begin) * time_precision:.2f}| "
            outputs.append(timestamp)
            outputs.append([])
        else:
            outputs[-1].append(token)
    outputs = [
        s if isinstance(s, str) else tokenizer.decode(s, skip_special_tokens=skip_special_tokens) for s in outputs
    ]
    return "".join(outputs).replace("< |", "<|").replace("| >", "|>")

def decode_wav(audio, asr_model, prefix=""):
    device = next(asr_model.parameters()).device

    # Ensure audio is a list of numpy arrays (expected by feature_extractor)
    if isinstance(audio, torch.Tensor):
        audio = audio.numpy()
    if audio.ndim == 1:
        audio = [audio]  # Wrap in a list if not already a batch

    input_values = feature_extractor.pad(
        [{"input_values": feature} for feature in audio],
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors="pt",
    )

    # Generate outputs using the ASR model
    output_beam_ids = asr_model.generate(
        input_values['input_values'].to(device),
        attention_mask=input_values['attention_mask'].to(device),
        decoder_input_ids=tokenizer.batch_encode_plus([prefix] * len(audio), return_tensors="pt")['input_ids'][..., :-1].to(device),
        generation_config=GenerationConfig(decoder_start_token_id=tokenizer.bos_token_id),
        max_length=250,
        num_beams=25,
        no_repeat_ngram_size=4,
        num_return_sequences=1,
        early_stopping=True,
        return_dict_in_generate=True,
        output_scores=True,
    )

    # Decode the output tokens into text
    output_text = [decode_tokens(sequence) for sequence in output_beam_ids.sequences]

    return output_text

# A function to split the wav into chunks of 20 seconds and then decode each chunk

def chunk_and_decode_wav(input_file, output_dir, chunk_duration=20):
    audio = AudioSegment.from_file(input_file)
    
    # Convert to mono and set sample rate to 16kHz
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)  # 2 bytes = 16-bit
    
    # Get the audio duration in milliseconds
    audio_length = len(audio)
    chunk_length = chunk_duration * 1000  # Convert seconds to milliseconds
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Split and save chunks
    for i, start_time in enumerate(range(0, audio_length, chunk_length)):
        end_time = min(start_time + chunk_length, audio_length)
        chunk = audio[start_time:end_time]
        chunk.export(f"{output_dir}/chunk_{i + 1}.wav", format="wav")
        print(f"Exported chunk {i + 1}: {start_time / 1000}s - {end_time / 1000}s")
    
    print("Audio splitting complete.")


# https://huggingface.co/nguyenvulebinh/wav2vec2-bartpho/resolve/main/sample_news.wav
chunk_and_decode_wav("bruh.mp3", "chunks")
chunk_files = sorted(glob.glob("chunks/*.wav"))
for chunk_file in chunk_files:
    # Load the chunk with torchaudio
    audio, sample_rate = torchaudio.load(chunk_file)

    # Resample to 16kHz if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        audio = resampler(audio)

    # Ensure audio is mono
    if audio.shape[0] > 1:  # If multiple channels
        audio = torch.mean(audio, dim=0, keepdim=True)

    audio = audio.squeeze()  # Remove channel dimension
    transcription = decode_wav(audio, model)
    print(f"Transcription for {chunk_file}: {transcription}")

# Clean up the chunks directory
for chunk_file in chunk_files:
    os.remove(chunk_file)