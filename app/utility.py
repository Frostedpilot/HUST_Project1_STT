import requests
import samplerate
import threading
import warnings
import torch
import os
import glob
import re
import json
import httpx
import torchaudio
import soundfile as sf
import assemblyai as aai
from transformers import (
    SpeechEncoderDecoderModel,
    AutoFeatureExtractor,
    AutoTokenizer,
    GenerationConfig,
)
from silero_vad import (
    load_silero_vad,
    read_audio,
    get_speech_timestamps,
    save_audio,
    collect_chunks,
)
from transformers import Wav2Vec2Model
from faster_whisper import WhisperModel
from PyQt6.QtCore import QRunnable, pyqtSignal, QObject, QSettings
from deepgram import FileSource, PrerecordedOptions
from yt_dlp import YoutubeDL

warnings.filterwarnings("ignore")

settings = QSettings("Frostedpilot", "STT_app")
BASE_DIR = settings.value("BASE_DIR")
print("utility BASE DIR:", BASE_DIR)


class APIError(Exception):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.message = message
        self.code = code


def update_cuda_device():
    if torch.cuda.is_available():
        settings.setValue("device", "cuda")
    else:
        settings.setValue("device", "cpu")


def update_utility_base_dir(new_base_dir):
    global BASE_DIR
    BASE_DIR = new_base_dir
    print("Updated utility BASE_DIR:", BASE_DIR)


def preprocess_audio(audio_path):
    os.makedirs(os.path.join(BASE_DIR, "res"), exist_ok=True)
    local_file = os.path.join(BASE_DIR, "res/audio.wav")
    data, sr = sf.read(audio_path)
    # Convert to mono (if needed) and resample to 16kHz
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:

        ratio = 16000 / sr
        data = samplerate.resample(data, ratio, "sinc_best")
        sr = 16000

    sf.write(local_file, data, sr, format="wav")


def check_deepgram_api_key(api_key):
    url = "https://api.deepgram.com/v1/auth/token"
    params = {"Authorization": f"Token {api_key}"}
    try:
        response = requests.get(url, headers=params)
    except Exception:
        raise APIError("Bad request", 400)
    if response.status_code == 200:
        print("Success!")
        return True
    elif response.status_code == 400:
        raise APIError("Bad request", response.status_code)
    elif response.status_code == 401:
        raise APIError("Unauthorized", response.status_code)
    elif response.status_code == 404:
        raise APIError("Not found", response.status_code)
    else:
        raise APIError("Unknown error", response.status_code)


def check_assemblyai_api_key(api_key):
    url = "https://api.assemblyai.com/v2/transcript"
    params = {"Authorization": api_key}
    try:
        response = requests.get(url, headers=params)
    except:
        raise APIError("Bad request")
    if response.status_code == 200:
        print("Success!")
        return True
    elif response.status_code == 400:
        raise APIError("Bad request", response.status_code)
    elif response.status_code == 401:
        raise APIError("Unauthorized", response.status_code)
    elif response.status_code == 404:
        raise APIError("Not found", response.status_code)
    else:
        raise APIError("Unknown error", response.status_code)


def load_whisper(model_size):
    print(f"Loading OpenAI Whisper: {model_size}")
    model = WhisperModel(
        model_size, device=settings.value("device"), compute_type="float16"
    )
    return model


def load_wav2vec(model_size):
    if model_size == "vietnamese":
        print(f"Loading Facebook Wav2Vec: {model_size}")
        model_path = "nguyenvulebinh/wav2vec2-bartpho"
        model = SpeechEncoderDecoderModel.from_pretrained(model_path).eval()
        if torch.cuda.is_available():
            model = model.cuda()
        return model
    elif model_size == "english":
        print(f"Loading Facebook Wav2Vec: {model_size}")
        model_path = "facebook/wav2vec2-base-960h"
        model = Wav2Vec2Model.from_pretrained(model_path, torch_dtype=torch.float16)
        if torch.cuda.is_available():
            model = model.cuda()
        return model


def transcribe_wav2vec(model, signals, vad=False, stop_event=None):
    print("Transcribing using Facebook Wav2Vec")
    audio_path = os.path.join(BASE_DIR, "res/audio.wav")
    model_path = "nguyenvulebinh/wav2vec2-bartpho"
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
            (
                s
                if isinstance(s, str)
                else tokenizer.decode(s, skip_special_tokens=skip_special_tokens)
            )
            for s in outputs
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
            input_values["input_values"].to(device),
            attention_mask=input_values["attention_mask"].to(device),
            decoder_input_ids=tokenizer.batch_encode_plus(
                [prefix] * len(audio), return_tensors="pt"
            )["input_ids"][..., :-1].to(device),
            generation_config=GenerationConfig(
                decoder_start_token_id=tokenizer.bos_token_id
            ),
            max_length=250,
            num_beams=25,
            no_repeat_ngram_size=4,
            num_return_sequences=1,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # Decode the output tokens into text
        output_text = [
            decode_tokens(sequence) for sequence in output_beam_ids.sequences
        ]

        return output_text

    # A function to split the wav into chunks of 20 seconds and then decode each chunk

    def chunk_and_decode_wav(input_file, output_dir, chunk_duration=20):

        data, samplerate = sf.read(input_file)
        # Resample to 16kHz if necessary
        if samplerate != 16000:
            ratio = 16000 / samplerate
            data = samplerate.resample(data, ratio, "sinc_best")
        # Get the audio duration in milliseconds
        audio_length = len(data)
        chunk_length = chunk_duration * 1000  # Convert seconds to milliseconds

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Split and save chunks
        for i, start_time in enumerate(range(0, audio_length, chunk_length)):
            end_time = min(start_time + chunk_length, audio_length)
            chunk = data[start_time:end_time]
            sf.write(f"{output_dir}/chunk_{i + 1}.wav", chunk, samplerate, format="wav")
            print(f"Exported chunk {i + 1}: {start_time / 1000}s - {end_time / 1000}s")

        print("Audio splitting complete.")

    # https://huggingface.co/nguyenvulebinh/wav2vec2-bartpho/resolve/main/sample_news.wav

    if vad:
        vad_model = load_silero_vad()
        wav = read_audio(audio_path)
        speech_timestamp = get_speech_timestamps(wav, vad_model)

        # Get how many seconds of non-speech silero vad has detected
        silero_vad_speech = 0
        for i in range(len(speech_timestamp)):
            silero_vad_speech += (
                speech_timestamp[i]["end"] - speech_timestamp[i]["start"]
            )

        silero_vad_speech /= 16000  # sample rate

        second = silero_vad_speech % 60
        _minute = silero_vad_speech // 60
        minute = _minute % 60
        hour = _minute // 60

        print(
            f"Silero VAD detected {int(hour)} hours {int(minute)} minutes {int(second)} seconds of speech"
        )

        save_audio(
            os.path.join(BASE_DIR, "speech.wav"), collect_chunks(speech_timestamp, wav)
        )
        audio_path = os.path.join(BASE_DIR, "speech.wav")

    chunk_and_decode_wav(audio_path, os.path.join(BASE_DIR, "chunks"))
    chunk_files = sorted(glob.glob(os.path.join(BASE_DIR, "chunks/*.wav")))

    res = ""

    for chunk_file in chunk_files:
        if stop_event and stop_event.is_set():
            break
        result = ""
        # Load the chunk with torchaudio
        audio, sample_rate = torchaudio.load(chunk_file)

        # Resample to 16kHz if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000
            )
            audio = resampler(audio)

        # Ensure audio is mono
        if audio.shape[0] > 1:  # If multiple channels
            audio = torch.mean(audio, dim=0, keepdim=True)

        audio = audio.squeeze()  # Remove channel dimension
        transcription = decode_wav(audio, model)
        # print(transcription)
        pattern = [r"<\|\d+\.\d+\|", r"\|\d+\.\d+\|>"]
        for i in range(len(transcription)):
            fragment = re.sub("|".join(pattern), "", transcription[i])
            result += fragment
        if signals:
            signals.segment_added.emit(result)
        res += result

    # Clean up the chunks directory
    for chunk_file in chunk_files:
        os.remove(chunk_file)

    # clear cuda cache to avoid memory leak
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return res


def transcribe_whisper(model, language, signals, vad=True, stop_event=None):
    print("Transcribing using OpenAI Whisper")
    audio_path = os.path.join(BASE_DIR, "res/audio.wav")
    try:
        result = ""
        # Transcribe the audio
        segments, _ = model.transcribe(audio_path, vad_filter=vad, language=language)
        for segment in segments:
            # Since python is short-circuiting, if stop_event is None, the second condition will not be evaluated, so no error will be raised
            if stop_event and stop_event.is_set():
                print("Transcription stopped")
                break
            result += segment.text + " "
            if signals:
                signals.segment_added.emit(segment.text)
    except Exception as e:
        if signals:
            signals.error.emit(f"Error transcribing: {e}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("end")

    return result


def transcribe_deepgram(client, language):
    print("Transcribing using Deepgram")
    audio_path = os.path.join(BASE_DIR, "res/audio.wav")
    with open(audio_path, "rb") as file:
        buffer_data = file.read()

    payload: FileSource = {
        "buffer": buffer_data,
    }

    # STEP 2: Configure Deepgram options for audio analysis
    options = PrerecordedOptions(
        model="nova-2",
        smart_format=True,
    )

    if language:
        options.language = language
    else:
        options.detect_language = True

    # STEP 3: Call the transcribe_file method with the text payload and options
    response = client.listen.rest.v("1").transcribe_file(
        payload, options, timeout=httpx.Timeout(300, connect=10.0)
    )

    # STEP 4: Print the response
    res_json = response.to_json(indent=4)
    res_json = json.loads(res_json)

    transcription = res_json["results"]["channels"][0]["alternatives"][0]["transcript"]
    confidence = res_json["results"]["channels"][0]["alternatives"][0]["confidence"]
    print(f"Transcription: {transcription}")
    print(f"Confidence: {confidence}")
    return transcription


def transcribe_assemblyai(client, language):
    print("Transcribing using AssemblyAI")
    audio_path = os.path.join(BASE_DIR, "res/audio.wav")
    if language:
        config = aai.TranscriptionConfig(language_code=language)
    else:
        config = aai.TranscriptionConfig(language_detection=True)
    client.config = config

    transcript = client.transcribe(audio_path)

    if transcript.status == aai.TranscriptStatus.error:
        print("Transcription failed: ", transcript.error)
        return "Transcription failed"
    else:
        print(transcript.text)
        return transcript.text


def download_yt_link(url):
    os.makedirs(os.path.join(BASE_DIR, "downloads"), exist_ok=True)
    path = os.path.join(BASE_DIR, "downloads/test")
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "ffmpeg_location": os.path.join(BASE_DIR, "binaries"),
        "outtmpl": str(path),
    }
    print(ydl_opts["ffmpeg_location"])
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print(e)
        return None
    else:
        file = os.path.join(BASE_DIR, "downloads/test.wav")
        return file


class YoutubeDLThread(QRunnable):
    def __init__(self, url):
        super().__init__()
        self.url = url
        self.signals = YoutubeDLSignal()

    def run(self):
        file = download_yt_link(self.url)
        self.signals.finished.emit()
        if file:
            self.signals.result.emit(file, self.url)
        else:
            self.signals.error.emit("Error downloading YouTube link")


class YoutubeDLSignal(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    result = pyqtSignal(str, str)


class ModelLoadThread(QRunnable):
    def __init__(self, text):
        super().__init__()
        self.text = text
        self.signals = ModelLoadSignal()

    def run(self):
        if self.text.startswith("OpenAI Whisper"):
            model = load_whisper(self.text.split(":")[-1].strip())
            self.signals.result.emit(model)
            self.signals.finished.emit()
        elif self.text.startswith("Facebook Wav2Vec"):
            model = load_wav2vec(self.text.split(":")[-1].strip())
            self.signals.result.emit(model)
            self.signals.finished.emit()


class ModelLoadSignal(QObject):
    finished = pyqtSignal()
    result = pyqtSignal(object)
    error = pyqtSignal(str)


class TranscribeThread(QRunnable):
    def __init__(
        self,
        model_name,
        model,
        audio_path,
        language,
        clients,
        whisper_vad=False,
        w2v_vad=False,
    ):
        super().__init__()
        self.model_name = model_name
        self.model = model
        self.audio_path = audio_path
        self.signals = TranscribeSignal()
        self.language = language
        self.clients = clients
        self.signals = TranscribeSignal()
        self.whisper_vad = whisper_vad
        self.w2v_vad = w2v_vad
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    def run(self):
        preprocess_audio(self.audio_path)
        if self.model_name.startswith("OpenAI Whisper"):
            transcribe_whisper(
                self.model,
                self.language,
                self.signals,
                self.whisper_vad,
                self.stop_event,
            )
            self.signals.finished.emit()
        elif self.model_name.startswith("Facebook Wav2Vec"):
            transcribe_wav2vec(self.model, self.signals, self.w2v_vad, self.stop_event)
            self.signals.finished.emit()
        elif self.model_name == "DeepGram":
            result = transcribe_deepgram(self.clients["DeepGram"], self.language)
            self.signals.result.emit(result)
            self.signals.finished.emit()
        elif self.model_name == "AssemblyAI":
            result = transcribe_assemblyai(self.clients["AssemblyAI"], self.language)
            self.signals.result.emit(result)
            self.signals.finished.emit()
        os.remove(os.path.join(BASE_DIR, "res/audio.wav"))


class TranscribeSignal(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    result = pyqtSignal(str)
    segment_added = pyqtSignal(str)
