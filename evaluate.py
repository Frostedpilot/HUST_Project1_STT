import argparse
import os
import jiwer
import datetime
import json
from app.utility import (
    load_whisper,
    load_wav2vec,
    transcribe_whisper,
    transcribe_wav2vec,
    transcribe_assemblyai,
    transcribe_deepgram,
    preprocess_audio,
    update_utility_base_dir,
)
from deepgram import DeepgramClient
import assemblyai as aai


def build_text_normalizer_wer():
    text_normalizer = jiwer.Compose(
        [
            jiwer.Strip(),
            jiwer.RemoveSpecificWords(
                [
                    ".",
                    ",",
                    "?",
                    "!",
                    ":",
                    ";",
                    "-",
                    "(",
                    ")",
                    "[",
                    "]",
                    "{",
                    "}",
                    "'",
                    '"',
                ]
            ),
            jiwer.RemoveMultipleSpaces(),
            jiwer.ToLowerCase(),
            jiwer.ReduceToSingleSentence(),
            jiwer.ReduceToListOfListOfWords(),
        ]
    )
    return text_normalizer


def build_text_normalizer_cer():
    text_normalizer = jiwer.Compose(
        [
            jiwer.Strip(),
            jiwer.RemoveSpecificWords(
                [
                    ".",
                    ",",
                    "?",
                    "!",
                    ":",
                    ";",
                    "-",
                    "(",
                    ")",
                    "[",
                    "]",
                    "{",
                    "}",
                    "'",
                    '"',
                ]
            ),
            jiwer.RemoveMultipleSpaces(),
            jiwer.ToLowerCase(),
            jiwer.ReduceToSingleSentence(),
            jiwer.ReduceToListOfListOfChars(),
        ]
    )
    return text_normalizer


def evaluate_model(
    audio_dir,
    model_name,
    language,
    log_dir="evaluation_logs",
    model_size=None,
    vad=False,
):
    os.makedirs(log_dir, exist_ok=True)

    text_normalizer_wer = build_text_normalizer_wer()
    text_normalizer_cer = build_text_normalizer_cer()

    with open("secret.json", "r") as f:
        api_keys = json.load(f)
        deepgram_api_key = api_keys["DEEPGRAM_API_KEY"]
        assemblyai_api_key = api_keys["ASSEMBLYAI_API_KEY"]

    if language == "en":
        available_models = {
            "local": {
                "OpenAI Whisper": ["tiny", "base", "medium", "turbo"],
                "Facebook Wav2Vec": ["english"],
            },
            "api": {
                "Deepgram": deepgram_api_key,
                "AssemblyAI": assemblyai_api_key,
            },
        }
    elif language == "vi":
        available_models = {
            "local": {
                "OpenAI Whisper": ["medium", "turbo"],
                "Facebook Wav2Vec": ["vietnamese"],
            },
            "api": {
                "Deepgram": deepgram_api_key,
                "AssemblyAI": assemblyai_api_key,
            },
        }

    models_to_evaluate = {"local": {}, "api": {}}
    if model_name:
        if model_name in available_models["local"]:
            if model_size and model_size in available_models["local"][model_name]:
                models_to_evaluate["local"][model_name] = [model_size]
            elif not model_size:
                models_to_evaluate["local"][model_name] = available_models["local"][
                    model_name
                ]
            else:
                print(
                    f"Invalid model size '{model_size}' for model '{model_name}'. Available sizes are: {available_models["local"][model_name]}"
                )
                return
        elif model_name in available_models["api"]:
            models_to_evaluate["api"][model_name] = available_models["api"][model_name]
        else:
            print(
                f"Invalid model name: {model_name}. Available models are: {list(available_models[i] for i in available_models.keys())}"
            )
            return
    else:
        models_to_evaluate = available_models

    _evaluate_local(
        models_to_evaluate["local"],
        audio_dir,
        language,
        log_dir,
        text_normalizer_wer,
        text_normalizer_cer,
        vad,
    )
    _evaluate_api(
        models_to_evaluate["api"],
        audio_dir,
        language,
        log_dir,
        text_normalizer_wer,
        text_normalizer_cer,
    )


def _evaluate_local(
    models_to_evaluate,
    audio_dir,
    language,
    log_dir,
    text_normalizer_wer,
    text_normalizer_cer,
    vad,
):
    for model_name, model_sizes in models_to_evaluate.items():
        for model_size in model_sizes:
            print(f"Evaluating model: {model_name} ({model_size})")

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file_path = os.path.join(log_dir, "evaluation_log.txt")

            if model_name == "OpenAI Whisper":
                model = load_whisper(model_size)
            elif model_name == "Facebook Wav2Vec":
                model = load_wav2vec(model_size)
            else:
                print(f"Unsupported model: {model_name}")
                continue  # Skip to the next model

            total_wer = 0
            total_cer = 0
            num_files = 0

            for filename in os.listdir(audio_dir):
                if filename.endswith(".wav") or filename.endswith(".mp3"):
                    audio_path = os.path.join(audio_dir, filename)
                    transcript_path = os.path.splitext(audio_path)[0] + ".txt"

                    if os.path.exists(transcript_path):
                        try:
                            with open(transcript_path, "r", encoding="utf-8") as f:
                                reference_text = f.read().strip()

                            preprocess_audio(audio_path)
                            if model_name == "OpenAI Whisper":
                                hypothesis_text = transcribe_whisper(
                                    model, language, signals=None, vad=vad
                                )
                            elif model_name == "Facebook Wav2Vec":
                                hypothesis_text = transcribe_wav2vec(
                                    model, signals=None, vad=vad
                                )
                            os.remove("res/audio.wav")

                            wer = jiwer.wer(
                                reference_text,
                                hypothesis_text,
                                truth_transform=text_normalizer_wer,
                                hypothesis_transform=text_normalizer_wer,
                            )
                            cer = jiwer.cer(
                                reference_text,
                                hypothesis_text,
                                truth_transform=text_normalizer_cer,
                                hypothesis_transform=text_normalizer_cer,
                            )

                            total_wer += wer
                            total_cer += cer
                            num_files += 1

                            print(
                                f"Processed: {filename} - WER: {wer:.4f}, CER: {cer:.4f}"
                            )

                        except Exception as e:
                            print(f"Error processing {filename}: {e}")

            avg_wer = total_wer / num_files if num_files > 0 else 0
            avg_cer = total_cer / num_files if num_files > 0 else 0

            # Log the results to a file
            with open(log_file_path, "a") as log_file:
                log_file.write(f"Model: {model_name}\n")
                log_file.write(f"Model Size: {model_size}\n")
                log_file.write(f"Language: {language}\n")
                log_file.write(f"Timestamp: {timestamp}\n")
                log_file.write(f"Average WER: {avg_wer:.4f}\n")
                log_file.write(f"Average CER: {avg_cer:.4f}\n")
                log_file.write("*" * 50 + "\n")

            print(
                f"Evaluation complete for {model_name} ({model_size}). Results logged to: {log_file_path}"
            )


def _evaluate_api(
    models_to_evaluate,
    audio_dir,
    language,
    log_dir,
    text_normalizer_wer,
    text_normalizer_cer,
):
    for model_name, api_key in models_to_evaluate.items():
        print(f"Evaluating model: {model_name}")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(log_dir, "evaluation_log.txt")

        if model_name == "Deepgram":
            client = DeepgramClient(api_key)
        elif model_name == "AssemblyAI":
            aai.settings.api_key = api_key
            client = aai.Transcriber()

        total_wer = 0
        total_cer = 0
        num_files = 0

        for filename in os.listdir(audio_dir):
            if filename.endswith(".wav") or filename.endswith(".mp3"):
                audio_path = os.path.join(audio_dir, filename)
                transcript_path = os.path.splitext(audio_path)[0] + ".txt"

                if os.path.exists(transcript_path):
                    try:
                        with open(transcript_path, "r", encoding="utf-8") as f:
                            reference_text = f.read().strip()

                        preprocess_audio(audio_path)
                        if model_name == "Deepgram":
                            hypothesis_text = transcribe_deepgram(client, "vi")
                        elif model_name == "AssemblyAI":
                            hypothesis_text = transcribe_assemblyai(client, "vi")

                        wer = jiwer.wer(
                            reference_text,
                            hypothesis_text,
                            truth_transform=text_normalizer_wer,
                            hypothesis_transform=text_normalizer_wer,
                        )
                        cer = jiwer.cer(
                            reference_text,
                            hypothesis_text,
                            truth_transform=text_normalizer_cer,
                            hypothesis_transform=text_normalizer_cer,
                        )

                        total_wer += wer
                        total_cer += cer
                        num_files += 1

                        print(f"Processed: {filename} - WER: {wer:.4f}, CER: {cer:.4f}")

                    except Exception as e:
                        print(f"Error processing {filename}: {e}")

        avg_wer = total_wer / num_files if num_files > 0 else 0
        avg_cer = total_cer / num_files if num_files > 0 else 0

        # Log the results to a file
        with open(log_file_path, "a") as log_file:
            log_file.write(f"Model: {model_name}\n")
            log_file.write(f"Language: {language}\n")
            log_file.write(f"Timestamp: {timestamp}\n")
            log_file.write(f"Average WER: {avg_wer:.4f}\n")
            log_file.write(f"Average CER: {avg_cer:.4f}\n")
            log_file.write("*" * 50 + "\n")

        print(
            f"Evaluation complete for {model_name}. Results logged to: {log_file_path}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate speech-to-text models.")
    parser.add_argument("audio_dir", type=str, help="Path to the audio directory")
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model to evaluate (e.g., 'OpenAI Whisper', 'Facebook Wav2Vec'). If not specified, all available models will be evaluated.",
        required=False,
    )
    parser.add_argument(
        "--language",
        type=str,
        default="vi",
        help="Language code for transcription (default: vi)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="evaluation_logs",
        help="Directory to save evaluation logs (default: evaluation_logs)",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        help="Size of the model to load, only for whisper model (e.g., 'tiny', 'base', 'medium')",
        required=False,
    )
    parser.add_argument(
        "--vad",
        type=str,
        help="Whether to use VAD for silence removal (default: False)",
        default="False",
        required=False,
    )

    args = parser.parse_args()

    args.vad = args.vad.lower() == "true"

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    update_utility_base_dir(BASE_DIR)

    evaluate_model(
        args.audio_dir,
        args.model_name,
        args.language,
        args.log_dir,
        args.model_size,
        args.vad,
    )
