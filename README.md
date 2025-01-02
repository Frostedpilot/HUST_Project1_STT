# Speech-to-Text App (HUST Project 1)

## Table of Contents

- [About](#about)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Future Works](#future-works)

## About

This application is a speech-to-text (STT) system developed as part of the Project 1 course at the Hanoi University of Science and Technology (HUST). It leverages state-of-the-art deep learning models and APIs to provide accurate and efficient audio transcription. The application supports multiple transcription models, including OpenAI's Whisper, Facebook's Wav2Vec 2.0, and APIs from Deepgram and AssemblyAI. It is designed to be user-friendly, allowing users to easily transcribe audio from local files.

## Features

- **Multiple Transcription Models:**
  - Local models:
    - OpenAI Whisper (various sizes: Tiny, Base, Medium, Large and Turbo)
    - Facebook Wav2Vec 2.0 (only Vietnamese model for now)
  - API-based models:
    - [Deepgram](https://developers.deepgram.com/)
    - [AssemblyAI](https://www.assemblyai.com/docs/)
- **Language Support:**
  - English
  - Vietnamese
  - Automatic language detection (for some models)
- **Voice Activity Detection (VAD):**
  - Option to use Silero VAD for removing silent segments before transcription with Whisper and Wav2Vec 2.0 (may improve accuracy).

## Getting Started

### Prerequisites

- **Python 3.12+:** Ensure you have Python 3.12 or a later version installed on your system.
- **FFmpeg (Optional):** This is required for audio processing. Although the binaries for both `ffmpeg` and `ffprobe` are provided with the installer, it is required to install FFmpeg and either add the binaries to PATH or copy the binaries to the `app/binaries/` folder if you want to work with the source code. You can download FFmpeg from the official website: https://ffmpeg.org/download.html
- **Git (Optional):** To clone the repository, you'll need Git installed. You can download Git from: https://git-scm.com/downloads
- **Required Python Packages:** The required packages are listed in `requirements.txt`.

### Installation

#### Use the code directly

1. **Clone the repository (or download the source code):**

   ```bash
   git clone https://github.com/Frostedpilot/HUST_Project1_STT.git
   cd HUST_Project1_STT
   ```

   If you don't have Git, download the source code as a ZIP file and extract it.

2. **Create and Activate a Virtual Environment (Recommended):**

   Using venv (recommended)

   ```bash
   python3 -m venv packenv
   .\packenv\Scripts\activate
   ```

   Or using virtualenv (if you prefer)

   ```bash
   pip install virtualenv
   virtualenv packenv
   .\packenv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   And install the appropriate Pytorch version (version 2.5.1, with or without cuda) from its website: https://pytorch.org/get-started/locally/

4. **Run the code**
   ```bash
   python3 app/app.py
   ```

#### Build from source

1. **Clone the repository (or download the source code):**

   ```bash
   git clone https://github.com/Frostedpilot/HUST_Project1_STT.git
   cd HUST_Project1_STT
   ```

   If you don't have Git, download the source code as a ZIP file and extract it.

2. **Create and Activate a Virtual Environment (Recommended):**

   Using venv (recommended)

   ```bash
   python3 -m venv packenv
   .\packenv\Scripts\activate
   ```

   Or using virtualenv (if you prefer)

   ```bash
   pip install virtualenv
   virtualenv packenv
   .\packenv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   And install the appropriate Pytorch version (version 2.5.1, with or without cuda) from its website: https://pytorch.org/get-started/locally/
4. **Build the executable:**

   ```bash
   PyInstaller app.spec
   ```

5. **Create the installer (Optional):**

### API Keys

To use the Deepgram and AssemblyAI models, you need to obtain API keys from their respective websites. Refer to their respective websites for instructions:

- Deepgram: https://developers.deepgram.com/
- AssemblyAI: https://www.assemblyai.com/docs/

## Usage

1. **Launch the Application:**
   - Run the executable file created by PyInstaller.
   - Or run the command:
   ```bash
   python3 app/app.py
   ```
2. **Select a Model:**
   - Choose the desired transcription model from the "Model" dropdown menu.
   - If using a local model, select the model size (e.g., "tiny," "base," "medium").
3. **Select a Language:**
   - Choose the language of the audio file from the "Language" dropdown menu.
4. **Choose an Audio File:**
   - Click the "Choose File" button and select the audio file you want to transcribe.
5. **Start Transcription:**
   - Click the "Start" button to begin the transcription process.
   - The transcribed text will appear in the text area below.
6. **Stop Transcription:**
   - Click the "Stop" button to end the transcription process.
7. **(Optional) Configure Settings:**
   - Access the "Settings" menu to configure API keys, default models, and other preferences.

## Troubleshooting

- **Error: 'charmap' codec can't decode byte...**
  - This error indicates an encoding issue with your ground truth text files. Make sure the text files are encoded in UTF-8. You can change the encoding using a text editor like Notepad.
- **Application stops without error when loading local model.**
  - This is likely due to insufficient memory (RAM or GPU VRAM). Try closing unnecessary programs, using a smaller model, split the audio into smaller chunks or use the API-based models instead of local models.
- **Error: `Requested float16 compute type, but the target device or backend do not support efficient float16 computation.`**
  - This means your hardware or software backend doesn't fully support `float16` precision. The application will automatically switch to `float32` in this case. While this ensures the application runs, local models will run slower and potentially have slightly lower accuracy. If speed is critical, you might need to upgrade your hardware or use API-based models.
- **Other notes:**
  - While the models are capable of processing audio longer than 2 hours, you might experience significant delays when using audio files longer than 1 hour. This is because preprocessing audio for local models and uploading audio data for API-based models can take a very long time, and network issues may arise. For a smoother experience, it is recommend using audio up to 1 hour long.
  - The Whisper and Wav2Vec2 models are by default downloaded to the `C:\Users\YourUserName\.cache\huggingface\hub\` folder. If you want to delete the local model files, go to that folder and delete any folder that has `wav2vec` or `whisper` in their name
  - Larger Whisper models can use up a lot of GPU VRAM. For referenced, my NVIDIA GeForce RTX 3050Ti Mobile 4GB can barely handle Whisper Turbo, and the only model that it can't run is the Whisper Large model.
  - When building the .exe file and the Windows installer, Windows Defender will flag the app as Trojan. This is false positive, as can be inspected from the source code, there is no malicious code injected.
  - The console when using built exe file can be turned off by setting `console=False` in the `EXE` class in `app.spec` file. Note that doing this can further trigger Windows Defender.
- **Other Issues:**
  - If you encounter any other issues, please check the [GitHub Issues](https://github.com/Frostedpilot/HUST_Project1_STT/issues) page to see if the problem has already been reported. If not, please open a new issue and provide as much detail as possible, including error messages, operating system, and steps to reproduce the problem.

## Future Works

- **Support for More Models:** Integrate additional speech-to-text models and APIs.
- **Improved Error Handling:** Implement more robust error handling and user feedback mechanisms.
- **Advanced Settings:** Expose more model-specific parameters and audio preprocessing options to the user.
- **Real-time Transcription:** Add support for real-time transcription from a microphone or audio stream.
- **Additional Features:**
  - Saving transcriptions to files (plain text, SRT, etc.).
  - Speaker diarization (identifying different speakers in the audio).
  - Sentiment analysis of the transcribed text.
