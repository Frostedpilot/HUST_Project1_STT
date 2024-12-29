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
    - Deepgram
    - AssemblyAI
- **Language Support:**
  - English
  - Vietnamese
  - Automatic language detection (for some models)
- **Voice Activity Detection (VAD):**
  - Option to use Silero VAD for removing silent segments before transcription with Whisper and Wav2Vec 2.0 (may improve accuracy).

## Getting Started

### Prerequisites

- **Python 3.12+:** Ensure you have Python 3.12 or a later version installed on your system.
- **FFmpeg (Optional):** This is required for audio processing. Although the binaries for both `ffmpeg` and `ffprobe` are provided with the installer, it is still recommended to install FFmpeg and add the binaries to PATH to avoid errors.
- **Git (Optional):** To clone the repository, you'll need Git installed.
- **Required Python Packages:** The required packages are listed in `requirements.txt`.

### Installation

#### Use the installer

Simply download the installer from the release page and follow the instructions.

#### Use the code directly

1. **Clone the repository (or download the source code):**

   ```bash
   git clone https://github.com/Frostedpilot/HUST_Project1_STT.git
   cd HUST_Project1_STT
   ```

   If you don't have Git, download the source code as a ZIP file and extract it.

2. **Create and Activate a Virtual Environment (Recommended):**

   ```bash
   # Using venv (recommended)
   python3 -m venv packenv
   .venv\Scripts\activate

   # Or using virtualenv (if you prefer)
   pip install virtualenv
   virtualenv packenv
   .venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

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

   ```bash
   # Using venv (recommended)
   python3 -m venv packenv
   .venv\Scripts\activate

   # Or using virtualenv (if you prefer)
   pip install virtualenv
   virtualenv packenv
   .venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Build the executable:**

   ```bash
   PyInstaller app.spec
   ```

5. **Create the installer (Optional):**

### API Keys

To use the Deepgram and AssemblyAI models, you need to obtain API keys from their respective websites. Refer to their respective websites for instructions.

## Usage

1. **Launch the Application:**
   - Run the executable file created by PyInstaller.
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

## Future Works

- **Support for More Models:** Integrate additional speech-to-text models and APIs.
- **Improved Error Handling:** Implement more robust error handling and user feedback mechanisms.
- **Advanced Settings:** Expose more model-specific parameters and audio preprocessing options to the user.
- **Real-time Transcription:** Add support for real-time transcription from a microphone or audio stream.
- **Additional Features:**
  - Saving transcriptions to files (plain text, SRT, etc.).
  - Speaker diarization (identifying different speakers in the audio).
  - Sentiment analysis of the transcribed text.
