import json

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)

# Path to the audio file
AUDIO_FILE = "bruh.wav"

def main():
    # Read the Deepgram API key from the secret json file
    with open("../secret.json", "r") as file:
        secret = json.load(file)
        api_key = secret["DEEPGRAM_API_KEY"]       
    
    # STEP 1 Create a Deepgram client using the API key
    deepgram = DeepgramClient(api_key=api_key)

    with open(AUDIO_FILE, "rb") as file:
        buffer_data = file.read()

    payload: FileSource = {
        "buffer": buffer_data,
    }

    #STEP 2: Configure Deepgram options for audio analysis
    options = PrerecordedOptions(
        model="nova-2",
        language="vi",
        smart_format=True,
    )

    # STEP 3: Call the transcribe_file method with the text payload and options
    response = deepgram.listen.rest.v("1").transcribe_file(payload, options)

    # STEP 4: Print the response
    res_json = response.to_json(indent=4)
    res_json = json.loads(res_json)

    transcription = res_json["results"]["channels"][0]["alternatives"][0]["transcript"]
    confidence = res_json["results"]["channels"][0]["alternatives"][0]["confidence"]
    print(f"Transcription: {transcription}")
    print(f"Confidence: {confidence}")



if __name__ == "__main__":
    main()