import requests

def check_deepgram_api_key(api_key):
    url = 'https://api.deepgram.com/v1/auth/token'
    params = {'Authorization': f'Token {api_key}'}
    response = requests.get(url, headers=params)
    if response.status_code == 200:
        print('Success!')
        return True
    elif response.status_code == 400:
        print('Bad request')
        return False
    elif response.status_code == 401:
        print('Unauthorized')
        return False
    elif response.status_code == 404:
        print('Not found')
        return False
    else:
        print('Unknown error')
        return False

def check_assemblyai_api_key(api_key):
    url = 'https://api.assemblyai.com/v2/transcript'
    params = {'Authorization': api_key}
    response = requests.get(url, headers=params)
    if response.status_code == 200:
        print('Success!')
        return True
    elif response.status_code == 400:
        print('Bad request')
        return False
    elif response.status_code == 401:
        print('Unauthorized')
        return False
    elif response.status_code == 404:
        print('Not found')
        return False
    else:
        print('Unknown error')
        return False

def load_whisper(model_size):
    print(f'Loading OpenAI Whisper: {model_size}')

def load_wav2vec(model_size):
    print(f'Loading Facebook Wav2Vec: {model_size}')