import requests
import json

with open('../secret.json') as f:
    data = json.load(f)
    api_key = data['DEEPGRAM_API_KEY']

base_url = 'https://api.deepgram.com/v1/'
query = {'check': 'auth/token'}

params = {'Authorization': f'Token {api_key}'}

response = requests.get(base_url + query['check'], headers=params)
if response.status_code == 200:
    print('Success!')
elif response.status_code == 400:
    print('Bad request')
elif response.status_code == 401:
    print('Unauthorized')
elif response.status_code == 404:
    print('Not found')
else:
    print('Unknown error')