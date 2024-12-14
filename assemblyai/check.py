import requests
import json

with open('../secret.json') as f:
    data = json.load(f)
    api_key = data['ASSEMBLYAI_API_KEY']

base_url = 'https://api.assemblyai.com/'
query = {'check': 'v2/transcript'}

params = {'Authorization': api_key}

response = requests.get(base_url + query['check'], headers=params, json=query)
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