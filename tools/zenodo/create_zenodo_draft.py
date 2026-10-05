import os
import requests
import json

ACCESS_TOKEN = os.environ.get('ZENODO_API_TOKEN')
headers = {"Content-Type": "application/json"}
params = {'access_token': ACCESS_TOKEN}

with open('release/performance_evaluation_v1/.zenodo.json', 'r') as f:
    metadata = json.load(f)

data = {'metadata': metadata}

response = requests.post('https://zenodo.org/api/deposit/depositions',
                         params=params,
                         json=data,
                         headers=headers)

if response.status_code == 201:
    res = response.json()
    print(json.dumps({
        'id': res['id'],
        'doi': res['metadata']['prereserve_doi']['doi'],
        'bucket': res['links']['bucket']
    }))
else:
    print(f"Error: {response.status_code}")
    print(response.json())
