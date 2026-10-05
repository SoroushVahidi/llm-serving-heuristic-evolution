import os
import requests
import json

ACCESS_TOKEN = os.environ.get('ZENODO_API_TOKEN')
bucket = "https://zenodo.org/api/files/7fa96e91-730f-417f-a52b-312671c60c4b"
deposition_id = "22865294"

file_path = "release/performance_evaluation_v1.zip"
filename = os.path.basename(file_path)

# 1. Upload file
params = {'access_token': ACCESS_TOKEN}
with open(file_path, "rb") as fp:
    r = requests.put(
        f"{bucket}/{filename}",
        data=fp,
        params=params,
    )

if r.status_code != 201:
    print(f"Error uploading: {r.status_code}")
    print(r.json())
    exit(1)
else:
    print("File uploaded successfully!")

# 2. Publish deposition
r = requests.post(
    f"https://zenodo.org/api/deposit/depositions/{deposition_id}/actions/publish",
    params=params
)

if r.status_code == 202:
    print("Deposition published successfully!")
    print(r.json()['doi_url'])
else:
    print(f"Error publishing: {r.status_code}")
    print(r.json())
    exit(1)
