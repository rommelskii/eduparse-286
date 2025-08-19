from __future__ import print_function
import os.path
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Define the scope
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# Authenticate and build the service
def authenticate():
    creds = None
    if os.path.exists('./backend/credentials.json'):
        creds = Credentials.from_authorized_user_file('./backend/credentials.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('./backend/credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)

# Upload a file to Google Drive
def upload_file(file_path, mime_type):
    service = authenticate()
    file_metadata = {'name': os.path.basename(file_path)}
    media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f"Uploaded {file_path} with File ID: {file.get('id')}")

# Example usage
upload_file('./inputs/audio/mlk.old.mp3', 'audio/mp3')


