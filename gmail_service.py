import os
import base64
from email.mime.text import MIMEText
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def create_gmail_service():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    service = build('gmail', 'v1', credentials=creds)
    return service

def send_email(to_email, subject, body):
    service = create_gmail_service()
    message = MIMEText(body, 'html')
    message['to'] = to_email
    message['from'] = 'me'
    message['subject'] = subject
    encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    create_message = {'raw': encoded_message}
    sent = service.users().messages().send(userId='me', body=create_message).execute()
    print(f"Email sent to {to_email}, id: {sent['id']}")
