"""
Google Drive Integration Module
Handles authentication and file operations with Google Drive.
"""

import os
import json
import io
from pathlib import Path
from typing import Optional, List, Dict, Any

import streamlit as st
import pandas as pd

# Google Drive API imports
try:
    from google.oauth2.credentials import Credentials
    from google.oauth2 import service_account
    from google_auth_oauthlib.flow import Flow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
    GDRIVE_AVAILABLE = True
except ImportError:
    GDRIVE_AVAILABLE = False


# OAuth scopes required
SCOPES = [
    'https://www.googleapis.com/auth/drive.file',
    'https://www.googleapis.com/auth/drive.readonly'
]


class GoogleDriveManager:
    """Manages Google Drive operations with OAuth authentication."""
    
    def __init__(self):
        self.service = None
        self.credentials = None
    
    def authenticate_with_service_account(self, credentials_json: Dict) -> bool:
        """Authenticate using a service account."""
        try:
            credentials = service_account.Credentials.from_service_account_info(
                credentials_json,
                scopes=SCOPES
            )
            self.credentials = credentials
            self.service = build('drive', 'v3', credentials=credentials)
            return True
        except Exception as e:
            st.error(f"Service account authentication failed: {e}")
            return False
    
    def authenticate_with_oauth(self, client_config: Dict) -> Optional[str]:
        """
        Start OAuth flow and return authorization URL.
        For Streamlit, this needs to be handled in multiple steps.
        """
        try:
            flow = Flow.from_client_config(
                client_config,
                scopes=SCOPES,
                redirect_uri='urn:ietf:wg:oauth:2.0:oob'  # For desktop apps
            )
            
            auth_url, _ = flow.authorization_url(prompt='consent')
            return auth_url, flow
        except Exception as e:
            st.error(f"OAuth initialization failed: {e}")
            return None, None
    
    def complete_oauth(self, flow: Flow, auth_code: str) -> bool:
        """Complete OAuth flow with authorization code."""
        try:
            flow.fetch_token(code=auth_code)
            self.credentials = flow.credentials
            self.service = build('drive', 'v3', credentials=self.credentials)
            return True
        except Exception as e:
            st.error(f"OAuth completion failed: {e}")
            return False
    
    def list_files(
        self,
        folder_id: Optional[str] = None,
        file_type: str = 'csv',
        max_results: int = 50
    ) -> List[Dict]:
        """List files from Google Drive."""
        if not self.service:
            return []
        
        try:
            query_parts = []
            
            # Filter by folder
            if folder_id:
                query_parts.append(f"'{folder_id}' in parents")
            
            # Filter by file type
            if file_type == 'csv':
                query_parts.append("mimeType='text/csv'")
            elif file_type == 'json':
                query_parts.append("mimeType='application/json'")
            
            query_parts.append("trashed=false")
            query = " and ".join(query_parts)
            
            results = self.service.files().list(
                q=query,
                pageSize=max_results,
                fields="files(id, name, mimeType, modifiedTime, size)"
            ).execute()
            
            return results.get('files', [])
        except Exception as e:
            st.error(f"Error listing files: {e}")
            return []
    
    def download_file(self, file_id: str) -> Optional[bytes]:
        """Download a file from Google Drive."""
        if not self.service:
            return None
        
        try:
            request = self.service.files().get_media(fileId=file_id)
            file_buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(file_buffer, request)
            
            done = False
            while not done:
                status, done = downloader.next_chunk()
            
            file_buffer.seek(0)
            return file_buffer.read()
        except Exception as e:
            st.error(f"Error downloading file: {e}")
            return None
    
    def download_csv(self, file_id: str) -> Optional[pd.DataFrame]:
        """Download a CSV file and return as DataFrame."""
        content = self.download_file(file_id)
        if content:
            try:
                return pd.read_csv(io.BytesIO(content))
            except Exception as e:
                st.error(f"Error parsing CSV: {e}")
        return None
    
    def upload_file(
        self,
        content: bytes,
        filename: str,
        mime_type: str = 'text/csv',
        folder_id: Optional[str] = None
    ) -> Optional[str]:
        """Upload a file to Google Drive."""
        if not self.service:
            return None
        
        try:
            file_metadata = {'name': filename}
            if folder_id:
                file_metadata['parents'] = [folder_id]
            
            media = MediaIoBaseUpload(
                io.BytesIO(content),
                mimetype=mime_type,
                resumable=True
            )
            
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            return file.get('id')
        except Exception as e:
            st.error(f"Error uploading file: {e}")
            return None
    
    def update_file(
        self,
        file_id: str,
        content: bytes,
        mime_type: str = 'text/csv'
    ) -> bool:
        """Update an existing file in Google Drive."""
        if not self.service:
            return False
        
        try:
            media = MediaIoBaseUpload(
                io.BytesIO(content),
                mimetype=mime_type,
                resumable=True
            )
            
            self.service.files().update(
                fileId=file_id,
                media_body=media
            ).execute()
            
            return True
        except Exception as e:
            st.error(f"Error updating file: {e}")
            return False


def render_gdrive_auth_ui():
    """Render Google Drive authentication UI in Streamlit."""
    
    if not GDRIVE_AVAILABLE:
        st.error("""
        Google Drive libraries not installed. 
        Run: `pip install google-api-python-client google-auth-oauthlib`
        """)
        return None
    
    st.markdown("### 🔐 Google Drive Authentication")
    
    auth_method = st.radio(
        "Authentication Method",
        options=['Service Account (Recommended)', 'OAuth 2.0'],
        help="Service Account is easier for automation; OAuth requires user consent"
    )
    
    if auth_method == 'Service Account (Recommended)':
        st.info("""
        **To use a Service Account:**
        1. Go to [Google Cloud Console](https://console.cloud.google.com/)
        2. Create a project and enable Google Drive API
        3. Create a Service Account and download JSON key
        4. Share your Drive folders with the service account email
        """)
        
        creds_file = st.file_uploader(
            "Upload Service Account JSON",
            type=['json'],
            key="service_account"
        )
        
        if creds_file:
            try:
                creds_json = json.load(creds_file)
                gdrive = GoogleDriveManager()
                
                if gdrive.authenticate_with_service_account(creds_json):
                    st.success("✅ Authenticated successfully!")
                    return gdrive
            except Exception as e:
                st.error(f"Invalid credentials file: {e}")
    
    else:  # OAuth 2.0
        st.info("""
        **To use OAuth 2.0:**
        1. Go to [Google Cloud Console](https://console.cloud.google.com/)
        2. Create OAuth 2.0 credentials (Desktop app)
        3. Download the client secrets JSON
        """)
        
        client_file = st.file_uploader(
            "Upload OAuth Client Secrets",
            type=['json'],
            key="oauth_client"
        )
        
        if client_file:
            try:
                client_config = json.load(client_file)
                gdrive = GoogleDriveManager()
                
                auth_url, flow = gdrive.authenticate_with_oauth(client_config)
                
                if auth_url:
                    st.markdown(f"**[Click here to authorize]({auth_url})**")
                    
                    auth_code = st.text_input(
                        "Enter authorization code:",
                        type="password"
                    )
                    
                    if auth_code and st.button("Complete Authentication"):
                        if gdrive.complete_oauth(flow, auth_code):
                            st.success("✅ Authenticated successfully!")
                            return gdrive
                            
            except Exception as e:
                st.error(f"OAuth error: {e}")
    
    return None


def render_gdrive_file_browser(gdrive: GoogleDriveManager):
    """Render a file browser for Google Drive."""
    
    st.markdown("### 📁 Browse Google Drive")
    
    folder_id = st.text_input(
        "Folder ID (optional)",
        placeholder="Leave empty to browse root",
        help="Enter a specific folder ID to browse"
    )
    
    file_type = st.selectbox(
        "File Type",
        options=['csv', 'json', 'all']
    )
    
    if st.button("🔍 List Files"):
        files = gdrive.list_files(
            folder_id=folder_id if folder_id else None,
            file_type=file_type if file_type != 'all' else None
        )
        
        if files:
            st.session_state['gdrive_files'] = files
        else:
            st.info("No files found")
    
    # Display file list
    if 'gdrive_files' in st.session_state:
        files = st.session_state['gdrive_files']
        
        for file in files:
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.text(file['name'])
            
            with col2:
                size = int(file.get('size', 0)) / 1024
                st.text(f"{size:.1f} KB")
            
            with col3:
                if st.button("Load", key=f"load_{file['id']}"):
                    return file['id']
    
    return None
