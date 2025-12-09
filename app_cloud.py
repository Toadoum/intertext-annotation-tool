# """
# Semantic Similarity Team Annotation Tool - Enhanced Version
# With improved Google Sheets debugging and local fallback
# """

# import streamlit as st
# import pandas as pd
# import json
# import os
# import traceback
# from datetime import datetime
# from io import StringIO
# from typing import Optional, Dict, List, Tuple
# import numpy as np

# # Page configuration
# st.set_page_config(
#     page_title="Team Annotation Tool",
#     page_icon="👥",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # =============================================================================
# # CONFIGURATION FROM SECRETS
# # =============================================================================

# def get_config():
#     """Load configuration from Streamlit secrets or defaults."""
#     config = {
#         'team_password': 'annotate2024',
#         'annotators': ['Sakayo', 'Annotator_2', 'Annotator_3', 'Annotator_4', 'Annotator_5'],
#         'spreadsheet_id': None,
#         'gcp_credentials': None,
#         'use_local_fallback': True,
#     }
    
#     # Try to load from secrets
#     try:
#         if 'auth' in st.secrets:
#             config['team_password'] = st.secrets.auth.team_password
        
#         if 'team' in st.secrets:
#             config['annotators'] = list(st.secrets.team.annotators)
        
#         if 'google_sheets' in st.secrets:
#             config['spreadsheet_id'] = st.secrets.google_sheets.spreadsheet_id
        
#         if 'gcp_service_account' in st.secrets:
#             config['gcp_credentials'] = dict(st.secrets.gcp_service_account)
            
#     except Exception as e:
#         st.error(f"Error loading secrets: {e}")
#         config['use_local_fallback'] = True
    
#     return config

# CONFIG = get_config()

# # =============================================================================
# # Custom CSS
# # =============================================================================

# st.markdown("""
# <style>
#     .main-header {
#         font-size: 2.5rem;
#         font-weight: bold;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 1rem;
#     }
#     .sentence-box {
#         background-color: #f0f2f6;
#         padding: 1.5rem;
#         border-radius: 10px;
#         margin: 0.5rem 0;
#         border-left: 4px solid #1f77b4;
#         font-size: 1.1rem;
#     }
#     .progress-bar {
#         background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
#         height: 8px;
#         border-radius: 4px;
#     }
#     .stat-card {
#         background: #f8f9fa;
#         padding: 1rem;
#         border-radius: 10px;
#         text-align: center;
#         border: 1px solid #e9ecef;
#     }
#     .debug-box {
#         background-color: #f8f9fa;
#         border-left: 4px solid #dc3545;
#         padding: 1rem;
#         margin: 1rem 0;
#         font-family: monospace;
#         font-size: 0.9rem;
#         overflow-x: auto;
#     }
# </style>
# """, unsafe_allow_html=True)

# # =============================================================================
# # LOCAL STORAGE (FALLBACK)
# # =============================================================================

# class LocalStorageBackend:
#     """Local storage backend for when Google Sheets fails."""
    
#     def __init__(self):
#         self.data = None
#         self.annotations = []
        
#     def upload_data(self, df: pd.DataFrame) -> bool:
#         """Upload dataset to local storage."""
#         try:
#             self.data = df
#             return True
#         except Exception as e:
#             st.error(f"Local upload failed: {e}")
#             return False
    
#     def get_data(self) -> Optional[pd.DataFrame]:
#         """Get dataset from local storage."""
#         return self.data
    
#     def save_annotation(self, index: int, annotator: str, score: float, notes: str = "") -> bool:
#         """Save annotation to local storage."""
#         try:
#             timestamp = datetime.now().isoformat()
#             self.annotations.append({
#                 'index': int(index),
#                 'annotator': annotator,
#                 'expert_score': float(score),
#                 'notes': notes,
#                 'timestamp': timestamp
#             })
#             return True
#         except Exception as e:
#             st.error(f"Local save failed: {e}")
#             return False
    
#     def get_annotations(self, annotator: str = None) -> pd.DataFrame:
#         """Get annotations from local storage."""
#         if not self.annotations:
#             return pd.DataFrame()
        
#         df = pd.DataFrame(self.annotations)
#         if annotator and not df.empty and 'annotator' in df.columns:
#             df = df[df['annotator'] == annotator]
        
#         return df
    
#     def get_all_annotations(self) -> pd.DataFrame:
#         """Get all annotations."""
#         return self.get_annotations()

# # =============================================================================
# # Google Sheets Backend (with enhanced debugging)
# # =============================================================================

# try:
#     import gspread
#     from google.oauth2.service_account import Credentials
#     from google.auth.exceptions import GoogleAuthError
#     from googleapiclient.errors import HttpError
#     GSHEETS_AVAILABLE = True
# except ImportError:
#     GSHEETS_AVAILABLE = False
#     st.warning("⚠️ gspread not installed. Install with: `pip install gspread google-auth`")


# class GoogleSheetsBackend:
#     """Google Sheets backend for team collaboration."""
    
#     SCOPES = [
#         'https://www.googleapis.com/auth/spreadsheets',
#         'https://www.googleapis.com/auth/drive.file'  # More restrictive scope
#     ]
    
#     def __init__(self):
#         self.client = None
#         self.spreadsheet = None
#         self.data_sheet = None
#         self.annotations_sheet = None
#         self.last_error = None
    
#     def get_connection(self, credentials_dict: Dict, spreadsheet_id: str):
#         """Get connection to Google Sheets."""
#         try:
#             # Debug: Show what credentials we have
#             debug_info = {
#                 'has_private_key': 'private_key' in credentials_dict and bool(credentials_dict['private_key']),
#                 'private_key_length': len(credentials_dict.get('private_key', '')) if 'private_key' in credentials_dict else 0,
#                 'client_email': credentials_dict.get('client_email', 'MISSING'),
#                 'project_id': credentials_dict.get('project_id', 'MISSING')
#             }
            
#             # Fix common private key formatting issues
#             if 'private_key' in credentials_dict:
#                 private_key = credentials_dict['private_key']
#                 # Ensure proper line breaks
#                 if '\\n' in private_key:
#                     credentials_dict['private_key'] = private_key.replace('\\n', '\n')
            
#             credentials = Credentials.from_service_account_info(
#                 credentials_dict,
#                 scopes=self.SCOPES
#             )
            
#             # Test the credentials
#             try:
#                 from google.auth.transport.requests import Request
#                 credentials.refresh(Request())
#             except Exception as refresh_error:
#                 self.last_error = f"Credential refresh failed: {refresh_error}"
#                 return None, None, debug_info
            
#             client = gspread.authorize(credentials)
            
#             # Try to open the spreadsheet
#             try:
#                 spreadsheet = client.open_by_key(spreadsheet_id)
#                 return client, spreadsheet, debug_info
#             except gspread.SpreadsheetNotFound:
#                 self.last_error = f"Spreadsheet not found. Check ID: {spreadsheet_id}"
#                 return None, None, debug_info
#             except Exception as e:
#                 self.last_error = f"Error opening spreadsheet: {e}"
#                 return None, None, debug_info
                
#         except GoogleAuthError as e:
#             self.last_error = f"Authentication failed: {e}"
#             return None, None, {}
#         except Exception as e:
#             self.last_error = f"Connection error: {e}"
#             return None, None, {}
    
#     def connect(self, credentials_dict: Dict, spreadsheet_id: str) -> bool:
#         """Connect to Google Sheets with detailed error reporting."""
#         if not credentials_dict:
#             self.last_error = "No credentials provided"
#             return False
        
#         if not spreadsheet_id:
#             self.last_error = "No spreadsheet ID provided"
#             return False
        
#         self.client, self.spreadsheet, debug_info = self.get_connection(credentials_dict, spreadsheet_id)
        
#         if self.spreadsheet:
#             try:
#                 self._ensure_sheets()
#                 return True
#             except Exception as e:
#                 self.last_error = f"Sheet setup failed: {e}"
#                 return False
#         else:
#             return False
    
#     def _ensure_sheets(self):
#         """Ensure required sheets exist."""
#         try:
#             sheet_names = [ws.title for ws in self.spreadsheet.worksheets()]
            
#             if 'data' not in sheet_names:
#                 self.data_sheet = self.spreadsheet.add_worksheet('data', 1000, 10)
#                 self.data_sheet.update('A1:D1', [['index', 'sentence1', 'sentence2', 'original_score']])
#             else:
#                 self.data_sheet = self.spreadsheet.worksheet('data')
            
#             if 'annotations' not in sheet_names:
#                 self.annotations_sheet = self.spreadsheet.add_worksheet('annotations', 1000, 10)
#                 self.annotations_sheet.update('A1:E1', [
#                     ['index', 'annotator', 'expert_score', 'notes', 'timestamp']
#                 ])
#             else:
#                 self.annotations_sheet = self.spreadsheet.worksheet('annotations')
                
#         except Exception as e:
#             self.last_error = f"Sheet creation failed: {e}"
#             raise
    
#     def upload_data(self, df: pd.DataFrame) -> bool:
#         """Upload dataset."""
#         try:
#             # Clear existing data
#             self.data_sheet.clear()
            
#             # Prepare data
#             data = [['index', 'sentence1', 'sentence2', 'original_score']]
#             for idx, row in df.iterrows():
#                 data.append([
#                     int(idx),
#                     str(row['sentence1'])[:500],  # Limit length
#                     str(row['sentence2'])[:500],
#                     float(row['score'])
#                 ])
            
#             # Upload in chunks to avoid timeout
#             chunk_size = 500
#             for i in range(0, len(data), chunk_size):
#                 chunk = data[i:i+chunk_size]
#                 start_row = i + 1
#                 self.data_sheet.update(f'A{start_row}:D{start_row + len(chunk) - 1}', chunk)
            
#             return True
#         except Exception as e:
#             self.last_error = f"Upload failed: {e}"
#             return False
    
#     def get_data(self) -> Optional[pd.DataFrame]:
#         """Get dataset from Google Sheets."""
#         try:
#             records = self.data_sheet.get_all_records()
#             if records:
#                 return pd.DataFrame(records)
#             return pd.DataFrame()
#         except Exception as e:
#             self.last_error = f"Error loading data: {e}"
#             return None
    
#     def save_annotation(self, index: int, annotator: str, score: float, notes: str = "") -> bool:
#         """Save annotation."""
#         try:
#             timestamp = datetime.now().isoformat()
#             self.annotations_sheet.append_row([
#                 int(index), 
#                 annotator, 
#                 float(score), 
#                 notes[:100],  # Limit note length
#                 timestamp
#             ])
#             return True
#         except Exception as e:
#             self.last_error = f"Save failed: {e}"
#             return False
    
#     def get_annotations(self, annotator: str = None) -> pd.DataFrame:
#         """Get annotations from Google Sheets."""
#         try:
#             records = self.annotations_sheet.get_all_records()
#             df = pd.DataFrame(records) if records else pd.DataFrame()
            
#             if annotator and not df.empty and 'annotator' in df.columns:
#                 df = df[df['annotator'] == annotator]
            
#             return df
#         except Exception as e:
#             self.last_error = f"Get annotations failed: {e}"
#             return pd.DataFrame()
    
#     def get_all_annotations(self) -> pd.DataFrame:
#         """Get all annotations."""
#         return self.get_annotations()

# # =============================================================================
# # HYBRID BACKEND MANAGER
# # =============================================================================

# class BackendManager:
#     """Manage multiple backends with fallback."""
    
#     def __init__(self):
#         self.google_backend = None
#         self.local_backend = None
#         self.active_backend = None
#         self.mode = "local"  # "google" or "local"
        
#     def initialize(self):
#         """Initialize backends."""
#         self.local_backend = LocalStorageBackend()
        
#         # Try Google Sheets if configured
#         if GSHEETS_AVAILABLE and CONFIG['gcp_credentials'] and CONFIG['spreadsheet_id']:
#             self.google_backend = GoogleSheetsBackend()
#             if self.google_backend.connect(CONFIG['gcp_credentials'], CONFIG['spreadsheet_id']):
#                 self.active_backend = self.google_backend
#                 self.mode = "google"
#                 return True
#             else:
#                 st.sidebar.warning(f"⚠️ Google Sheets failed: {self.google_backend.last_error}")
#                 if CONFIG['use_local_fallback']:
#                     st.sidebar.info("Using local storage as fallback")
        
#         # Fallback to local
#         self.active_backend = self.local_backend
#         self.mode = "local"
#         return False
    
#     def get_backend(self):
#         """Get active backend."""
#         return self.active_backend
    
#     def get_mode(self):
#         """Get current mode."""
#         return self.mode
    
#     def switch_to_local(self):
#         """Switch to local backend."""
#         self.active_backend = self.local_backend
#         self.mode = "local"
#         return True
    
#     def switch_to_google(self, credentials_dict: Dict, spreadsheet_id: str):
#         """Switch to Google backend."""
#         if not GSHEETS_AVAILABLE:
#             return False
        
#         self.google_backend = GoogleSheetsBackend()
#         if self.google_backend.connect(credentials_dict, spreadsheet_id):
#             self.active_backend = self.google_backend
#             self.mode = "google"
#             return True
#         return False

# # =============================================================================
# # Session State
# # =============================================================================

# def init_session_state():
#     """Initialize session state."""
#     defaults = {
#         'authenticated': False,
#         'current_user': None,
#         'data': None,
#         'current_index': 0,
#         'backend_manager': None,
#         'filter_mode': 'my_pending',
#         'show_original': False,
#         'show_debug': False,
#     }
#     for key, value in defaults.items():
#         if key not in st.session_state:
#             st.session_state[key] = value
    
#     # Initialize backend manager
#     if st.session_state.backend_manager is None:
#         st.session_state.backend_manager = BackendManager()

# # =============================================================================
# # Authentication
# # =============================================================================

# def render_login():
#     """Render login screen."""
#     st.markdown('<h1 class="main-header">👥 Team Annotation Tool</h1>', unsafe_allow_html=True)
    
#     col1, col2, col3 = st.columns([1, 2, 1])
    
#     with col2:
#         st.markdown("### 🔐 Login")
        
#         annotator = st.selectbox("Your name", [""] + CONFIG['annotators'])
#         password = st.text_input("Team password", type="password")
        
#         if st.button("Login", type="primary", use_container_width=True):
#             if not annotator:
#                 st.error("Select your name")
#             elif password != CONFIG['team_password']:
#                 st.error("Wrong password")
#             else:
#                 st.session_state.authenticated = True
#                 st.session_state.current_user = annotator
                
#                 # Initialize backend
#                 backend_manager = BackendManager()
#                 backend_manager.initialize()
#                 st.session_state.backend_manager = backend_manager
                
#                 st.rerun()

# # =============================================================================
# # Data Management
# # =============================================================================

# def load_data():
#     """Load data from backend."""
#     backend = st.session_state.backend_manager.get_backend()
#     if backend:
#         df = backend.get_data()
#         if df is not None and not df.empty:
#             st.session_state.data = df
#             return True
#     return False

# def get_user_annotations() -> Dict:
#     """Get current user's annotations."""
#     backend = st.session_state.backend_manager.get_backend()
#     if backend:
#         ann_df = backend.get_annotations(st.session_state.current_user)
#         if not ann_df.empty and 'index' in ann_df.columns and 'expert_score' in ann_df.columns:
#             return {int(row['index']): float(row['expert_score']) for _, row in ann_df.iterrows()}
#     return {}

# # =============================================================================
# # Debug Panel
# # =============================================================================

# def render_debug_panel():
#     """Render debug information."""
#     with st.expander("🔧 Debug Information", expanded=st.session_state.show_debug):
#         st.markdown("### Configuration Status")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.markdown("**Library Status**")
#             st.write(f"- gspread available: {GSHEETS_AVAILABLE}")
#             st.write(f"- pandas version: {pd.__version__}")
            
#             backend = st.session_state.backend_manager.get_backend()
#             st.write(f"- Backend type: {type(backend).__name__}")
#             st.write(f"- Backend mode: {st.session_state.backend_manager.get_mode()}")
        
#         with col2:
#             st.markdown("**Data Status**")
#             if st.session_state.data is not None:
#                 st.write(f"- Items loaded: {len(st.session_state.data)}")
#                 st.write(f"- Columns: {list(st.session_state.data.columns)}")
#                 if 'score' in st.session_state.data.columns:
#                     st.write(f"- Score range: {st.session_state.data['score'].min():.2f} to {st.session_state.data['score'].max():.2f}")
#             else:
#                 st.write("- No data loaded")
            
#             ann = get_user_annotations()
#             st.write(f"- My annotations: {len(ann)}")
        
#         st.markdown("---")
#         st.markdown("**Google Sheets Configuration**")
        
#         has_creds = CONFIG['gcp_credentials'] is not None
#         has_sheet_id = CONFIG['spreadsheet_id'] is not None
        
#         st.write(f"- Credentials provided: {has_creds}")
#         st.write(f"- Spreadsheet ID provided: {has_sheet_id}")
        
#         if has_creds:
#             creds = CONFIG['gcp_credentials']
#             st.write(f"- Project ID: {creds.get('project_id', 'MISSING')}")
#             st.write(f"- Client email: {creds.get('client_email', 'MISSING')}")
#             has_key = 'private_key' in creds and bool(creds['private_key'])
#             st.write(f"- Private key present: {has_key}")
#             if has_key:
#                 key_len = len(creds['private_key'])
#                 st.write(f"- Private key length: {key_len} chars")
#                 # Check for common issues
#                 if '\\n' in creds['private_key']:
#                     st.warning("⚠️ Private key contains escaped newlines (\\n instead of actual newlines)")
        
#         st.markdown("---")
        
#         if st.button("🔄 Test Google Sheets Connection"):
#             with st.spinner("Testing connection..."):
#                 test_backend = GoogleSheetsBackend()
#                 if test_backend.connect(CONFIG['gcp_credentials'], CONFIG['spreadsheet_id']):
#                     st.success("✅ Connection successful!")
#                 else:
#                     st.error(f"❌ Connection failed: {test_backend.last_error}")
        
#         if st.button("🔄 Switch to Local Mode"):
#             st.session_state.backend_manager.switch_to_local()
#             st.rerun()

# # =============================================================================
# # Navigation & Annotation UI (same as before with minor adjustments)
# # =============================================================================

# def get_filtered_indices() -> List[int]:
#     """Get indices based on filter."""
#     if st.session_state.data is None:
#         return []
    
#     all_idx = list(range(len(st.session_state.data)))
#     my_ann = get_user_annotations()
    
#     mode = st.session_state.filter_mode
#     if mode == 'my_pending':
#         return [i for i in all_idx if i not in my_ann]
#     elif mode == 'my_done':
#         return [i for i in all_idx if i in my_ann]
#     return all_idx

# def navigate(direction: str):
#     """Navigate items."""
#     filtered = get_filtered_indices()
#     if not filtered:
#         return
    
#     curr = st.session_state.current_index
    
#     if direction == 'next':
#         nxt = [i for i in filtered if i > curr]
#         st.session_state.current_index = nxt[0] if nxt else filtered[0]
#     else:
#         prv = [i for i in filtered if i < curr]
#         st.session_state.current_index = prv[-1] if prv else filtered[-1]

# def save_annotation(idx: int, score: float, notes: str = ""):
#     """Save annotation."""
#     backend = st.session_state.backend_manager.get_backend()
#     if backend:
#         backend.save_annotation(idx, st.session_state.current_user, score, notes)

# def render_annotation_ui():
#     """Render annotation interface."""
#     df = st.session_state.data
#     idx = st.session_state.current_index
    
#     if idx >= len(df):
#         idx = 0
#         st.session_state.current_index = 0
    
#     row = df.iloc[idx]
#     my_ann = get_user_annotations()
#     is_done = idx in my_ann
    
#     # Navigation header
#     c1, c2, c3 = st.columns([1, 3, 1])
    
#     with c1:
#         if st.button("⬅️ Prev", use_container_width=True):
#             navigate('prev')
#             st.rerun()
    
#     with c2:
#         backend_mode = st.session_state.backend_manager.get_mode()
#         mode_icon = "☁️" if backend_mode == "google" else "💻"
#         status = "✅ Done" if is_done else "⏳ Pending"
#         bg = "#d4edda" if is_done else "#fff3cd"
#         st.markdown(
#             f'<div style="text-align:center;padding:0.5rem;background:{bg};border-radius:5px;">'
#             f'<b>#{idx + 1} / {len(df)}</b> | {status} | {mode_icon} {backend_mode}</div>',
#             unsafe_allow_html=True
#         )
    
#     with c3:
#         if st.button("Next ➡️", use_container_width=True):
#             navigate('next')
#             st.rerun()
    
#     st.markdown("---")
    
#     # Sentences
#     c1, c2 = st.columns(2)
    
#     with c1:
#         st.markdown("**Sentence 1**")
#         st.markdown(f'<div class="sentence-box">{row["sentence1"]}</div>', unsafe_allow_html=True)
    
#     with c2:
#         st.markdown("**Sentence 2**")
#         st.markdown(f'<div class="sentence-box">{row["sentence2"]}</div>', unsafe_allow_html=True)
    
#     if st.session_state.show_original:
#         score_val = row.get('original_score', row.get('score', 'N/A'))
#         st.caption(f"Original score: {score_val}")
    
#     st.markdown("---")
    
#     # Annotation
#     st.markdown("**Your Score** (0 = unrelated → 1 = identical)")
    
#     current_val = my_ann.get(idx, 0.5)
#     score = st.slider("Score", 0.0, 1.0, float(current_val), 0.01, key=f"sl_{idx}", label_visibility="collapsed")
    
#     # Quick buttons
#     cols = st.columns(6)
#     for i, v in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
#         with cols[i]:
#             if st.button(str(v), key=f"q{v}_{idx}", use_container_width=True):
#                 save_annotation(idx, v)
#                 navigate('next')
#                 st.rerun()
    
#     notes = st.text_input("Notes (optional)", key=f"n_{idx}")
    
#     # Actions
#     c1, c2, c3 = st.columns(3)
    
#     with c1:
#         if st.button("💾 Save", use_container_width=True):
#             save_annotation(idx, score, notes)
#             st.success("Saved!")
    
#     with c2:
#         if st.button("✅ Save & Next", type="primary", use_container_width=True):
#             save_annotation(idx, score, notes)
#             navigate('next')
#             st.rerun()
    
#     with c3:
#         if st.button("⏭️ Skip", use_container_width=True):
#             navigate('next')
#             st.rerun()

# # =============================================================================
# # Dashboard & Export (same as before)
# # =============================================================================

# def render_dashboard():
#     """Render team dashboard."""
#     st.markdown("### 📊 Team Progress")
    
#     backend = st.session_state.backend_manager.get_backend()
    
#     if backend is None:
#         st.info("No backend connected")
#         return
    
#     all_ann = backend.get_all_annotations()
#     total = len(st.session_state.data) if st.session_state.data is not None else 0
    
#     if all_ann.empty:
#         st.info("No annotations yet")
#         return
    
#     # Stats
#     c1, c2, c3, c4 = st.columns(4)
    
#     with c1:
#         st.metric("Total Items", total)
#     with c2:
#         st.metric("Annotated", all_ann['index'].nunique())
#     with c3:
#         st.metric("Annotations", len(all_ann))
#     with c4:
#         st.metric("Annotators", all_ann['annotator'].nunique())
    
#     # Per-annotator
#     st.markdown("#### By Annotator")
#     stats = all_ann.groupby('annotator').agg(
#         count=('index', 'count'),
#         avg_score=('expert_score', 'mean')
#     ).round(2)
#     st.dataframe(stats, use_container_width=True)
    
#     # Agreement
#     if 'index' in all_ann.columns:
#         multi = all_ann.groupby('index').filter(lambda x: len(x) >= 2)
#         if not multi.empty:
#             st.markdown("#### Inter-Annotator Agreement")
            
#             agreements = []
#             for idx, grp in multi.groupby('index'):
#                 scores = grp['expert_score'].values[:2]
#                 if len(scores) >= 2:
#                     agreements.append(1 - abs(scores[0] - scores[1]))
            
#             if agreements:
#                 mean_agr = np.mean(agreements)
#                 st.metric("Mean Agreement", f"{mean_agr:.1%}")
#                 st.caption(f"Based on {len(agreements)} items with 2+ annotations")

# def render_export():
#     """Render export section."""
#     st.markdown("### 💾 Export")
    
#     if st.session_state.data is None:
#         st.info("No data loaded")
#         return
    
#     df = st.session_state.data.copy()
    
#     # Get all annotations
#     backend = st.session_state.backend_manager.get_backend()
#     if backend:
#         all_ann = backend.get_all_annotations()
#     else:
#         all_ann = pd.DataFrame()
    
#     if all_ann.empty:
#         st.warning("No annotations to export")
#         return
    
#     # Pivot
#     if 'index' in all_ann.columns and 'annotator' in all_ann.columns:
#         pivot = all_ann.pivot_table(index='index', columns='annotator', values='expert_score', aggfunc='first')
#         df = df.merge(pivot, left_index=True, right_index=True, how='left')
        
#         # Consensus
#         ann_cols = [c for c in pivot.columns]
#         if ann_cols:
#             df['expert_consensus'] = df[ann_cols].mean(axis=1)
    
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
#     c1, c2 = st.columns(2)
    
#     with c1:
#         csv_buf = StringIO()
#         df.to_csv(csv_buf, index=False)
#         st.download_button(
#             "📥 CSV (Full)",
#             csv_buf.getvalue(),
#             f"annotations_{timestamp}.csv",
#             "text/csv",
#             use_container_width=True
#         )
    
#     with c2:
#         # Instruction format
#         json_data = []
#         score_col = 'original_score' if 'original_score' in df.columns else 'score'
        
#         if 'expert_consensus' in df.columns:
#             for idx, row in df.iterrows():
#                 if pd.notna(row.get('expert_consensus')):
#                     json_data.append({
#                         "instruction": f"Output a number between 0 and 1 describing the semantic similarity between the following two sentences:\nSentence 1: {row['sentence1']}\nSentence 2: {row['sentence2']}",
#                         "input": "",
#                         "output": str(row.get(score_col, '')),
#                         "expert": str(round(row['expert_consensus'], 2))
#                     })
        
#         if json_data:
#             st.download_button(
#                 "📥 JSON (Instruction)",
#                 json.dumps(json_data, indent=2, ensure_ascii=False),
#                 f"annotations_{timestamp}.json",
#                 "application/json",
#                 use_container_width=True
#             )

# def render_upload():
#     """Render upload section."""
#     st.markdown("### 📤 Upload Dataset")
    
#     uploaded = st.file_uploader("CSV file", type=['csv'])
    
#     if uploaded:
#         df = pd.read_csv(uploaded)
        
#         required = ['sentence1', 'sentence2', 'score']
#         if not all(c in df.columns for c in required):
#             st.error(f"Need columns: {required}")
#             return
        
#         st.success(f"Loaded {len(df)} items")
#         st.dataframe(df.head())
        
#         if st.button("📤 Upload to Storage", type="primary"):
#             backend = st.session_state.backend_manager.get_backend()
#             if backend:
#                 with st.spinner("Uploading..."):
#                     if backend.upload_data(df):
#                         st.success("Uploaded!")
#                         st.session_state.data = df
#                         st.rerun()
#             else:
#                 st.session_state.data = df
#                 st.rerun()

# # =============================================================================
# # Sidebar
# # =============================================================================

# def render_sidebar():
#     """Render sidebar."""
#     with st.sidebar:
#         st.markdown(f"### 👤 {st.session_state.current_user}")
        
#         # Backend status
#         backend_mode = st.session_state.backend_manager.get_mode()
#         if backend_mode == "google":
#             st.success("☁️ Google Sheets Connected")
#         else:
#             st.info("💻 Local Storage Mode")
        
#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("🔄 Refresh"):
#                 load_data()
#                 st.rerun()
#         with col2:
#             if st.button("Logout"):
#                 st.session_state.authenticated = False
#                 st.session_state.data = None
#                 st.rerun()
        
#         st.markdown("---")
        
#         # Progress
#         if st.session_state.data is not None:
#             my_ann = get_user_annotations()
#             total = len(st.session_state.data)
#             done = len(my_ann)
#             pct = done / total if total else 0
            
#             st.markdown("**My Progress**")
#             st.progress(pct)
#             st.caption(f"{done} / {total} ({pct:.1%})")
            
#             pending = [i for i in range(total) if i not in my_ann]
#             if pending:
#                 if st.button(f"▶️ Next pending (#{pending[0]+1})"):
#                     st.session_state.current_index = pending[0]
#                     st.rerun()
        
#         st.markdown("---")
        
#         # Filter
#         st.session_state.filter_mode = st.selectbox(
#             "Show",
#             ['my_pending', 'my_done', 'all'],
#             format_func=lambda x: {'my_pending': '⏳ Pending', 'my_done': '✅ Done', 'all': '📋 All'}[x]
#         )
        
#         st.session_state.show_original = st.checkbox("Show original scores")
#         st.session_state.show_debug = st.checkbox("Show debug panel")
        
#         st.markdown("---")
        
#         # Jump
#         if st.session_state.data is not None:
#             jump = st.number_input("Jump to #", 1, len(st.session_state.data), st.session_state.current_index + 1)
#             if st.button("Go"):
#                 st.session_state.current_index = jump - 1
#                 st.rerun()

# # =============================================================================
# # Main
# # =============================================================================

# def main():
#     """Main entry point."""
#     init_session_state()
    
#     # Login
#     if not st.session_state.authenticated:
#         render_login()
#         return
    
#     # Load data if not loaded
#     if st.session_state.data is None:
#         with st.spinner("Loading data..."):
#             load_data()
    
#     # Render
#     render_sidebar()
    
#     st.markdown('<h1 class="main-header">👥 Team Annotation Tool</h1>', unsafe_allow_html=True)
    
#     # Show debug panel if enabled
#     if st.session_state.show_debug:
#         render_debug_panel()
    
#     tabs = st.tabs(["✏️ Annotate", "📊 Dashboard", "📤 Upload", "💾 Export"])
    
#     with tabs[0]:
#         if st.session_state.data is not None:
#             render_annotation_ui()
#         else:
#             st.info("📭 No data loaded yet.")
#             st.markdown("""
#             **To get started:**
#             1. Go to **Upload** tab and upload your CSV file, OR
#             2. Click **🔄 Refresh** in the sidebar if data was already uploaded
            
#             Your CSV should have columns: `sentence1`, `sentence2`, `score`
#             """)
    
#     with tabs[1]:
#         render_dashboard()
    
#     with tabs[2]:
#         render_upload()
    
#     with tabs[3]:
#         render_export()

# if __name__ == "__main__":
#     main()


"""
Semantic Similarity Team Annotation Tool - French Version
Interface en français avec explication des scores
"""

import streamlit as st
import pandas as pd
import json
import os
import traceback
from datetime import datetime
from io import StringIO
from typing import Optional, Dict, List, Tuple
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Outil d'Annotation d'Équipe",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CONFIGURATION FROM SECRETS
# =============================================================================

def get_config():
    """Load configuration from Streamlit secrets or defaults."""
    config = {
        'team_password': 'annotate2024',
        'annotators': ['Sakayo', 'Annotateur_2', 'Annotateur_3', 'Annotateur_4', 'Annotateur_5'],
        'spreadsheet_id': None,
        'gcp_credentials': None,
        'use_local_fallback': True,
    }
    
    # Try to load from secrets
    try:
        if 'auth' in st.secrets:
            config['team_password'] = st.secrets.auth.team_password
        
        if 'team' in st.secrets:
            config['annotators'] = list(st.secrets.team.annotators)
        
        if 'google_sheets' in st.secrets:
            config['spreadsheet_id'] = st.secrets.google_sheets.spreadsheet_id
        
        if 'gcp_service_account' in st.secrets:
            config['gcp_credentials'] = dict(st.secrets.gcp_service_account)
            
    except Exception as e:
        st.error(f"Erreur de chargement des secrets: {e}")
        config['use_local_fallback'] = True
    
    return config

CONFIG = get_config()

# =============================================================================
# Custom CSS
# =============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sentence-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
        font-size: 1.1rem;
    }
    .progress-bar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 8px;
        border-radius: 4px;
    }
    .stat-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .score-explanation {
        background-color: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOCAL STORAGE (FALLBACK)
# =============================================================================

class LocalStorageBackend:
    """Local storage backend for when Google Sheets fails."""
    
    def __init__(self):
        self.data = None
        self.annotations = []
        
    def upload_data(self, df: pd.DataFrame) -> bool:
        """Upload dataset to local storage."""
        try:
            self.data = df
            return True
        except Exception as e:
            st.error(f"Échec du téléversement local: {e}")
            return False
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """Get dataset from local storage."""
        return self.data
    
    def save_annotation(self, index: int, annotator: str, score: float, notes: str = "") -> bool:
        """Save annotation to local storage."""
        try:
            timestamp = datetime.now().isoformat()
            self.annotations.append({
                'index': int(index),
                'annotator': annotator,
                'expert_score': float(score),
                'notes': notes,
                'timestamp': timestamp
            })
            return True
        except Exception as e:
            st.error(f"Échec de la sauvegarde locale: {e}")
            return False
    
    def get_annotations(self, annotator: str = None) -> pd.DataFrame:
        """Get annotations from local storage."""
        if not self.annotations:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.annotations)
        if annotator and not df.empty and 'annotator' in df.columns:
            df = df[df['annotator'] == annotator]
        
        return df
    
    def get_all_annotations(self) -> pd.DataFrame:
        """Get all annotations."""
        return self.get_annotations()

# =============================================================================
# Google Sheets Backend (with enhanced debugging)
# =============================================================================

try:
    import gspread
    from google.oauth2.service_account import Credentials
    from google.auth.exceptions import GoogleAuthError
    from googleapiclient.errors import HttpError
    GSHEETS_AVAILABLE = True
except ImportError:
    GSHEETS_AVAILABLE = False
    st.warning("⚠️ gspread non installé. Installez avec: `pip install gspread google-auth`")

class GoogleSheetsBackend:
    """Google Sheets backend for team collaboration."""
    
    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive.file'
    ]
    
    def __init__(self):
        self.client = None
        self.spreadsheet = None
        self.data_sheet = None
        self.annotations_sheet = None
        self.last_error = None
    
    def get_connection(self, credentials_dict: Dict, spreadsheet_id: str):
        """Get connection to Google Sheets."""
        try:
            if 'private_key' in credentials_dict:
                private_key = credentials_dict['private_key']
                if '\\n' in private_key:
                    credentials_dict['private_key'] = private_key.replace('\\n', '\n')
            
            credentials = Credentials.from_service_account_info(
                credentials_dict,
                scopes=self.SCOPES
            )
            
            client = gspread.authorize(credentials)
            spreadsheet = client.open_by_key(spreadsheet_id)
            return client, spreadsheet
                
        except GoogleAuthError as e:
            self.last_error = f"Échec d'authentification: {e}"
            return None, None
        except Exception as e:
            self.last_error = f"Erreur de connexion: {e}"
            return None, None
    
    def connect(self, credentials_dict: Dict, spreadsheet_id: str) -> bool:
        """Connect to Google Sheets."""
        if not credentials_dict:
            self.last_error = "Aucune information d'identification fournie"
            return False
        
        if not spreadsheet_id:
            self.last_error = "Aucun ID de spreadsheet fourni"
            return False
        
        self.client, self.spreadsheet = self.get_connection(credentials_dict, spreadsheet_id)
        
        if self.spreadsheet:
            try:
                self._ensure_sheets()
                return True
            except Exception as e:
                self.last_error = f"Échec de la configuration des feuilles: {e}"
                return False
        else:
            return False
    
    def _ensure_sheets(self):
        """Ensure required sheets exist."""
        try:
            sheet_names = [ws.title for ws in self.spreadsheet.worksheets()]
            
            if 'data' not in sheet_names:
                self.data_sheet = self.spreadsheet.add_worksheet('data', 1000, 10)
                self.data_sheet.update('A1:D1', [['index', 'sentence1', 'sentence2', 'original_score']])
            else:
                self.data_sheet = self.spreadsheet.worksheet('data')
            
            if 'annotations' not in sheet_names:
                self.annotations_sheet = self.spreadsheet.add_worksheet('annotations', 1000, 10)
                self.annotations_sheet.update('A1:E1', [
                    ['index', 'annotator', 'expert_score', 'notes', 'timestamp']
                ])
            else:
                self.annotations_sheet = self.spreadsheet.worksheet('annotations')
                
        except Exception as e:
            self.last_error = f"Échec de la création des feuilles: {e}"
            raise
    
    def upload_data(self, df: pd.DataFrame) -> bool:
        """Upload dataset."""
        try:
            self.data_sheet.clear()
            data = [['index', 'sentence1', 'sentence2', 'original_score']]
            for idx, row in df.iterrows():
                data.append([
                    int(idx),
                    str(row['sentence1'])[:500],
                    str(row['sentence2'])[:500],
                    float(row['score'])
                ])
            
            chunk_size = 500
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i+chunk_size]
                start_row = i + 1
                self.data_sheet.update(f'A{start_row}:D{start_row + len(chunk) - 1}', chunk)
            
            return True
        except Exception as e:
            self.last_error = f"Échec du téléversement: {e}"
            return False
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """Get dataset from Google Sheets."""
        try:
            records = self.data_sheet.get_all_records()
            if records:
                return pd.DataFrame(records)
            return pd.DataFrame()
        except Exception as e:
            self.last_error = f"Erreur de chargement des données: {e}"
            return None
    
    def save_annotation(self, index: int, annotator: str, score: float, notes: str = "") -> bool:
        """Save annotation."""
        try:
            timestamp = datetime.now().isoformat()
            self.annotations_sheet.append_row([
                int(index), 
                annotator, 
                float(score), 
                notes[:100],
                timestamp
            ])
            return True
        except Exception as e:
            self.last_error = f"Échec de la sauvegarde: {e}"
            return False
    
    def get_annotations(self, annotator: str = None) -> pd.DataFrame:
        """Get annotations from Google Sheets."""
        try:
            records = self.annotations_sheet.get_all_records()
            df = pd.DataFrame(records) if records else pd.DataFrame()
            
            if annotator and not df.empty and 'annotator' in df.columns:
                df = df[df['annotator'] == annotator]
            
            return df
        except Exception as e:
            self.last_error = f"Échec de récupération des annotations: {e}"
            return pd.DataFrame()
    
    def get_all_annotations(self) -> pd.DataFrame:
        """Get all annotations."""
        return self.get_annotations()

# =============================================================================
# HYBRID BACKEND MANAGER
# =============================================================================

class BackendManager:
    """Manage multiple backends with fallback."""
    
    def __init__(self):
        self.google_backend = None
        self.local_backend = None
        self.active_backend = None
        self.mode = "local"
        
    def initialize(self):
        """Initialize backends."""
        self.local_backend = LocalStorageBackend()
        
        if GSHEETS_AVAILABLE and CONFIG['gcp_credentials'] and CONFIG['spreadsheet_id']:
            self.google_backend = GoogleSheetsBackend()
            if self.google_backend.connect(CONFIG['gcp_credentials'], CONFIG['spreadsheet_id']):
                self.active_backend = self.google_backend
                self.mode = "google"
                return True
            else:
                st.sidebar.warning(f"⚠️ Google Sheets échoué: {self.google_backend.last_error}")
                if CONFIG['use_local_fallback']:
                    st.sidebar.info("Utilisation du stockage local comme solution de secours")
        
        self.active_backend = self.local_backend
        self.mode = "local"
        return False
    
    def get_backend(self):
        """Get active backend."""
        return self.active_backend
    
    def get_mode(self):
        """Get current mode."""
        return self.mode

# =============================================================================
# Session State
# =============================================================================

def init_session_state():
    """Initialize session state."""
    defaults = {
        'authenticated': False,
        'current_user': None,
        'data': None,
        'current_index': 0,
        'backend_manager': None,
        'filter_mode': 'my_pending',
        'show_original': True,  # Show original scores by default
        'show_debug': False,    # Hide debug panel by default
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    if st.session_state.backend_manager is None:
        st.session_state.backend_manager = BackendManager()

# =============================================================================
# Authentication
# =============================================================================

def render_login():
    """Render login screen."""
    st.markdown('<h1 class="main-header">👥 Outil d\'Annotation d\'Équipe</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 Connexion")
        
        annotator = st.selectbox("Votre nom", [""] + CONFIG['annotators'])
        password = st.text_input("Mot de passe d'équipe", type="password")
        
        if st.button("Connexion", type="primary", use_container_width=True):
            if not annotator:
                st.error("Sélectionnez votre nom")
            elif password != CONFIG['team_password']:
                st.error("Mot de passe incorrect")
            else:
                st.session_state.authenticated = True
                st.session_state.current_user = annotator
                
                backend_manager = BackendManager()
                backend_manager.initialize()
                st.session_state.backend_manager = backend_manager
                
                st.rerun()

# =============================================================================
# Data Management
# =============================================================================

def load_data():
    """Load data from backend."""
    backend = st.session_state.backend_manager.get_backend()
    if backend:
        df = backend.get_data()
        if df is not None and not df.empty:
            st.session_state.data = df
            return True
    return False

def get_user_annotations() -> Dict:
    """Get current user's annotations."""
    backend = st.session_state.backend_manager.get_backend()
    if backend:
        ann_df = backend.get_annotations(st.session_state.current_user)
        if not ann_df.empty and 'index' in ann_df.columns and 'expert_score' in ann_df.columns:
            return {int(row['index']): float(row['expert_score']) for _, row in ann_df.iterrows()}
    return {}

# =============================================================================
# Navigation & Annotation UI
# =============================================================================

def get_filtered_indices() -> List[int]:
    """Get indices based on filter."""
    if st.session_state.data is None:
        return []
    
    all_idx = list(range(len(st.session_state.data)))
    my_ann = get_user_annotations()
    
    mode = st.session_state.filter_mode
    if mode == 'my_pending':
        return [i for i in all_idx if i not in my_ann]
    elif mode == 'my_done':
        return [i for i in all_idx if i in my_ann]
    return all_idx

def navigate(direction: str):
    """Navigate items."""
    filtered = get_filtered_indices()
    if not filtered:
        return
    
    curr = st.session_state.current_index
    
    if direction == 'next':
        nxt = [i for i in filtered if i > curr]
        st.session_state.current_index = nxt[0] if nxt else filtered[0]
    else:
        prv = [i for i in filtered if i < curr]
        st.session_state.current_index = prv[-1] if prv else filtered[-1]

def save_annotation(idx: int, score: float, notes: str = ""):
    """Save annotation."""
    backend = st.session_state.backend_manager.get_backend()
    if backend:
        backend.save_annotation(idx, st.session_state.current_user, score, notes)

def render_annotation_ui():
    """Render annotation interface."""
    df = st.session_state.data
    idx = st.session_state.current_index
    
    if idx >= len(df):
        idx = 0
        st.session_state.current_index = 0
    
    row = df.iloc[idx]
    my_ann = get_user_annotations()
    is_done = idx in my_ann
    
    # Navigation header
    c1, c2, c3 = st.columns([1, 3, 1])
    
    with c1:
        if st.button("⬅️ Précédent", use_container_width=True):
            navigate('prev')
            st.rerun()
    
    with c2:
        backend_mode = st.session_state.backend_manager.get_mode()
        mode_icon = "☁️" if backend_mode == "google" else "💻"
        status = "✅ Terminé" if is_done else "⏳ En attente"
        bg = "#d4edda" if is_done else "#fff3cd"
        st.markdown(
            f'<div style="text-align:center;padding:0.5rem;background:{bg};border-radius:5px;">'
            f'<b>#{idx + 1} / {len(df)}</b> | {status} | {mode_icon} {backend_mode}</div>',
            unsafe_allow_html=True
        )
    
    with c3:
        if st.button("Suivant ➡️", use_container_width=True):
            navigate('next')
            st.rerun()
    
    st.markdown("---")
    
    # Sentences
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("**Phrase 1**")
        st.markdown(f'<div class="sentence-box">{row["sentence1"]}</div>', unsafe_allow_html=True)
    
    with c2:
        st.markdown("**Phrase 2**")
        st.markdown(f'<div class="sentence-box">{row["sentence2"]}</div>', unsafe_allow_html=True)
    
    # Show original score with French label
    if st.session_state.show_original:
        score_val = row.get('original_score', row.get('score', 'N/A'))
        st.caption(f"**Score original:** {score_val}")
    
    st.markdown("---")
    
    # Score explanation box in French
    st.markdown('<div class="score-explanation">'
                '<b>📊 Échelle de notation:</b>'
                '<br>0.0 = Phrases complètement différentes (sans rapport)'
                '<br>1.0 = Identique en sens (même signification)'
                '<br><small>Utilisez les valeurs intermédiaires pour exprimer différents niveaux de similarité.</small>'
                '</div>', unsafe_allow_html=True)
    
    # Annotation
    st.markdown("**Votre Note** (0 = sans rapport → 1 = identique)")
    
    current_val = my_ann.get(idx, 0.5)
    score = st.slider("Note", 0.0, 1.0, float(current_val), 0.01, 
                     key=f"sl_{idx}", label_visibility="collapsed",
                     help="0.0 = sans rapport, 1.0 = identique en sens")
    
    # Quick buttons with French labels
    cols = st.columns(6)
    button_labels = ["0.0\nSans rapport", "0.2", "0.4", "0.6", "0.8", "1.0\nIdentique"]
    
    for i, (v, label) in enumerate(zip([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], button_labels)):
        with cols[i]:
            if st.button(label, key=f"q{v}_{idx}", use_container_width=True):
                save_annotation(idx, v)
                navigate('next')
                st.rerun()
    
    notes = st.text_input("Notes (optionnel)", key=f"n_{idx}")
    
    # Actions in French
    c1, c2, c3 = st.columns(3)
    
    with c1:
        if st.button("💾 Sauvegarder", use_container_width=True):
            save_annotation(idx, score, notes)
            st.success("Sauvegardé!")
    
    with c2:
        if st.button("✅ Sauvegarder & Suivant", type="primary", use_container_width=True):
            save_annotation(idx, score, notes)
            navigate('next')
            st.rerun()
    
    with c3:
        if st.button("⏭️ Passer", use_container_width=True):
            navigate('next')
            st.rerun()

# =============================================================================
# Dashboard & Export
# =============================================================================

def render_dashboard():
    """Render team dashboard."""
    st.markdown("### 📊 Progression de l'Équipe")
    
    backend = st.session_state.backend_manager.get_backend()
    
    if backend is None:
        st.info("Aucun backend connecté")
        return
    
    all_ann = backend.get_all_annotations()
    total = len(st.session_state.data) if st.session_state.data is not None else 0
    
    if all_ann.empty:
        st.info("Aucune annotation pour le moment")
        return
    
    # Stats in French
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.metric("Total des items", total)
    with c2:
        st.metric("Items annotés", all_ann['index'].nunique())
    with c3:
        st.metric("Annotations", len(all_ann))
    with c4:
        st.metric("Annotateurs", all_ann['annotator'].nunique())
    
    # Per-annotator in French
    st.markdown("#### Par Annotateur")
    stats = all_ann.groupby('annotator').agg(
        count=('index', 'count'),
        note_moyenne=('expert_score', 'mean')
    ).round(2)
    stats = stats.rename(columns={'count': 'Nombre', 'note_moyenne': 'Moyenne'})
    st.dataframe(stats, use_container_width=True)
    
    # Agreement
    if 'index' in all_ann.columns:
        multi = all_ann.groupby('index').filter(lambda x: len(x) >= 2)
        if not multi.empty:
            st.markdown("#### Accord Inter-Annotateurs")
            
            agreements = []
            for idx, grp in multi.groupby('index'):
                scores = grp['expert_score'].values[:2]
                if len(scores) >= 2:
                    agreements.append(1 - abs(scores[0] - scores[1]))
            
            if agreements:
                mean_agr = np.mean(agreements)
                st.metric("Accord moyen", f"{mean_agr:.1%}")
                st.caption(f"Basé sur {len(agreements)} items avec 2+ annotations")

def render_export():
    """Render export section."""
    st.markdown("### 💾 Exporter")
    
    if st.session_state.data is None:
        st.info("Aucune donnée chargée")
        return
    
    df = st.session_state.data.copy()
    
    backend = st.session_state.backend_manager.get_backend()
    if backend:
        all_ann = backend.get_all_annotations()
    else:
        all_ann = pd.DataFrame()
    
    if all_ann.empty:
        st.warning("Aucune annotation à exporter")
        return
    
    if 'index' in all_ann.columns and 'annotator' in all_ann.columns:
        pivot = all_ann.pivot_table(index='index', columns='annotator', values='expert_score', aggfunc='first')
        df = df.merge(pivot, left_index=True, right_index=True, how='left')
        
        ann_cols = [c for c in pivot.columns]
        if ann_cols:
            df['expert_consensus'] = df[ann_cols].mean(axis=1)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    c1, c2 = st.columns(2)
    
    with c1:
        csv_buf = StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            "📥 CSV (Complet)",
            csv_buf.getvalue(),
            f"annotations_{timestamp}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with c2:
        json_data = []
        score_col = 'original_score' if 'original_score' in df.columns else 'score'
        
        if 'expert_consensus' in df.columns:
            for idx, row in df.iterrows():
                if pd.notna(row.get('expert_consensus')):
                    json_data.append({
                        "instruction": f"Output a number between 0 and 1 describing the semantic similarity between the following two sentences:\nSentence 1: {row['sentence1']}\nSentence 2: {row['sentence2']}",
                        "input": "",
                        "output": str(row.get(score_col, '')),
                        "expert": str(round(row['expert_consensus'], 2))
                    })
        
        if json_data:
            st.download_button(
                "📥 JSON (Instruction)",
                json.dumps(json_data, indent=2, ensure_ascii=False),
                f"annotations_{timestamp}.json",
                "application/json",
                use_container_width=True
            )

def render_upload():
    """Render upload section."""
    st.markdown("### 📤 Téléverser un Dataset")
    
    uploaded = st.file_uploader("Fichier CSV", type=['csv'])
    
    if uploaded:
        df = pd.read_csv(uploaded)
        
        required = ['sentence1', 'sentence2', 'score']
        if not all(c in df.columns for c in required):
            st.error(f"Colonnes requises: {required}")
            return
        
        st.success(f"Chargé {len(df)} items")
        st.dataframe(df.head(), use_container_width=True)
        
        if st.button("📤 Téléverser vers le stockage", type="primary"):
            backend = st.session_state.backend_manager.get_backend()
            if backend:
                with st.spinner("Téléversement..."):
                    if backend.upload_data(df):
                        st.success("Téléversé!")
                        st.session_state.data = df
                        st.rerun()
            else:
                st.session_state.data = df
                st.rerun()

# =============================================================================
# Sidebar
# =============================================================================

def render_sidebar():
    """Render sidebar."""
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.current_user}")
        
        backend_mode = st.session_state.backend_manager.get_mode()
        if backend_mode == "google":
            st.success("☁️ Google Sheets Connecté")
        else:
            st.info("💻 Mode Stockage Local")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Rafraîchir"):
                load_data()
                st.rerun()
        with col2:
            if st.button("Déconnexion"):
                st.session_state.authenticated = False
                st.session_state.data = None
                st.rerun()
        
        st.markdown("---")
        
        # Progress in French
        if st.session_state.data is not None:
            my_ann = get_user_annotations()
            total = len(st.session_state.data)
            done = len(my_ann)
            pct = done / total if total else 0
            
            st.markdown("**Ma Progression**")
            st.progress(pct)
            st.caption(f"{done} / {total} ({pct:.1%})")
            
            pending = [i for i in range(total) if i not in my_ann]
            if pending:
                if st.button(f"▶️ Prochain en attente (#{pending[0]+1})"):
                    st.session_state.current_index = pending[0]
                    st.rerun()
        
        st.markdown("---")
        
        # Filter in French
        st.session_state.filter_mode = st.selectbox(
            "Afficher",
            ['my_pending', 'my_done', 'all'],
            format_func=lambda x: {
                'my_pending': '⏳ En attente', 
                'my_done': '✅ Terminé', 
                'all': '📋 Tous'
            }[x]
        )
        
        st.session_state.show_original = st.checkbox("Afficher les scores originaux", value=True)
        
        st.markdown("---")
        
        # Jump in French
        if st.session_state.data is not None:
            jump = st.number_input("Aller à #", 1, len(st.session_state.data), st.session_state.current_index + 1)
            if st.button("Aller"):
                st.session_state.current_index = jump - 1
                st.rerun()

# =============================================================================
# Main
# =============================================================================

def main():
    """Main entry point."""
    init_session_state()
    
    # Login
    if not st.session_state.authenticated:
        render_login()
        return
    
    # Load data if not loaded
    if st.session_state.data is None:
        with st.spinner("Chargement des données..."):
            load_data()
    
    # Render
    render_sidebar()
    
    st.markdown('<h1 class="main-header">👥 Outil d\'Annotation d\'Équipe</h1>', unsafe_allow_html=True)
    
    tabs = st.tabs(["✏️ Annoter", "📊 Tableau de bord", "📤 Téléverser", "💾 Exporter"])
    
    with tabs[0]:
        if st.session_state.data is not None:
            render_annotation_ui()
        else:
            st.info("📭 Aucune donnée chargée pour le moment.")
            st.markdown("""
            **Pour commencer:**
            1. Allez à l'onglet **Téléverser** et téléversez votre fichier CSV, OU
            2. Cliquez sur **🔄 Rafraîchir** dans la barre latérale si les données ont déjà été téléversées
            
            Votre CSV doit avoir les colonnes: `sentence1`, `sentence2`, `score`
            """)
    
    with tabs[1]:
        render_dashboard()
    
    with tabs[2]:
        render_upload()
    
    with tabs[3]:
        render_export()

if __name__ == "__main__":
    main()
