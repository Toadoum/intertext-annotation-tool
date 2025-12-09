"""
Semantic Similarity Team Annotation Tool
A Streamlit application for collaborative annotation with shared storage.
"""

import streamlit as st
import pandas as pd
import json
import os
import hashlib
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Team Annotation Tool",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CONFIGURATION - Edit these for your team
# =============================================================================

# Simple password protection (change this!)
TEAM_PASSWORD = "annotate2024"

# Annotator list - add your team members
ANNOTATORS = [
    "Sakayo",
    "Annotator_2",
    "Annotator_3",
    "Annotator_4",
    "Annotator_5",
]

# Annotation settings
ANNOTATIONS_PER_ITEM = 2  # How many annotators per item for agreement calculation
ITEMS_PER_BATCH = 100     # Items per annotation batch

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
    }
    .annotator-badge {
        background-color: #1f77b4;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
    }
    .agreement-high { color: #28a745; font-weight: bold; }
    .agreement-medium { color: #ffc107; font-weight: bold; }
    .agreement-low { color: #dc3545; font-weight: bold; }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Google Sheets Backend
# =============================================================================

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSHEETS_AVAILABLE = True
except ImportError:
    GSHEETS_AVAILABLE = False


class GoogleSheetsBackend:
    """Manages annotation storage in Google Sheets for team collaboration."""
    
    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    
    def __init__(self):
        self.client = None
        self.spreadsheet = None
        self.data_sheet = None
        self.annotations_sheet = None
        self.assignments_sheet = None
    
    def authenticate(self, credentials_dict: Dict) -> bool:
        """Authenticate with Google Sheets API."""
        try:
            credentials = Credentials.from_service_account_info(
                credentials_dict,
                scopes=self.SCOPES
            )
            self.client = gspread.authorize(credentials)
            return True
        except Exception as e:
            st.error(f"Authentication failed: {e}")
            return False
    
    def open_or_create_spreadsheet(self, spreadsheet_id: str = None, name: str = None) -> bool:
        """Open existing or create new spreadsheet."""
        try:
            if spreadsheet_id:
                self.spreadsheet = self.client.open_by_key(spreadsheet_id)
            elif name:
                try:
                    self.spreadsheet = self.client.open(name)
                except gspread.SpreadsheetNotFound:
                    self.spreadsheet = self.client.create(name)
            
            # Ensure required sheets exist
            self._ensure_sheets()
            return True
        except Exception as e:
            st.error(f"Error opening spreadsheet: {e}")
            return False
    
    def _ensure_sheets(self):
        """Create required sheets if they don't exist."""
        sheet_names = [ws.title for ws in self.spreadsheet.worksheets()]
        
        # Data sheet
        if 'data' not in sheet_names:
            self.data_sheet = self.spreadsheet.add_worksheet('data', 1000, 10)
            self.data_sheet.update('A1:D1', [['index', 'sentence1', 'sentence2', 'original_score']])
        else:
            self.data_sheet = self.spreadsheet.worksheet('data')
        
        # Annotations sheet
        if 'annotations' not in sheet_names:
            self.annotations_sheet = self.spreadsheet.add_worksheet('annotations', 10000, 10)
            self.annotations_sheet.update('A1:F1', [
                ['index', 'annotator', 'expert_score', 'notes', 'timestamp', 'batch_id']
            ])
        else:
            self.annotations_sheet = self.spreadsheet.worksheet('annotations')
        
        # Assignments sheet
        if 'assignments' not in sheet_names:
            self.assignments_sheet = self.spreadsheet.add_worksheet('assignments', 1000, 5)
            self.assignments_sheet.update('A1:D1', [
                ['annotator', 'batch_id', 'start_index', 'end_index']
            ])
        else:
            self.assignments_sheet = self.spreadsheet.worksheet('assignments')
    
    def upload_data(self, df: pd.DataFrame) -> bool:
        """Upload dataset to Google Sheets."""
        try:
            # Clear existing data
            self.data_sheet.clear()
            
            # Prepare data
            data = [['index', 'sentence1', 'sentence2', 'original_score']]
            for idx, row in df.iterrows():
                data.append([idx, row['sentence1'], row['sentence2'], row['score']])
            
            # Upload in batches
            self.data_sheet.update(f'A1:D{len(data)}', data)
            return True
        except Exception as e:
            st.error(f"Error uploading data: {e}")
            return False
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """Retrieve dataset from Google Sheets."""
        try:
            records = self.data_sheet.get_all_records()
            if records:
                return pd.DataFrame(records)
            return None
        except Exception as e:
            st.error(f"Error retrieving data: {e}")
            return None
    
    def save_annotation(
        self,
        index: int,
        annotator: str,
        score: float,
        notes: str = "",
        batch_id: str = ""
    ) -> bool:
        """Save a single annotation."""
        try:
            timestamp = datetime.now().isoformat()
            self.annotations_sheet.append_row([
                index, annotator, score, notes, timestamp, batch_id
            ])
            return True
        except Exception as e:
            st.error(f"Error saving annotation: {e}")
            return False
    
    def save_annotations_batch(self, annotations: List[Dict]) -> bool:
        """Save multiple annotations at once."""
        try:
            rows = []
            for ann in annotations:
                rows.append([
                    ann['index'],
                    ann['annotator'],
                    ann['score'],
                    ann.get('notes', ''),
                    ann.get('timestamp', datetime.now().isoformat()),
                    ann.get('batch_id', '')
                ])
            
            if rows:
                self.annotations_sheet.append_rows(rows)
            return True
        except Exception as e:
            st.error(f"Error saving batch: {e}")
            return False
    
    def get_annotations(self, annotator: str = None) -> pd.DataFrame:
        """Retrieve annotations, optionally filtered by annotator."""
        try:
            records = self.annotations_sheet.get_all_records()
            df = pd.DataFrame(records)
            
            if annotator and not df.empty:
                df = df[df['annotator'] == annotator]
            
            return df
        except Exception as e:
            st.error(f"Error retrieving annotations: {e}")
            return pd.DataFrame()
    
    def get_all_annotations(self) -> pd.DataFrame:
        """Get all annotations from all annotators."""
        try:
            records = self.annotations_sheet.get_all_records()
            return pd.DataFrame(records) if records else pd.DataFrame()
        except Exception as e:
            st.error(f"Error: {e}")
            return pd.DataFrame()
    
    def create_assignments(self, total_items: int, annotators: List[str]) -> bool:
        """Create batch assignments for annotators."""
        try:
            # Clear existing assignments
            self.assignments_sheet.clear()
            self.assignments_sheet.update('A1:D1', [
                ['annotator', 'batch_id', 'start_index', 'end_index']
            ])
            
            # Calculate batches
            num_batches = (total_items + ITEMS_PER_BATCH - 1) // ITEMS_PER_BATCH
            
            assignments = []
            for batch_idx in range(num_batches):
                start_idx = batch_idx * ITEMS_PER_BATCH
                end_idx = min(start_idx + ITEMS_PER_BATCH - 1, total_items - 1)
                batch_id = f"batch_{batch_idx + 1}"
                
                # Assign to multiple annotators for agreement
                for i in range(min(ANNOTATIONS_PER_ITEM, len(annotators))):
                    annotator = annotators[(batch_idx + i) % len(annotators)]
                    assignments.append([annotator, batch_id, start_idx, end_idx])
            
            if assignments:
                self.assignments_sheet.append_rows(assignments)
            
            return True
        except Exception as e:
            st.error(f"Error creating assignments: {e}")
            return False
    
    def get_assignments(self, annotator: str) -> List[Dict]:
        """Get assignments for a specific annotator."""
        try:
            records = self.assignments_sheet.get_all_records()
            return [r for r in records if r['annotator'] == annotator]
        except Exception as e:
            return []


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'authenticated': False,
        'current_user': None,
        'data': None,
        'current_index': 0,
        'local_annotations': {},
        'data_loaded': False,
        'backend': None,
        'backend_connected': False,
        'current_batch': None,
        'filter_mode': 'my_pending',
        'show_original_score': False,  # Hidden by default for blind annotation
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# Authentication
# =============================================================================

def render_login():
    """Render login screen."""
    st.markdown('<h1 class="main-header">👥 Team Annotation Tool</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 Team Login")
        
        annotator = st.selectbox(
            "Select your name",
            options=[""] + ANNOTATORS,
            key="login_annotator"
        )
        
        password = st.text_input(
            "Team password",
            type="password",
            key="login_password"
        )
        
        if st.button("Login", type="primary", use_container_width=True):
            if not annotator:
                st.error("Please select your name")
            elif password != TEAM_PASSWORD:
                st.error("Incorrect password")
            else:
                st.session_state.authenticated = True
                st.session_state.current_user = annotator
                st.rerun()
        
        st.markdown("---")
        st.info("💡 Contact your team lead if you need access or forgot the password.")


# =============================================================================
# Backend Setup
# =============================================================================

def render_backend_setup():
    """Render backend configuration UI."""
    st.markdown("### 🔧 Connect to Shared Storage")
    
    backend_type = st.radio(
        "Storage Backend",
        options=['Google Sheets (Recommended)', 'Local Only'],
        help="Google Sheets allows real-time collaboration"
    )
    
    if backend_type == 'Google Sheets (Recommended)':
        if not GSHEETS_AVAILABLE:
            st.error("Install required packages: `pip install gspread google-auth`")
            return
        
        st.info("""
        **Setup Instructions:**
        1. Create a Google Cloud project and enable Sheets API
        2. Create a Service Account and download JSON key
        3. Create a new Google Sheet and share it with the service account email
        4. Copy the Spreadsheet ID from the URL
        """)
        
        creds_file = st.file_uploader(
            "Upload Service Account JSON",
            type=['json'],
            key="gsheets_creds"
        )
        
        spreadsheet_id = st.text_input(
            "Spreadsheet ID",
            placeholder="e.g., 1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
            help="Find this in your Google Sheet URL"
        )
        
        if creds_file and spreadsheet_id:
            if st.button("🔗 Connect", type="primary"):
                try:
                    creds_dict = json.load(creds_file)
                    backend = GoogleSheetsBackend()
                    
                    if backend.authenticate(creds_dict):
                        if backend.open_or_create_spreadsheet(spreadsheet_id=spreadsheet_id):
                            st.session_state.backend = backend
                            st.session_state.backend_connected = True
                            st.success("✅ Connected to Google Sheets!")
                            st.rerun()
                except Exception as e:
                    st.error(f"Connection failed: {e}")
    
    else:
        st.warning("Local mode: Annotations won't sync with team members.")
        if st.button("Continue in Local Mode"):
            st.session_state.backend_connected = True
            st.rerun()


# =============================================================================
# Data Management
# =============================================================================

def render_data_upload():
    """Render data upload interface for admins."""
    st.markdown("### 📤 Upload Dataset")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="CSV with columns: sentence1, sentence2, score"
    )
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Validate
        required = ['sentence1', 'sentence2', 'score']
        if not all(c in df.columns for c in required):
            st.error(f"Missing required columns: {required}")
            return
        
        st.success(f"✅ Loaded {len(df)} items")
        st.dataframe(df.head(10))
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Upload to Shared Storage", type="primary"):
                if st.session_state.backend:
                    with st.spinner("Uploading..."):
                        if st.session_state.backend.upload_data(df):
                            st.success("Data uploaded!")
                            st.session_state.data = df
                            st.session_state.data_loaded = True
                else:
                    st.session_state.data = df
                    st.session_state.data_loaded = True
                st.rerun()
        
        with col2:
            if st.button("📋 Create Assignments"):
                if st.session_state.backend and len(df) > 0:
                    with st.spinner("Creating assignments..."):
                        if st.session_state.backend.create_assignments(len(df), ANNOTATORS):
                            st.success("Assignments created!")


def load_shared_data():
    """Load data from shared backend."""
    if st.session_state.backend:
        df = st.session_state.backend.get_data()
        if df is not None and not df.empty:
            st.session_state.data = df
            st.session_state.data_loaded = True
            return True
    return False


# =============================================================================
# Annotation Interface
# =============================================================================

def get_user_annotations() -> Dict:
    """Get current user's annotations from backend."""
    if st.session_state.backend:
        ann_df = st.session_state.backend.get_annotations(st.session_state.current_user)
        if not ann_df.empty:
            return {int(row['index']): row['expert_score'] for _, row in ann_df.iterrows()}
    return st.session_state.local_annotations


def get_my_pending_indices() -> List[int]:
    """Get indices that current user hasn't annotated yet."""
    if st.session_state.data is None:
        return []
    
    all_indices = set(range(len(st.session_state.data)))
    my_annotations = get_user_annotations()
    annotated = set(my_annotations.keys())
    
    return sorted(list(all_indices - annotated))


def get_filtered_indices() -> List[int]:
    """Get indices based on current filter."""
    if st.session_state.data is None:
        return []
    
    mode = st.session_state.filter_mode
    my_annotations = get_user_annotations()
    all_indices = list(range(len(st.session_state.data)))
    
    if mode == 'my_pending':
        return [i for i in all_indices if i not in my_annotations]
    elif mode == 'my_completed':
        return [i for i in all_indices if i in my_annotations]
    else:
        return all_indices


def navigate(direction: str):
    """Navigate to next/previous item."""
    filtered = get_filtered_indices()
    if not filtered:
        return
    
    current = st.session_state.current_index
    
    if direction == 'next':
        next_indices = [i for i in filtered if i > current]
        st.session_state.current_index = next_indices[0] if next_indices else filtered[0]
    else:
        prev_indices = [i for i in filtered if i < current]
        st.session_state.current_index = prev_indices[-1] if prev_indices else filtered[-1]


def save_annotation(index: int, score: float, notes: str = ""):
    """Save annotation to backend."""
    # Save locally first
    st.session_state.local_annotations[index] = score
    
    # Save to backend
    if st.session_state.backend:
        st.session_state.backend.save_annotation(
            index=index,
            annotator=st.session_state.current_user,
            score=score,
            notes=notes,
            batch_id=st.session_state.current_batch or ""
        )


def render_annotation_interface():
    """Render the main annotation interface."""
    df = st.session_state.data
    idx = st.session_state.current_index
    
    if idx >= len(df):
        st.session_state.current_index = 0
        idx = 0
    
    row = df.iloc[idx]
    my_annotations = get_user_annotations()
    
    # Header
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.button("⬅️ Previous", use_container_width=True):
            navigate('prev')
            st.rerun()
    
    with col2:
        is_annotated = idx in my_annotations
        status = "✅ You annotated this" if is_annotated else "⏳ Pending"
        st.markdown(
            f'<div style="text-align: center; padding: 0.5rem; background: {"#d4edda" if is_annotated else "#fff3cd"}; border-radius: 5px;">'
            f'<strong>Item {idx + 1} of {len(df)}</strong> | {status}</div>',
            unsafe_allow_html=True
        )
    
    with col3:
        if st.button("Next ➡️", use_container_width=True):
            navigate('next')
            st.rerun()
    
    st.markdown("---")
    
    # Sentences
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📝 Sentence 1")
        st.markdown(f'<div class="sentence-box">{row["sentence1"]}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 📝 Sentence 2")
        st.markdown(f'<div class="sentence-box">{row["sentence2"]}</div>', unsafe_allow_html=True)
    
    # Show original score only if enabled (for review mode)
    if st.session_state.show_original_score:
        st.info(f"**Original Score:** {row['original_score']:.2f}")
    
    st.markdown("---")
    
    # Annotation input
    st.markdown("#### 🎯 Your Annotation")
    
    current_value = my_annotations.get(idx, 0.5)
    
    expert_score = st.slider(
        "Similarity Score (0 = unrelated, 1 = identical meaning)",
        min_value=0.0,
        max_value=1.0,
        value=float(current_value),
        step=0.01,
        key=f"slider_{idx}"
    )
    
    # Quick buttons
    st.markdown("**Quick select:**")
    cols = st.columns(6)
    for i, val in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
        with cols[i]:
            if st.button(f"{val}", key=f"q_{val}_{idx}", use_container_width=True):
                save_annotation(idx, val)
                st.rerun()
    
    notes = st.text_area("Notes (optional)", height=60, key=f"notes_{idx}")
    
    st.markdown("---")
    
    # Actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save", type="secondary", use_container_width=True):
            save_annotation(idx, expert_score, notes)
            st.success("Saved!")
    
    with col2:
        if st.button("✅ Save & Next", type="primary", use_container_width=True):
            save_annotation(idx, expert_score, notes)
            navigate('next')
            st.rerun()
    
    with col3:
        if st.button("⏭️ Skip", use_container_width=True):
            navigate('next')
            st.rerun()


# =============================================================================
# Statistics & Agreement
# =============================================================================

def calculate_agreement_stats() -> Dict:
    """Calculate inter-annotator agreement statistics."""
    if not st.session_state.backend:
        return {}
    
    all_annotations = st.session_state.backend.get_all_annotations()
    
    if all_annotations.empty:
        return {}
    
    # Group by index
    grouped = all_annotations.groupby('index')
    
    agreements = []
    for idx, group in grouped:
        if len(group) >= 2:
            scores = group['expert_score'].values
            # Calculate pairwise differences
            diff = abs(scores[0] - scores[1]) if len(scores) >= 2 else 0
            agreements.append(1 - diff)  # Convert to agreement
    
    if not agreements:
        return {}
    
    return {
        'mean_agreement': np.mean(agreements),
        'std_agreement': np.std(agreements),
        'items_with_multiple': len(agreements),
        'total_annotations': len(all_annotations),
        'annotators': all_annotations['annotator'].nunique()
    }


def render_team_dashboard():
    """Render team statistics dashboard."""
    st.markdown("### 📊 Team Dashboard")
    
    if not st.session_state.backend:
        st.info("Connect to shared storage to see team statistics")
        return
    
    all_annotations = st.session_state.backend.get_all_annotations()
    
    if all_annotations.empty:
        st.info("No annotations yet")
        return
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    
    total_items = len(st.session_state.data) if st.session_state.data is not None else 0
    
    with col1:
        st.metric("Total Items", total_items)
    
    with col2:
        unique_annotated = all_annotations['index'].nunique()
        st.metric("Items Annotated", unique_annotated)
    
    with col3:
        st.metric("Total Annotations", len(all_annotations))
    
    with col4:
        st.metric("Active Annotators", all_annotations['annotator'].nunique())
    
    # Per-annotator breakdown
    st.markdown("#### 👥 Annotator Progress")
    
    annotator_stats = all_annotations.groupby('annotator').agg({
        'index': 'count',
        'expert_score': 'mean'
    }).rename(columns={'index': 'Count', 'expert_score': 'Avg Score'})
    
    st.dataframe(annotator_stats, use_container_width=True)
    
    # Agreement stats
    agreement = calculate_agreement_stats()
    if agreement:
        st.markdown("#### 🤝 Inter-Annotator Agreement")
        
        col1, col2 = st.columns(2)
        with col1:
            score = agreement['mean_agreement']
            color = 'agreement-high' if score > 0.8 else ('agreement-medium' if score > 0.6 else 'agreement-low')
            st.markdown(f"Mean Agreement: <span class='{color}'>{score:.2%}</span>", unsafe_allow_html=True)
        
        with col2:
            st.metric("Items with 2+ Annotations", agreement['items_with_multiple'])


# =============================================================================
# Export Functions
# =============================================================================

def generate_merged_export() -> Tuple[pd.DataFrame, List[Dict]]:
    """Generate merged export with all annotations."""
    df = st.session_state.data.copy()
    
    if st.session_state.backend:
        all_annotations = st.session_state.backend.get_all_annotations()
    else:
        # Local only
        all_annotations = pd.DataFrame([
            {'index': k, 'annotator': st.session_state.current_user, 'expert_score': v}
            for k, v in st.session_state.local_annotations.items()
        ])
    
    if all_annotations.empty:
        return df, []
    
    # Pivot annotations
    pivot = all_annotations.pivot_table(
        index='index',
        columns='annotator',
        values='expert_score',
        aggfunc='first'
    )
    
    # Merge with original data
    df = df.merge(pivot, left_index=True, right_index=True, how='left')
    
    # Calculate consensus (mean of expert scores)
    expert_cols = [c for c in df.columns if c in ANNOTATORS]
    if expert_cols:
        df['expert_consensus'] = df[expert_cols].mean(axis=1)
        df['expert_std'] = df[expert_cols].std(axis=1)
    
    # Generate instruction tuning format
    instruction_data = []
    for idx, row in df.iterrows():
        if pd.notna(row.get('expert_consensus')):
            instruction_data.append({
                "instruction": f"Output a number between 0 and 1 describing the semantic similarity between the following two sentences:\nSentence 1: {row['sentence1']}\nSentence 2: {row['sentence2']}",
                "input": "",
                "output": str(row['original_score']),
                "expert": str(round(row['expert_consensus'], 2))
            })
    
    return df, instruction_data


def render_export_section():
    """Render export options."""
    st.markdown("### 💾 Export Annotations")
    
    if st.session_state.data is None:
        st.info("Load data first")
        return
    
    df_export, json_export = generate_merged_export()
    
    col1, col2 = st.columns(2)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with col1:
        csv_buffer = StringIO()
        df_export.to_csv(csv_buffer, index=False)
        
        st.download_button(
            "📥 Download CSV (Full)",
            data=csv_buffer.getvalue(),
            file_name=f"annotations_full_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        if json_export:
            json_str = json.dumps(json_export, indent=2, ensure_ascii=False)
            
            st.download_button(
                "📥 Download JSON (Instruction)",
                data=json_str,
                file_name=f"annotations_instruction_{timestamp}.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.button("📥 No annotations yet", disabled=True, use_container_width=True)


# =============================================================================
# Sidebar
# =============================================================================

def render_sidebar():
    """Render sidebar with user info and controls."""
    with st.sidebar:
        # User info
        st.markdown(f'### <span class="annotator-badge">{st.session_state.current_user}</span>', unsafe_allow_html=True)
        
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.session_state.current_user = None
            st.rerun()
        
        st.markdown("---")
        
        # My progress
        if st.session_state.data is not None:
            my_annotations = get_user_annotations()
            total = len(st.session_state.data)
            done = len(my_annotations)
            
            st.markdown("### 📊 My Progress")
            st.progress(done / total if total > 0 else 0)
            st.markdown(f"**{done} / {total}** ({done/total*100:.1f}%)")
            
            pending = get_my_pending_indices()
            if pending:
                st.markdown(f"Next pending: **#{pending[0] + 1}**")
                if st.button("Go to next pending"):
                    st.session_state.current_index = pending[0]
                    st.rerun()
        
        st.markdown("---")
        
        # Filter
        st.markdown("### 🔍 Filter")
        st.session_state.filter_mode = st.selectbox(
            "Show items",
            options=['my_pending', 'my_completed', 'all'],
            format_func=lambda x: {
                'my_pending': '⏳ My Pending',
                'my_completed': '✅ My Completed',
                'all': '📋 All Items'
            }[x]
        )
        
        st.markdown("---")
        
        # Options
        st.markdown("### ⚙️ Options")
        st.session_state.show_original_score = st.checkbox(
            "Show original scores",
            value=st.session_state.show_original_score,
            help="Enable for review mode"
        )
        
        st.markdown("---")
        
        # Navigation
        if st.session_state.data is not None:
            st.markdown("### 🧭 Jump to")
            jump = st.number_input(
                "Item #",
                min_value=1,
                max_value=len(st.session_state.data),
                value=st.session_state.current_index + 1
            )
            if st.button("Go"):
                st.session_state.current_index = jump - 1
                st.rerun()


# =============================================================================
# Main App
# =============================================================================

def main():
    """Main application entry point."""
    init_session_state()
    
    # Authentication
    if not st.session_state.authenticated:
        render_login()
        return
    
    # Backend setup
    if not st.session_state.backend_connected:
        st.markdown('<h1 class="main-header">👥 Team Annotation Tool</h1>', unsafe_allow_html=True)
        render_backend_setup()
        return
    
    # Load data if not loaded
    if not st.session_state.data_loaded:
        load_shared_data()
    
    # Sidebar
    render_sidebar()
    
    # Main content
    st.markdown('<h1 class="main-header">👥 Team Annotation Tool</h1>', unsafe_allow_html=True)
    
    tabs = st.tabs(["✏️ Annotate", "📊 Team Dashboard", "📤 Upload Data", "💾 Export"])
    
    with tabs[0]:
        if st.session_state.data_loaded:
            render_annotation_interface()
        else:
            st.info("No data loaded. Go to 'Upload Data' tab or ask your team lead to upload the dataset.")
    
    with tabs[1]:
        render_team_dashboard()
    
    with tabs[2]:
        render_data_upload()
    
    with tabs[3]:
        render_export_section()


if __name__ == "__main__":
    main()
