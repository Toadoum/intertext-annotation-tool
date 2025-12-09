"""
Semantic Similarity Team Annotation Tool - Cloud Version
Optimized for Streamlit Cloud deployment with secrets management.
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from io import StringIO
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
# CONFIGURATION FROM SECRETS
# =============================================================================

def get_config():
    """Load configuration from Streamlit secrets or defaults."""
    config = {
        'team_password': 'annotate2024',
        'annotators': ['Sakayo', 'Annotator_2', 'Annotator_3', 'Annotator_4', 'Annotator_5'],
        'spreadsheet_id': None,
        'gcp_credentials': None,
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
        st.error(f"Error loading secrets: {e}")
    
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
    """Google Sheets backend for team collaboration."""
    
    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    
    def __init__(self):
        self.client = None
        self.spreadsheet = None
        self.data_sheet = None
        self.annotations_sheet = None
    
    @st.cache_resource
    def get_connection(_self, credentials_dict: Dict, spreadsheet_id: str):
        """Get cached connection to Google Sheets."""
        try:
            credentials = Credentials.from_service_account_info(
                credentials_dict,
                scopes=_self.SCOPES
            )
            client = gspread.authorize(credentials)
            spreadsheet = client.open_by_key(spreadsheet_id)
            return client, spreadsheet
        except Exception as e:
            st.error(f"Connection failed: {e}")
            return None, None
    
    def connect(self, credentials_dict: Dict, spreadsheet_id: str) -> bool:
        """Connect to Google Sheets."""
        self.client, self.spreadsheet = self.get_connection(credentials_dict, spreadsheet_id)
        
        if self.spreadsheet:
            self._ensure_sheets()
            return True
        return False
    
    def _ensure_sheets(self):
        """Ensure required sheets exist."""
        sheet_names = [ws.title for ws in self.spreadsheet.worksheets()]
        
        if 'data' not in sheet_names:
            self.data_sheet = self.spreadsheet.add_worksheet('data', 10000, 10)
            self.data_sheet.update('A1:D1', [['index', 'sentence1', 'sentence2', 'original_score']])
        else:
            self.data_sheet = self.spreadsheet.worksheet('data')
        
        if 'annotations' not in sheet_names:
            self.annotations_sheet = self.spreadsheet.add_worksheet('annotations', 50000, 10)
            self.annotations_sheet.update('A1:E1', [
                ['index', 'annotator', 'expert_score', 'notes', 'timestamp']
            ])
        else:
            self.annotations_sheet = self.spreadsheet.worksheet('annotations')
    
    def upload_data(self, df: pd.DataFrame) -> bool:
        """Upload dataset."""
        try:
            self.data_sheet.clear()
            data = [['index', 'sentence1', 'sentence2', 'original_score']]
            for idx, row in df.iterrows():
                data.append([int(idx), str(row['sentence1']), str(row['sentence2']), float(row['score'])])
            
            # Upload in chunks
            chunk_size = 1000
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i+chunk_size]
                start_row = i + 1
                end_row = start_row + len(chunk) - 1
                self.data_sheet.update(f'A{start_row}:D{end_row}', chunk)
            
            return True
        except Exception as e:
            st.error(f"Upload failed: {e}")
            return False
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """Get dataset from Google Sheets."""
        try:
            records = self.data_sheet.get_all_records()
            if records:
                return pd.DataFrame(records)
        except Exception as e:
            st.error(f"Error loading data: {e}")
        return None
    
    def save_annotation(self, index: int, annotator: str, score: float, notes: str = "") -> bool:
        """Save annotation."""
        try:
            timestamp = datetime.now().isoformat()
            self.annotations_sheet.append_row([int(index), annotator, float(score), notes, timestamp])
            return True
        except Exception as e:
            st.error(f"Save failed: {e}")
            return False
    
    def get_annotations(self, annotator: str = None) -> pd.DataFrame:
        """Get annotations from Google Sheets."""
        try:
            records = self.annotations_sheet.get_all_records()
            df = pd.DataFrame(records) if records else pd.DataFrame()
            
            if annotator and not df.empty and 'annotator' in df.columns:
                df = df[df['annotator'] == annotator]
            
            return df
        except Exception:
            return pd.DataFrame()
    
    def get_all_annotations(self) -> pd.DataFrame:
        """Get all annotations."""
        return self.get_annotations()


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
        'local_annotations': {},
        'backend': None,
        'connected': False,
        'filter_mode': 'my_pending',
        'show_original': False,
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
        st.markdown("### 🔐 Login")
        
        annotator = st.selectbox("Your name", [""] + CONFIG['annotators'])
        password = st.text_input("Team password", type="password")
        
        if st.button("Login", type="primary", use_container_width=True):
            if not annotator:
                st.error("Select your name")
            elif password != CONFIG['team_password']:
                st.error("Wrong password")
            else:
                st.session_state.authenticated = True
                st.session_state.current_user = annotator
                st.rerun()


# =============================================================================
# Data & Backend
# =============================================================================

def connect_backend():
    """Connect to Google Sheets backend."""
    if not GSHEETS_AVAILABLE:
        st.sidebar.error("❌ gspread not installed")
        return False
    
    if not CONFIG['gcp_credentials']:
        st.sidebar.warning("⚠️ No GCP credentials in secrets")
        return False
    
    if not CONFIG['spreadsheet_id']:
        st.sidebar.warning("⚠️ No spreadsheet_id in secrets")
        return False
    
    try:
        backend = GoogleSheetsBackend()
        if backend.connect(CONFIG['gcp_credentials'], CONFIG['spreadsheet_id']):
            st.session_state.backend = backend
            st.session_state.connected = True
            return True
        else:
            st.sidebar.error("❌ Backend connection failed")
    except Exception as e:
        st.sidebar.error(f"❌ Connection error: {e}")
    
    return False


def load_data():
    """Load data from backend."""
    if st.session_state.backend:
        df = st.session_state.backend.get_data()
        if df is not None and not df.empty:
            st.session_state.data = df
            return True
    return False


def get_user_annotations() -> Dict:
    """Get current user's annotations."""
    if st.session_state.backend:
        ann_df = st.session_state.backend.get_annotations(st.session_state.current_user)
        if not ann_df.empty and 'index' in ann_df.columns and 'expert_score' in ann_df.columns:
            return {int(row['index']): float(row['expert_score']) for _, row in ann_df.iterrows()}
    return st.session_state.local_annotations


# =============================================================================
# Navigation
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


# =============================================================================
# Annotation Interface
# =============================================================================

def save_annotation(idx: int, score: float, notes: str = ""):
    """Save annotation."""
    st.session_state.local_annotations[idx] = score
    
    if st.session_state.backend:
        st.session_state.backend.save_annotation(
            idx, st.session_state.current_user, score, notes
        )
        # Clear cache to refresh
        st.session_state.backend.get_annotations.clear()


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
        if st.button("⬅️ Prev", use_container_width=True):
            navigate('prev')
            st.rerun()
    
    with c2:
        status = "✅ Done" if is_done else "⏳ Pending"
        bg = "#d4edda" if is_done else "#fff3cd"
        st.markdown(
            f'<div style="text-align:center;padding:0.5rem;background:{bg};border-radius:5px;">'
            f'<b>#{idx + 1} / {len(df)}</b> | {status}</div>',
            unsafe_allow_html=True
        )
    
    with c3:
        if st.button("Next ➡️", use_container_width=True):
            navigate('next')
            st.rerun()
    
    st.markdown("---")
    
    # Sentences
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("**Sentence 1**")
        st.markdown(f'<div class="sentence-box">{row["sentence1"]}</div>', unsafe_allow_html=True)
    
    with c2:
        st.markdown("**Sentence 2**")
        st.markdown(f'<div class="sentence-box">{row["sentence2"]}</div>', unsafe_allow_html=True)
    
    if st.session_state.show_original:
        score_val = row.get('original_score', row.get('score', 'N/A'))
        st.caption(f"Original score: {score_val}")
    
    st.markdown("---")
    
    # Annotation
    st.markdown("**Your Score** (0 = unrelated → 1 = identical)")
    
    current_val = my_ann.get(idx, 0.5)
    score = st.slider("Score", 0.0, 1.0, float(current_val), 0.01, key=f"sl_{idx}", label_visibility="collapsed")
    
    # Quick buttons
    cols = st.columns(6)
    for i, v in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]):
        with cols[i]:
            if st.button(str(v), key=f"q{v}_{idx}", use_container_width=True):
                save_annotation(idx, v)
                navigate('next')
                st.rerun()
    
    notes = st.text_input("Notes (optional)", key=f"n_{idx}")
    
    # Actions
    c1, c2, c3 = st.columns(3)
    
    with c1:
        if st.button("💾 Save", use_container_width=True):
            save_annotation(idx, score, notes)
            st.success("Saved!")
    
    with c2:
        if st.button("✅ Save & Next", type="primary", use_container_width=True):
            save_annotation(idx, score, notes)
            navigate('next')
            st.rerun()
    
    with c3:
        if st.button("⏭️ Skip", use_container_width=True):
            navigate('next')
            st.rerun()


# =============================================================================
# Dashboard
# =============================================================================

def render_dashboard():
    """Render team dashboard."""
    st.markdown("### 📊 Team Progress")
    
    if not st.session_state.backend:
        st.info("Connect to shared storage for team stats")
        return
    
    all_ann = st.session_state.backend.get_all_annotations()
    total = len(st.session_state.data) if st.session_state.data is not None else 0
    
    if all_ann.empty:
        st.info("No annotations yet")
        return
    
    # Stats
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.metric("Total Items", total)
    with c2:
        st.metric("Annotated", all_ann['index'].nunique())
    with c3:
        st.metric("Annotations", len(all_ann))
    with c4:
        st.metric("Annotators", all_ann['annotator'].nunique())
    
    # Per-annotator
    st.markdown("#### By Annotator")
    stats = all_ann.groupby('annotator').agg(
        count=('index', 'count'),
        avg_score=('expert_score', 'mean')
    ).round(2)
    st.dataframe(stats, use_container_width=True)
    
    # Agreement
    if 'index' in all_ann.columns:
        multi = all_ann.groupby('index').filter(lambda x: len(x) >= 2)
        if not multi.empty:
            st.markdown("#### Inter-Annotator Agreement")
            
            agreements = []
            for idx, grp in multi.groupby('index'):
                scores = grp['expert_score'].values[:2]
                if len(scores) >= 2:
                    agreements.append(1 - abs(scores[0] - scores[1]))
            
            if agreements:
                mean_agr = np.mean(agreements)
                st.metric("Mean Agreement", f"{mean_agr:.1%}")
                st.caption(f"Based on {len(agreements)} items with 2+ annotations")


# =============================================================================
# Export
# =============================================================================

def render_export():
    """Render export section."""
    st.markdown("### 💾 Export")
    
    if st.session_state.data is None:
        st.info("No data loaded")
        return
    
    df = st.session_state.data.copy()
    
    # Get all annotations
    if st.session_state.backend:
        all_ann = st.session_state.backend.get_all_annotations()
    else:
        all_ann = pd.DataFrame([
            {'index': k, 'annotator': st.session_state.current_user, 'expert_score': v}
            for k, v in st.session_state.local_annotations.items()
        ])
    
    if all_ann.empty:
        st.warning("No annotations to export")
        return
    
    # Pivot
    if 'index' in all_ann.columns and 'annotator' in all_ann.columns:
        pivot = all_ann.pivot_table(index='index', columns='annotator', values='expert_score', aggfunc='first')
        df = df.merge(pivot, left_index=True, right_index=True, how='left')
        
        # Consensus
        ann_cols = [c for c in pivot.columns]
        if ann_cols:
            df['expert_consensus'] = df[ann_cols].mean(axis=1)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    c1, c2 = st.columns(2)
    
    with c1:
        csv_buf = StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            "📥 CSV (Full)",
            csv_buf.getvalue(),
            f"annotations_{timestamp}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with c2:
        # Instruction format
        json_data = []
        # Handle both 'score' and 'original_score' column names
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


# =============================================================================
# Upload
# =============================================================================

def render_upload():
    """Render upload section."""
    st.markdown("### 📤 Upload Dataset")
    
    uploaded = st.file_uploader("CSV file", type=['csv'])
    
    if uploaded:
        df = pd.read_csv(uploaded)
        
        required = ['sentence1', 'sentence2', 'score']
        if not all(c in df.columns for c in required):
            st.error(f"Need columns: {required}")
            return
        
        st.success(f"Loaded {len(df)} items")
        st.dataframe(df.head())
        
        if st.button("📤 Upload to Shared Storage", type="primary"):
            if st.session_state.backend:
                with st.spinner("Uploading..."):
                    if st.session_state.backend.upload_data(df):
                        st.success("Uploaded!")
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
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh"):
                # Force reload data from Google Sheets
                if st.session_state.backend:
                    df = st.session_state.backend.get_data()
                    if df is not None and not df.empty:
                        st.session_state.data = df
                        st.success(f"Loaded {len(df)} items")
                st.rerun()
        with col2:
            if st.button("Logout"):
                st.session_state.authenticated = False
                st.session_state.data = None
                st.rerun()
        
        st.markdown("---")
        
        # Progress
        if st.session_state.data is not None:
            my_ann = get_user_annotations()
            total = len(st.session_state.data)
            done = len(my_ann)
            pct = done / total if total else 0
            
            st.markdown("**My Progress**")
            st.progress(pct)
            st.caption(f"{done} / {total} ({pct:.1%})")
            
            pending = [i for i in range(total) if i not in my_ann]
            if pending:
                if st.button(f"▶️ Next pending (#{pending[0]+1})"):
                    st.session_state.current_index = pending[0]
                    st.rerun()
        
        st.markdown("---")
        
        # Filter
        st.session_state.filter_mode = st.selectbox(
            "Show",
            ['my_pending', 'my_done', 'all'],
            format_func=lambda x: {'my_pending': '⏳ Pending', 'my_done': '✅ Done', 'all': '📋 All'}[x]
        )
        
        st.session_state.show_original = st.checkbox("Show original scores")
        
        st.markdown("---")
        
        # Jump
        if st.session_state.data is not None:
            jump = st.number_input("Jump to #", 1, len(st.session_state.data), st.session_state.current_index + 1)
            if st.button("Go"):
                st.session_state.current_index = jump - 1
                st.rerun()
        
        st.markdown("---")
        
        # Status info
        st.caption("**Status**")
        st.caption(f"Backend: {'✅' if st.session_state.connected else '❌'}")
        data_status = f"✅ {len(st.session_state.data)} items" if st.session_state.data is not None else "❌ No data"
        st.caption(f"Data: {data_status}")


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
    
    # Connect backend (try each time if not connected)
    if not st.session_state.connected:
        connect_backend()
    
    # Load data
    if st.session_state.connected and st.session_state.data is None:
        load_data()
    
    # Render
    render_sidebar()
    
    st.markdown('<h1 class="main-header">👥 Team Annotation Tool</h1>', unsafe_allow_html=True)
    
    # Show connection debug info if backend not connected
    if not st.session_state.connected:
        with st.expander("🔧 Debug: Connection Info"):
            st.write("**Checking configuration...**")
            st.write(f"- gspread installed: {GSHEETS_AVAILABLE}")
            st.write(f"- Spreadsheet ID set: {CONFIG['spreadsheet_id'] is not None}")
            st.write(f"- GCP credentials set: {CONFIG['gcp_credentials'] is not None}")
            
            if CONFIG['gcp_credentials']:
                st.write(f"- Project ID: {CONFIG['gcp_credentials'].get('project_id', 'MISSING')}")
                st.write(f"- Client email: {CONFIG['gcp_credentials'].get('client_email', 'MISSING')}")
                has_key = 'private_key' in CONFIG['gcp_credentials'] and CONFIG['gcp_credentials']['private_key']
                st.write(f"- Private key present: {has_key}")
            
            if st.button("🔄 Retry Connection"):
                st.session_state.connected = False
                st.rerun()
    
    tabs = st.tabs(["✏️ Annotate", "📊 Dashboard", "📤 Upload", "💾 Export"])
    
    with tabs[0]:
        if st.session_state.data is not None:
            render_annotation_ui()
        else:
            st.info("📭 No data loaded yet.")
            st.markdown("""
            **To get started:**
            1. Go to **Upload** tab and upload your CSV file, OR
            2. Click **🔄 Refresh** in the sidebar if data was already uploaded
            
            Your CSV should have columns: `sentence1`, `sentence2`, `score`
            """)
    
    with tabs[1]:
        render_dashboard()
    
    with tabs[2]:
        render_upload()
    
    with tabs[3]:
        render_export()


if __name__ == "__main__":
    main()
