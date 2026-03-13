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
    config = {
        'team_password': 'annotate2024',
        'annotators': ['Sakayo', 'Annotateur_2', 'Annotateur_3', 'Annotateur_4', 'Annotateur_5'],
        'spreadsheet_id': None,
        'gcp_credentials': None,
        'use_local_fallback': True,
    }
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
# NEW FORMAT — required columns
# =============================================================================

REQUIRED_COLS = ['sentence1', 'sentence2', 'score']
NEW_COLS      = ['source_balkan', 'source_trans_saharan', 'pair_id', 'theme_label', 'pair_type']

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
    .source-box {
        background-color: #eaf4fb;
        padding: 0.6rem 1rem;
        border-radius: 6px;
        margin: 0.3rem 0;
        border-left: 3px solid #5ba4cf;
        font-size: 0.8rem;
        font-family: monospace;
        color: #2c3e50;
        word-break: break-all;
    }
    .meta-badge {
        display: inline-block;
        background-color: #e8f0fe;
        color: #3c4a7a;
        border-radius: 12px;
        padding: 0.2rem 0.7rem;
        font-size: 0.78rem;
        margin: 0.2rem 0.2rem 0.2rem 0;
        font-weight: 600;
    }
    .ptype-cross      { background-color: #d4edda; color: #155724; }
    .ptype-intra_balk { background-color: #fff3cd; color: #856404; }
    .ptype-intra_sah  { background-color: #fce4ec; color: #880e4f; }
    .score-explanation {
        background-color: #f8f9fa;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
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
# LOCAL STORAGE BACKEND
# =============================================================================

class LocalStorageBackend:
    def __init__(self):
        self.data = None
        self.annotations = []

    def upload_data(self, df: pd.DataFrame) -> bool:
        try:
            self.data = df.copy()
            return True
        except Exception as e:
            st.error(f"Échec du téléversement local: {e}")
            return False

    def get_data(self) -> Optional[pd.DataFrame]:
        return self.data

    def save_annotation(self, index: int, annotator: str, score: float, notes: str = "") -> bool:
        try:
            self.annotations.append({
                'index': int(index),
                'annotator': annotator,
                'expert_score': float(score),
                'notes': notes,
                'timestamp': datetime.now().isoformat()
            })
            return True
        except Exception as e:
            st.error(f"Échec de la sauvegarde locale: {e}")
            return False

    def get_annotations(self, annotator: str = None) -> pd.DataFrame:
        if not self.annotations:
            return pd.DataFrame()
        df = pd.DataFrame(self.annotations)
        if annotator and not df.empty and 'annotator' in df.columns:
            df = df[df['annotator'] == annotator]
        return df

    def get_all_annotations(self) -> pd.DataFrame:
        return self.get_annotations()

# =============================================================================
# GOOGLE SHEETS BACKEND
# =============================================================================

try:
    import gspread
    from google.oauth2.service_account import Credentials
    from google.auth.exceptions import GoogleAuthError
    GSHEETS_AVAILABLE = True
except ImportError:
    GSHEETS_AVAILABLE = False

class GoogleSheetsBackend:
    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive.file'
    ]

    # Extended data columns to store in GSheets
    DATA_HEADERS = [
        'index', 'sentence1', 'sentence2', 'score',
        'source_balkan', 'source_trans_saharan',
        'pair_id', 'theme_label', 'pair_type'
    ]

    def __init__(self):
        self.client = None
        self.spreadsheet = None
        self.data_sheet = None
        self.annotations_sheet = None
        self.last_error = None

    def get_connection(self, credentials_dict, spreadsheet_id):
        try:
            if 'private_key' in credentials_dict:
                pk = credentials_dict['private_key']
                if '\\n' in pk:
                    credentials_dict['private_key'] = pk.replace('\\n', '\n')
            creds = Credentials.from_service_account_info(credentials_dict, scopes=self.SCOPES)
            client = gspread.authorize(creds)
            spreadsheet = client.open_by_key(spreadsheet_id)
            return client, spreadsheet
        except Exception as e:
            self.last_error = f"Erreur de connexion: {e}"
            return None, None

    def connect(self, credentials_dict, spreadsheet_id) -> bool:
        if not credentials_dict or not spreadsheet_id:
            self.last_error = "Identifiants manquants"
            return False
        self.client, self.spreadsheet = self.get_connection(credentials_dict, spreadsheet_id)
        if self.spreadsheet:
            try:
                self._ensure_sheets()
                return True
            except Exception as e:
                self.last_error = f"Échec configuration feuilles: {e}"
                return False
        return False

    def _ensure_sheets(self):
        sheet_names = [ws.title for ws in self.spreadsheet.worksheets()]
        if 'data' not in sheet_names:
            self.data_sheet = self.spreadsheet.add_worksheet('data', 1000, len(self.DATA_HEADERS))
            self.data_sheet.update('A1', [self.DATA_HEADERS])
        else:
            self.data_sheet = self.spreadsheet.worksheet('data')

        if 'annotations' not in sheet_names:
            self.annotations_sheet = self.spreadsheet.add_worksheet('annotations', 1000, 6)
            self.annotations_sheet.update('A1', [['index', 'annotator', 'expert_score', 'notes', 'timestamp']])
        else:
            self.annotations_sheet = self.spreadsheet.worksheet('annotations')

    def upload_data(self, df: pd.DataFrame) -> bool:
        try:
            self.data_sheet.clear()
            rows = [self.DATA_HEADERS]
            for idx, row in df.iterrows():
                rows.append([
                    int(idx),
                    str(row.get('sentence1', ''))[:500],
                    str(row.get('sentence2', ''))[:500],
                    float(row.get('score', 0.0)),
                    str(row.get('source_balkan', ''))[:300],
                    str(row.get('source_trans_saharan', ''))[:300],
                    str(row.get('pair_id', '')),
                    str(row.get('theme_label', '')),
                    str(row.get('pair_type', '')),
                ])
            chunk_size = 500
            for i in range(0, len(rows), chunk_size):
                chunk = rows[i:i + chunk_size]
                self.data_sheet.update(f'A{i+1}', chunk)
            return True
        except Exception as e:
            self.last_error = f"Échec du téléversement: {e}"
            return False

    def get_data(self) -> Optional[pd.DataFrame]:
        try:
            records = self.data_sheet.get_all_records()
            return pd.DataFrame(records) if records else pd.DataFrame()
        except Exception as e:
            self.last_error = f"Erreur chargement: {e}"
            return None

    def save_annotation(self, index: int, annotator: str, score: float, notes: str = "") -> bool:
        try:
            self.annotations_sheet.append_row([
                int(index), annotator, float(score), notes[:100],
                datetime.now().isoformat()
            ])
            return True
        except Exception as e:
            self.last_error = f"Échec sauvegarde: {e}"
            return False

    def get_annotations(self, annotator: str = None) -> pd.DataFrame:
        try:
            records = self.annotations_sheet.get_all_records()
            df = pd.DataFrame(records) if records else pd.DataFrame()
            if annotator and not df.empty and 'annotator' in df.columns:
                df = df[df['annotator'] == annotator]
            return df
        except Exception as e:
            self.last_error = f"Échec récupération: {e}"
            return pd.DataFrame()

    def get_all_annotations(self) -> pd.DataFrame:
        return self.get_annotations()

# =============================================================================
# BACKEND MANAGER
# =============================================================================

class BackendManager:
    def __init__(self):
        self.google_backend = None
        self.local_backend = None
        self.active_backend = None
        self.mode = "local"

    def initialize(self):
        self.local_backend = LocalStorageBackend()
        if GSHEETS_AVAILABLE and CONFIG['gcp_credentials'] and CONFIG['spreadsheet_id']:
            self.google_backend = GoogleSheetsBackend()
            if self.google_backend.connect(CONFIG['gcp_credentials'], CONFIG['spreadsheet_id']):
                self.active_backend = self.google_backend
                self.mode = "google"
                return True
            else:
                st.sidebar.warning(f"⚠️ Google Sheets échoué: {self.google_backend.last_error}")
        self.active_backend = self.local_backend
        self.mode = "local"
        return False

    def get_backend(self):
        return self.active_backend

    def get_mode(self):
        return self.mode

# =============================================================================
# SESSION STATE
# =============================================================================

def init_session_state():
    defaults = {
        'authenticated': False,
        'current_user': None,
        'data': None,
        'current_index': 0,
        'backend_manager': None,
        'filter_mode': 'my_pending',
        'show_original': True,
        'show_meta': True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.backend_manager is None:
        st.session_state.backend_manager = BackendManager()

# =============================================================================
# AUTHENTICATION
# =============================================================================

def render_login():
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
                bm = BackendManager()
                bm.initialize()
                st.session_state.backend_manager = bm
                st.rerun()

# =============================================================================
# DATA HELPERS
# =============================================================================

def load_data():
    backend = st.session_state.backend_manager.get_backend()
    if backend:
        df = backend.get_data()
        if df is not None and not df.empty:
            st.session_state.data = df
            return True
    return False

def get_user_annotations() -> Dict:
    backend = st.session_state.backend_manager.get_backend()
    if backend:
        ann_df = backend.get_annotations(st.session_state.current_user)
        if not ann_df.empty and 'index' in ann_df.columns and 'expert_score' in ann_df.columns:
            return {int(r['index']): float(r['expert_score']) for _, r in ann_df.iterrows()}
    return {}

def get_filtered_indices() -> List[int]:
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
    backend = st.session_state.backend_manager.get_backend()
    if backend:
        backend.save_annotation(idx, st.session_state.current_user, score, notes)

# =============================================================================
# ANNOTATION UI
# =============================================================================

PTYPE_LABELS = {
    'cross':       ('🔀 Cross-corpus',    'ptype-cross'),
    'intra_balk':  ('🌍 Intra-Balkanique', 'ptype-intra_balk'),
    'intra_sah':   ('🏜️ Intra-Saharien',  'ptype-intra_sah'),
}

def render_annotation_ui():
    df = st.session_state.data
    idx = st.session_state.current_index
    if idx >= len(df):
        idx = 0
        st.session_state.current_index = 0

    row = df.iloc[idx]
    my_ann = get_user_annotations()
    is_done = idx in my_ann

    # ── Navigation header ──────────────────────────────────────────────────
    c1, c2, c3 = st.columns([1, 3, 1])
    with c1:
        if st.button("⬅️ Précédent", use_container_width=True):
            navigate('prev')
            st.rerun()
    with c2:
        mode_icon = "☁️" if st.session_state.backend_manager.get_mode() == "google" else "💻"
        status = "✅ Terminé" if is_done else "⏳ En attente"
        bg = "#d4edda" if is_done else "#fff3cd"
        st.markdown(
            f'<div style="text-align:center;padding:0.5rem;background:{bg};border-radius:5px;">'
            f'<b>#{idx + 1} / {len(df)}</b> | {status} | {mode_icon}</div>',
            unsafe_allow_html=True
        )
    with c3:
        if st.button("Suivant ➡️", use_container_width=True):
            navigate('next')
            st.rerun()

    st.markdown("---")

    # ── Metadata badges ────────────────────────────────────────────────────
    if st.session_state.show_meta:
        badges_html = ""

        # pair_id
        pair_id = str(row.get('pair_id', '')).strip()
        if pair_id:
            badges_html += f'<span class="meta-badge">🔑 {pair_id}</span>'

        # pair_type
        ptype = str(row.get('pair_type', '')).strip()
        if ptype:
            label, css = PTYPE_LABELS.get(ptype, (ptype, ''))
            badges_html += f'<span class="meta-badge {css}">{label}</span>'

        # theme_label
        theme = str(row.get('theme_label', '')).strip()
        if theme and theme not in ('', 'nan'):
            badges_html += f'<span class="meta-badge">🏷️ {theme}</span>'

        if badges_html:
            st.markdown(badges_html, unsafe_allow_html=True)

    # ── Sentences ──────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Phrase 1** — Corpus Balkanique")
        st.markdown(f'<div class="sentence-box">{row["sentence1"]}</div>', unsafe_allow_html=True)
        src_balk = str(row.get('source_balkan', '')).strip()
        if src_balk and src_balk not in ('', 'nan') and st.session_state.show_meta:
            st.markdown(f'<div class="source-box">📄 {src_balk}</div>', unsafe_allow_html=True)

    with c2:
        st.markdown("**Phrase 2** — Corpus Trans-Saharien")
        st.markdown(f'<div class="sentence-box">{row["sentence2"]}</div>', unsafe_allow_html=True)
        src_sah = str(row.get('source_trans_saharan', '')).strip()
        if src_sah and src_sah not in ('', 'nan') and st.session_state.show_meta:
            st.markdown(f'<div class="source-box">📄 {src_sah}</div>', unsafe_allow_html=True)

    # ── Original score ─────────────────────────────────────────────────────
    if st.session_state.show_original:
        orig = row.get('score', 'N/A')
        st.caption(f"**Score automatique (heuristique):** {orig}")

    st.markdown("---")

    # ── Scoring ────────────────────────────────────────────────────────────
    st.markdown('<div class="score-explanation">'
                '<b>📊 Échelle de notation:</b>'
                '<br>0.0 = Phrases complètement différentes (sans rapport)'
                '<br>1.0 = Identique en sens (même signification)'
                '<br><small>Utilisez les valeurs intermédiaires pour exprimer différents niveaux de similarité.</small>'
                '</div>', unsafe_allow_html=True)

    st.markdown("**Votre Note** (0 = sans rapport → 1 = identique)")
    current_val = my_ann.get(idx, 0.5)
    score = st.slider("Note", 0.0, 1.0, float(current_val), 0.01,
                      key=f"sl_{idx}", label_visibility="collapsed",
                      help="0.0 = sans rapport, 1.0 = identique en sens")

    cols = st.columns(6)
    for i, (v, label) in enumerate(zip(
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        ["0.0\nSans rapport", "0.2", "0.4", "0.6", "0.8", "1.0\nIdentique"]
    )):
        with cols[i]:
            if st.button(label, key=f"q{v}_{idx}", use_container_width=True):
                save_annotation(idx, v)
                navigate('next')
                st.rerun()

    notes = st.text_input("Notes (optionnel)", key=f"n_{idx}")

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
# DASHBOARD
# =============================================================================

def render_dashboard():
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

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total des items", total)
    with c2: st.metric("Items annotés", all_ann['index'].nunique())
    with c3: st.metric("Annotations", len(all_ann))
    with c4: st.metric("Annotateurs", all_ann['annotator'].nunique())

    st.markdown("#### Par Annotateur")
    stats = all_ann.groupby('annotator').agg(
        Nombre=('index', 'count'),
        Moyenne=('expert_score', 'mean')
    ).round(3)
    st.dataframe(stats, use_container_width=True)

    # Pair-type breakdown if data loaded
    if st.session_state.data is not None and 'pair_type' in st.session_state.data.columns:
        st.markdown("#### Répartition par type de paire")
        df = st.session_state.data.copy()
        df['index_col'] = df.index
        merged = all_ann.merge(df[['index_col', 'pair_type', 'theme_label']],
                               left_on='index', right_on='index_col', how='left')
        if 'pair_type' in merged.columns:
            pt = merged.groupby('pair_type').agg(
                Annotées=('index', 'count'),
                Score_moyen=('expert_score', 'mean')
            ).round(3)
            st.dataframe(pt, use_container_width=True)

    # Agreement
    multi = all_ann.groupby('index').filter(lambda x: len(x) >= 2)
    if not multi.empty:
        st.markdown("#### Accord Inter-Annotateurs")
        agreements = []
        for _, grp in multi.groupby('index'):
            s = grp['expert_score'].values[:2]
            if len(s) >= 2:
                agreements.append(1 - abs(s[0] - s[1]))
        if agreements:
            st.metric("Accord moyen", f"{np.mean(agreements):.1%}")
            st.caption(f"Basé sur {len(agreements)} items avec 2+ annotations")

# =============================================================================
# UPLOAD
# =============================================================================

def render_upload():
    st.markdown("### 📤 Téléverser un Dataset")
    st.markdown("""
    **Format attendu** — colonnes requises : `sentence1`, `sentence2`, `score`  
    Colonnes supplémentaires reconnues : `source_balkan`, `source_trans_saharan`, `pair_id`, `theme_label`, `pair_type`
    """)

    uploaded = st.file_uploader("Fichier CSV", type=['csv'])

    if uploaded:
        df = pd.read_csv(uploaded)

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            st.error(f"Colonnes manquantes : {missing}")
            return

        # Fill optional new columns with empty string if absent
        for col in NEW_COLS:
            if col not in df.columns:
                df[col] = ""

        st.success(f"✅ {len(df)} paires chargées — {df.shape[1]} colonnes")

        # Preview with new columns
        preview_cols = REQUIRED_COLS + [c for c in NEW_COLS if df[c].astype(str).str.strip().ne('').any()]
        st.dataframe(df[preview_cols].head(5), use_container_width=True)

        # Show pair_type distribution if present
        if 'pair_type' in df.columns and df['pair_type'].astype(str).str.strip().ne('').any():
            st.markdown("**Répartition des types de paires :**")
            st.dataframe(df['pair_type'].value_counts().rename("Nombre"), use_container_width=True)

        if st.button("📤 Téléverser vers le stockage", type="primary"):
            backend = st.session_state.backend_manager.get_backend()
            with st.spinner("Téléversement en cours…"):
                if backend and backend.upload_data(df):
                    st.success("Téléversé avec succès!")
                    st.session_state.data = df
                    st.rerun()
                else:
                    # Fallback: store in session state directly
                    st.session_state.data = df
                    st.rerun()

# =============================================================================
# EXPORT
# =============================================================================

def render_export():
    st.markdown("### 💾 Exporter")
    if st.session_state.data is None:
        st.info("Aucune donnée chargée")
        return

    df = st.session_state.data.copy()
    backend = st.session_state.backend_manager.get_backend()
    all_ann = backend.get_all_annotations() if backend else pd.DataFrame()

    if all_ann.empty:
        st.warning("Aucune annotation à exporter")
        return

    if 'index' in all_ann.columns and 'annotator' in all_ann.columns:
        pivot = all_ann.pivot_table(
            index='index', columns='annotator',
            values='expert_score', aggfunc='first'
        )
        df = df.merge(pivot, left_index=True, right_index=True, how='left')
        ann_cols = list(pivot.columns)
        if ann_cols:
            df['expert_consensus'] = df[ann_cols].mean(axis=1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    c1, c2 = st.columns(2)

    with c1:
        buf = StringIO()
        df.to_csv(buf, index=False)
        st.download_button(
            "📥 CSV (Complet)",
            buf.getvalue(),
            f"annotations_{timestamp}.csv",
            "text/csv",
            use_container_width=True
        )

    with c2:
        json_data = []
        if 'expert_consensus' in df.columns:
            for _, row in df.iterrows():
                if pd.notna(row.get('expert_consensus')):
                    entry = {
                        "instruction": (
                            f"Output a number between 0 and 1 describing the semantic similarity "
                            f"between the following two sentences:\n"
                            f"Sentence 1: {row['sentence1']}\nSentence 2: {row['sentence2']}"
                        ),
                        "input": "",
                        "output": str(round(row.get('score', ''), 4)),
                        "expert": str(round(row['expert_consensus'], 2)),
                    }
                    # Preserve new metadata columns
                    for col in NEW_COLS:
                        if col in row and str(row[col]).strip() not in ('', 'nan'):
                            entry[col] = str(row[col])
                    json_data.append(entry)

        if json_data:
            st.download_button(
                "📥 JSON (Instruction + Métadonnées)",
                json.dumps(json_data, indent=2, ensure_ascii=False),
                f"annotations_{timestamp}.json",
                "application/json",
                use_container_width=True
            )

# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.current_user}")

        mode = st.session_state.backend_manager.get_mode()
        if mode == "google":
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

        st.session_state.filter_mode = st.selectbox(
            "Afficher",
            ['my_pending', 'my_done', 'all'],
            format_func=lambda x: {
                'my_pending': '⏳ En attente',
                'my_done': '✅ Terminé',
                'all': '📋 Tous'
            }[x]
        )

        st.session_state.show_original = st.checkbox("Afficher le score automatique", value=True)
        st.session_state.show_meta = st.checkbox("Afficher métadonnées (source/thème)", value=True)

        st.markdown("---")

        if st.session_state.data is not None:
            jump = st.number_input(
                "Aller à #", 1, len(st.session_state.data),
                st.session_state.current_index + 1
            )
            if st.button("Aller"):
                st.session_state.current_index = jump - 1
                st.rerun()

# =============================================================================
# MAIN
# =============================================================================

def main():
    init_session_state()

    if not st.session_state.authenticated:
        render_login()
        return

    if st.session_state.data is None:
        with st.spinner("Chargement des données…"):
            load_data()

    render_sidebar()

    st.markdown('<h1 class="main-header">👥 Outil d\'Annotation d\'Équipe</h1>', unsafe_allow_html=True)

    tabs = st.tabs(["✏️ Annoter", "📊 Tableau de bord", "📤 Téléverser", "💾 Exporter"])

    with tabs[0]:
        if st.session_state.data is not None:
            render_annotation_ui()
        else:
            st.info("📭 Aucune donnée chargée.")
            st.markdown("""
            **Pour commencer :**
            1. Allez à l'onglet **Téléverser** et téléversez votre CSV, **ou**
            2. Cliquez sur **🔄 Rafraîchir** si des données ont déjà été téléversées.

            **Format attendu :** `sentence1`, `sentence2`, `score`, `source_balkan`,
            `source_trans_saharan`, `pair_id`, `theme_label`, `pair_type`
            """)

    with tabs[1]:
        render_dashboard()

    with tabs[2]:
        render_upload()

    with tabs[3]:
        render_export()


if __name__ == "__main__":
    main()
