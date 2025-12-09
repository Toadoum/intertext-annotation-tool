"""
Semantic Similarity Data Annotation Tool
A Streamlit application for annotating sentence pair similarity scores.
Supports Google Drive integration for large datasets.
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Semantic Similarity Annotator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
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
    .score-display {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #ff6b6b 0%, #feca57 50%, #48dbfb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .progress-card {
        background-color: #e8f4ea;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
    }
    .annotation-complete {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 0.5rem;
        border-radius: 5px;
        color: #155724;
    }
    .annotation-pending {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        padding: 0.5rem;
        border-radius: 5px;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'data': None,
        'current_index': 0,
        'annotations': {},
        'annotator_name': '',
        'data_loaded': False,
        'gdrive_authenticated': False,
        'file_name': '',
        'annotation_history': [],
        'filter_mode': 'all',
        'show_original_score': True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_from_gdrive():
    """Load data from Google Drive using file ID."""
    st.subheader("📁 Load from Google Drive")
    
    st.info("""
    **How to get your Google Drive file ID:**
    1. Open your CSV file in Google Drive
    2. Click "Share" → "Copy link"
    3. The link looks like: `https://drive.google.com/file/d/FILE_ID/view`
    4. Copy the `FILE_ID` part and paste below
    
    **Note:** Make sure the file is shared as "Anyone with the link can view"
    """)
    
    file_id = st.text_input(
        "Enter Google Drive File ID",
        placeholder="e.g., 1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"
    )
    
    if st.button("📥 Load from Google Drive", type="primary"):
        if file_id:
            try:
                # Construct direct download URL
                url = f"https://drive.google.com/uc?export=download&id={file_id}"
                
                with st.spinner("Loading data from Google Drive..."):
                    df = pd.read_csv(url)
                    
                    # Validate required columns
                    required_cols = ['sentence1', 'sentence2', 'score']
                    if not all(col in df.columns for col in required_cols):
                        st.error(f"CSV must contain columns: {required_cols}")
                        return False
                    
                    st.session_state.data = df
                    st.session_state.data_loaded = True
                    st.session_state.file_name = f"gdrive_{file_id[:8]}"
                    st.success(f"✅ Loaded {len(df)} sentence pairs from Google Drive!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                st.info("Make sure the file is publicly accessible or try downloading and uploading directly.")
        else:
            st.warning("Please enter a valid File ID")
    
    return False


def load_from_upload():
    """Load data from uploaded file."""
    st.subheader("📤 Upload CSV File")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with columns: sentence1, sentence2, score"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate required columns
            required_cols = ['sentence1', 'sentence2', 'score']
            if not all(col in df.columns for col in required_cols):
                st.error(f"CSV must contain columns: {required_cols}")
                return False
            
            st.session_state.data = df
            st.session_state.data_loaded = True
            st.session_state.file_name = uploaded_file.name.replace('.csv', '')
            st.success(f"✅ Loaded {len(df)} sentence pairs!")
            return True
            
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            return False
    
    return False


def load_existing_annotations():
    """Load existing annotations from a JSON file."""
    st.subheader("📂 Load Existing Annotations")
    
    annotation_file = st.file_uploader(
        "Upload existing annotations (JSON)",
        type=['json'],
        key="annotation_upload",
        help="Resume annotation from a previously saved session"
    )
    
    if annotation_file is not None:
        try:
            annotations = json.load(annotation_file)
            st.session_state.annotations = {int(k): v for k, v in annotations.items()}
            st.success(f"✅ Loaded {len(annotations)} existing annotations!")
            return True
        except Exception as e:
            st.error(f"Error loading annotations: {str(e)}")
    
    return False


def get_annotation_stats():
    """Calculate annotation statistics."""
    if st.session_state.data is None:
        return {}
    
    total = len(st.session_state.data)
    annotated = len(st.session_state.annotations)
    remaining = total - annotated
    progress = (annotated / total * 100) if total > 0 else 0
    
    return {
        'total': total,
        'annotated': annotated,
        'remaining': remaining,
        'progress': progress
    }


def render_sidebar():
    """Render the sidebar with controls and statistics."""
    with st.sidebar:
        st.markdown("### 👤 Annotator Info")
        st.session_state.annotator_name = st.text_input(
            "Your Name",
            value=st.session_state.annotator_name,
            placeholder="Enter your name"
        )
        
        st.markdown("---")
        
        # Progress statistics
        stats = get_annotation_stats()
        if stats:
            st.markdown("### 📊 Progress")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total", stats['total'])
                st.metric("Annotated", stats['annotated'])
            with col2:
                st.metric("Remaining", stats['remaining'])
                st.metric("Progress", f"{stats['progress']:.1f}%")
            
            st.progress(stats['progress'] / 100)
        
        st.markdown("---")
        
        # Navigation controls
        st.markdown("### 🧭 Navigation")
        
        # Filter options
        filter_mode = st.selectbox(
            "Show",
            options=['all', 'annotated', 'unannotated'],
            format_func=lambda x: {
                'all': '📋 All Items',
                'annotated': '✅ Annotated Only',
                'unannotated': '⏳ Unannotated Only'
            }.get(x, x)
        )
        st.session_state.filter_mode = filter_mode
        
        # Jump to specific index
        if st.session_state.data is not None:
            jump_to = st.number_input(
                "Jump to index",
                min_value=0,
                max_value=len(st.session_state.data) - 1,
                value=st.session_state.current_index
            )
            if st.button("🎯 Go"):
                st.session_state.current_index = jump_to
                st.rerun()
        
        st.markdown("---")
        
        # Display options
        st.markdown("### ⚙️ Options")
        st.session_state.show_original_score = st.checkbox(
            "Show original score",
            value=st.session_state.show_original_score,
            help="Display the original similarity score"
        )
        
        st.markdown("---")
        
        # Export section
        st.markdown("### 💾 Export")
        render_export_buttons()
        
        st.markdown("---")
        
        # Reset option
        if st.button("🔄 Reset Session", type="secondary"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def get_filtered_indices():
    """Get indices based on current filter mode."""
    if st.session_state.data is None:
        return []
    
    all_indices = list(range(len(st.session_state.data)))
    
    if st.session_state.filter_mode == 'annotated':
        return [i for i in all_indices if i in st.session_state.annotations]
    elif st.session_state.filter_mode == 'unannotated':
        return [i for i in all_indices if i not in st.session_state.annotations]
    else:
        return all_indices


def navigate(direction):
    """Navigate to next/previous item based on filter."""
    filtered = get_filtered_indices()
    if not filtered:
        return
    
    current = st.session_state.current_index
    
    if direction == 'next':
        # Find next index in filtered list
        next_indices = [i for i in filtered if i > current]
        if next_indices:
            st.session_state.current_index = next_indices[0]
        else:
            # Wrap around
            st.session_state.current_index = filtered[0]
    else:
        # Find previous index in filtered list
        prev_indices = [i for i in filtered if i < current]
        if prev_indices:
            st.session_state.current_index = prev_indices[-1]
        else:
            # Wrap around
            st.session_state.current_index = filtered[-1]


def render_annotation_interface():
    """Render the main annotation interface."""
    df = st.session_state.data
    idx = st.session_state.current_index
    
    if idx >= len(df):
        st.session_state.current_index = 0
        idx = 0
    
    row = df.iloc[idx]
    
    # Header with navigation
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        if st.button("⬅️ Previous", use_container_width=True):
            navigate('prev')
            st.rerun()
    
    with col2:
        # Status indicator
        is_annotated = idx in st.session_state.annotations
        status_class = "annotation-complete" if is_annotated else "annotation-pending"
        status_text = "✅ Annotated" if is_annotated else "⏳ Pending"
        st.markdown(
            f'<div class="{status_class}" style="text-align: center;">'
            f'<strong>Item {idx + 1} of {len(df)}</strong> | {status_text}</div>',
            unsafe_allow_html=True
        )
    
    with col3:
        if st.button("Next ➡️", use_container_width=True):
            navigate('next')
            st.rerun()
    
    st.markdown("---")
    
    # Sentence display
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📝 Sentence 1")
        st.markdown(
            f'<div class="sentence-box">{row["sentence1"]}</div>',
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown("#### 📝 Sentence 2")
        st.markdown(
            f'<div class="sentence-box">{row["sentence2"]}</div>',
            unsafe_allow_html=True
        )
    
    # Original score (if enabled)
    if st.session_state.show_original_score:
        st.markdown(f"**Original Score:** `{row['score']:.2f}`")
    
    st.markdown("---")
    
    # Annotation input
    st.markdown("#### 🎯 Your Annotation")
    
    # Get current annotation value
    current_annotation = st.session_state.annotations.get(idx, row['score'])
    
    # Slider for score
    expert_score = st.slider(
        "Similarity Score (0 = completely different, 1 = identical meaning)",
        min_value=0.0,
        max_value=1.0,
        value=float(current_annotation),
        step=0.01,
        key=f"slider_{idx}"
    )
    
    # Quick buttons for common values
    st.markdown("**Quick select:**")
    quick_cols = st.columns(6)
    quick_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    for i, val in enumerate(quick_values):
        with quick_cols[i]:
            if st.button(f"{val}", key=f"quick_{val}_{idx}", use_container_width=True):
                st.session_state.annotations[idx] = val
                st.rerun()
    
    # Notes field
    notes = st.text_area(
        "Notes (optional)",
        value=st.session_state.annotations.get(f"{idx}_notes", ""),
        placeholder="Add any notes about this annotation...",
        height=80
    )
    
    st.markdown("---")
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("💾 Save & Stay", type="secondary", use_container_width=True):
            st.session_state.annotations[idx] = expert_score
            if notes:
                st.session_state.annotations[f"{idx}_notes"] = notes
            st.success("Saved!")
            st.rerun()
    
    with col2:
        if st.button("✅ Save & Next", type="primary", use_container_width=True):
            st.session_state.annotations[idx] = expert_score
            if notes:
                st.session_state.annotations[f"{idx}_notes"] = notes
            navigate('next')
            st.rerun()
    
    with col3:
        if st.button("⏭️ Skip", use_container_width=True):
            navigate('next')
            st.rerun()


def generate_instruction_json():
    """Generate JSON in instruction tuning format."""
    df = st.session_state.data
    annotations = st.session_state.annotations
    
    output = []
    for idx in sorted([k for k in annotations.keys() if isinstance(k, int)]):
        row = df.iloc[idx]
        expert_score = annotations[idx]
        
        instruction = (
            f"Output a number between 0 and 1 describing the semantic similarity "
            f"between the following two sentences:\n"
            f"Sentence 1: {row['sentence1']}\n"
            f"Sentence 2: {row['sentence2']}"
        )
        
        item = {
            "instruction": instruction,
            "input": "",
            "output": str(row['score']),
            "expert": str(expert_score)
        }
        
        # Add notes if present
        notes = annotations.get(f"{idx}_notes", "")
        if notes:
            item["notes"] = notes
        
        output.append(item)
    
    return output


def generate_csv():
    """Generate annotated CSV."""
    df = st.session_state.data.copy()
    annotations = st.session_state.annotations
    
    # Add expert annotation column
    df['expert_score'] = df.index.map(
        lambda x: annotations.get(x, None)
    )
    
    # Add notes column
    df['notes'] = df.index.map(
        lambda x: annotations.get(f"{x}_notes", "")
    )
    
    # Add annotator info
    df['annotator'] = st.session_state.annotator_name
    df['annotated'] = df['expert_score'].notna()
    
    return df


def render_export_buttons():
    """Render export buttons in sidebar."""
    if st.session_state.data is None or not st.session_state.annotations:
        st.info("No annotations to export yet")
        return
    
    # JSON export (instruction tuning format)
    json_data = generate_instruction_json()
    json_str = json.dumps(json_data, indent=2, ensure_ascii=False)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    st.download_button(
        label="📥 Download JSON",
        data=json_str,
        file_name=f"annotations_{st.session_state.file_name}_{timestamp}.json",
        mime="application/json",
        use_container_width=True
    )
    
    # CSV export
    csv_df = generate_csv()
    csv_buffer = StringIO()
    csv_df.to_csv(csv_buffer, index=False)
    
    st.download_button(
        label="📥 Download CSV",
        data=csv_buffer.getvalue(),
        file_name=f"annotations_{st.session_state.file_name}_{timestamp}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # Annotations backup (for resuming)
    annotations_backup = json.dumps(st.session_state.annotations, indent=2)
    
    st.download_button(
        label="💾 Backup Progress",
        data=annotations_backup,
        file_name=f"backup_{st.session_state.file_name}_{timestamp}.json",
        mime="application/json",
        use_container_width=True,
        help="Save your progress to resume later"
    )


def render_overview():
    """Render an overview of all annotations."""
    st.markdown("### 📊 Annotation Overview")
    
    df = st.session_state.data
    annotations = st.session_state.annotations
    
    # Create summary dataframe
    summary_data = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        is_annotated = idx in annotations
        
        summary_data.append({
            'Index': idx,
            'Sentence 1': row['sentence1'][:50] + '...' if len(row['sentence1']) > 50 else row['sentence1'],
            'Sentence 2': row['sentence2'][:50] + '...' if len(row['sentence2']) > 50 else row['sentence2'],
            'Original': row['score'],
            'Expert': annotations.get(idx, '-'),
            'Status': '✅' if is_annotated else '⏳'
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Display with filters
    status_filter = st.selectbox(
        "Filter by status",
        options=['All', 'Annotated ✅', 'Pending ⏳']
    )
    
    if status_filter == 'Annotated ✅':
        summary_df = summary_df[summary_df['Status'] == '✅']
    elif status_filter == 'Pending ⏳':
        summary_df = summary_df[summary_df['Status'] == '⏳']
    
    # Display table with click to navigate
    st.dataframe(
        summary_df,
        use_container_width=True,
        height=400
    )
    
    # Quick navigation
    nav_idx = st.number_input(
        "Click to navigate to index:",
        min_value=0,
        max_value=len(df) - 1,
        value=st.session_state.current_index
    )
    
    if st.button("Go to Selected"):
        st.session_state.current_index = nav_idx
        st.rerun()


def main():
    """Main application entry point."""
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📝 Semantic Similarity Annotator</h1>', unsafe_allow_html=True)
    
    # If data not loaded, show loading interface
    if not st.session_state.data_loaded:
        st.markdown("""
        Welcome to the Semantic Similarity Annotation Tool! 
        
        This tool helps you annotate sentence pairs with similarity scores for:
        - **Training data quality improvement**
        - **Expert annotation for instruction tuning**
        - **Dataset validation and correction**
        
        Choose a method to load your data:
        """)
        
        tab1, tab2, tab3 = st.tabs([
            "📁 Google Drive",
            "📤 Upload File",
            "📂 Resume Session"
        ])
        
        with tab1:
            load_from_gdrive()
        
        with tab2:
            load_from_upload()
        
        with tab3:
            st.markdown("""
            **To resume a previous session:**
            1. Load your original data file first
            2. Then upload your annotation backup file
            """)
            load_existing_annotations()
        
        # Sample data option
        st.markdown("---")
        if st.button("🧪 Try with Sample Data"):
            # Create sample data
            sample_data = pd.DataFrame({
                'sentence1': [
                    "A plane is taking off.",
                    "A man is playing a large flute.",
                    "Three men are playing chess.",
                ],
                'sentence2': [
                    "An air plane is taking off.",
                    "A man is playing a flute.",
                    "Two men are playing chess.",
                ],
                'score': [1.0, 0.76, 0.52]
            })
            st.session_state.data = sample_data
            st.session_state.data_loaded = True
            st.session_state.file_name = "sample"
            st.rerun()
    
    else:
        # Render sidebar
        render_sidebar()
        
        # Main content tabs
        tab1, tab2 = st.tabs(["✏️ Annotate", "📊 Overview"])
        
        with tab1:
            render_annotation_interface()
        
        with tab2:
            render_overview()


if __name__ == "__main__":
    main()
