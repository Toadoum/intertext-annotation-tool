# 👥 Team Semantic Similarity Annotation Tool

A Streamlit-based collaborative annotation tool for semantic similarity datasets. Designed for research teams creating high-quality training data for NLP models.

## 🌟 Features

### For Annotators
- **Clean Interface**: Slider + quick-select buttons for fast annotation
- **Progress Tracking**: See your personal progress and remaining items
- **Smart Navigation**: Filter by pending/completed, jump to specific items
- **Blind Annotation**: Original scores hidden by default to prevent bias

### For Teams
- **Real-time Collaboration**: Google Sheets backend syncs all annotations
- **Inter-Annotator Agreement**: Automatic calculation of annotation consistency
- **Team Dashboard**: Monitor progress across all annotators
- **Task Assignment**: Distribute work across team members

### Export Formats
- **CSV**: Full dataset with all annotator scores and consensus
- **JSON**: Instruction-tuning format for LLM fine-tuning

## 🚀 Deployment Options

### Option 1: Streamlit Cloud (Recommended for Teams)

**Free, always online, zero maintenance**

1. **Fork/Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/annotation-tool.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repo and `app_cloud.py`
   - Deploy!

3. **Configure Secrets**
   - In Streamlit Cloud dashboard → Your app → Settings → Secrets
   - Copy from `secrets.toml.template` and fill in your values

### Option 2: Docker (Self-hosted)

```bash
# Build and run
docker-compose up -d

# Access at http://localhost:8501
```

### Option 3: Local Development

```bash
pip install -r requirements.txt
streamlit run app_cloud.py
```

---

## 🔧 Setup Guide

### Step 1: Google Cloud Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select existing)
3. Enable APIs:
   - Google Sheets API
   - Google Drive API
4. Create Service Account:
   - IAM & Admin → Service Accounts → Create
   - Download JSON key file
5. Note the service account email (looks like: `xxx@project.iam.gserviceaccount.com`)

### Step 2: Google Sheets Setup

1. Create a new Google Sheet
2. Share it with your service account email (Editor access)
3. Copy the Spreadsheet ID from URL:
   ```
   https://docs.google.com/spreadsheets/d/SPREADSHEET_ID_HERE/edit
   ```

### Step 3: Configure the App

**For Streamlit Cloud** - Add to Secrets:

```toml
[auth]
team_password = "your_secure_password"

[team]
annotators = ["Sakayo", "Alice", "Bob", "Charlie"]

[google_sheets]
spreadsheet_id = "your-spreadsheet-id"

[gcp_service_account]
type = "service_account"
project_id = "your-project"
private_key_id = "key-id"
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "service-account@project.iam.gserviceaccount.com"
client_id = "123456789"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/..."
```

**For Local/Docker** - Create `.streamlit/secrets.toml` with same content

---

## 📊 Data Format

### Input CSV

| Column | Description |
|--------|-------------|
| `sentence1` | First sentence |
| `sentence2` | Second sentence |
| `score` | Original similarity score (0-1) |

```csv
sentence1,sentence2,score
A plane is taking off.,An air plane is taking off.,1.0
A man is playing a large flute.,A man is playing a flute.,0.76
```

### Output CSV

Includes original columns plus:
- One column per annotator with their scores
- `expert_consensus`: Mean of all expert scores
- Timestamp and metadata

### Output JSON (Instruction Tuning)

```json
[
  {
    "instruction": "Output a number between 0 and 1...\nSentence 1: ...\nSentence 2: ...",
    "input": "",
    "output": "1.0",
    "expert": "0.95"
  }
]
```

---

## 👥 Team Workflow

### For Team Leads

1. **Setup**: Deploy app and configure Google Sheets
2. **Upload**: Go to Upload tab, upload your CSV dataset
3. **Share**: Give team members the app URL and password
4. **Monitor**: Track progress in Dashboard tab
5. **Export**: Download merged annotations when complete

### For Annotators

1. **Login**: Select your name, enter team password
2. **Annotate**: 
   - Read both sentences
   - Use slider or quick buttons (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
   - Click "Save & Next"
3. **Track Progress**: Check sidebar for your completion %

### Annotation Guidelines

| Score | Meaning |
|-------|---------|
| 0.0 | Completely unrelated sentences |
| 0.2 | Slightly related topic |
| 0.4 | Related but different meaning |
| 0.6 | Similar meaning with differences |
| 0.8 | Very similar meaning |
| 1.0 | Identical or equivalent meaning |

---

## 📁 Project Structure

```
semantic_similarity_annotator/
├── app.py              # Solo annotation (simple version)
├── app_team.py         # Team annotation (full features)
├── app_cloud.py        # Cloud-optimized (Streamlit Cloud)
├── gdrive_utils.py     # Google Drive utilities
├── requirements.txt    # Dependencies
├── secrets.toml.template  # Secrets configuration template
├── Dockerfile          # Container deployment
├── docker-compose.yml  # Docker orchestration
└── README.md           # Documentation
```

### Which version to use?

| Version | Use Case |
|---------|----------|
| `app.py` | Solo annotation, local files |
| `app_team.py` | Self-hosted team annotation |
| `app_cloud.py` | Streamlit Cloud deployment |

---

## 🔒 Security Notes

- Change `team_password` to something secure
- Never commit `secrets.toml` to git (add to `.gitignore`)
- Service account has access only to shared files
- Consider IP restrictions for production

---

## 🐛 Troubleshooting

### "Connection failed"
- Check service account JSON is valid
- Verify Spreadsheet ID is correct
- Ensure sheet is shared with service account email

### "No data loaded"
- Team lead needs to upload data first
- Check Upload tab

### Annotations not syncing
- Click refresh or wait 30-60 seconds (caching)
- Check internet connection

### Slow performance with large datasets
- Split into multiple sheets (5000 items each)
- Use pagination in future versions

---

## 📝 License

MIT License - free to use and modify for your research.
