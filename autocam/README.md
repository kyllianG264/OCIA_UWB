## AutoCam in This Repo

Lightweight import of the `autocam` proof of concept from Drive.

Included here:
- Streamlit app source code
- Clustering and feature extraction logic
- Minimal config files

Intentionally excluded:
- Raw videos
- YOLO `.pt` weights
- `.venv`
- `__pycache__`
- Generated previews and exports
- Client/NDA documents

Notes:
- The app is a local Streamlit tool for sorting pole-vault clips by athlete using pose extraction plus unsupervised clustering.
- Weight files are not committed here. `ultralytics` can download them on first use, or you can place compatible `.pt` files in `backend/`.

Quick start:

```powershell
cd third_party/autocam/backend
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
