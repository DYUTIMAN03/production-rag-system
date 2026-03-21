# Deployment Guide — Hugging Face Spaces (Free)

Deploy your RAG system for free on **Hugging Face Spaces** with Docker.
HF Spaces provides **16 GB RAM** and **2 vCPUs** — perfect for ML apps.

---

## 1. Push Code to GitHub

*(Skip if already done.)*

```bash
git add .
git commit -m "Ready for deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

## 2. Create a Hugging Face Space

1. Go to [huggingface.co](https://huggingface.co) and sign up / log in.
2. Click your profile picture (top right) → **New Space**.
3. Fill in the form:
   - **Space name**: `rag-system` (or any name you like)
   - **License**: MIT
   - **SDK**: Select **Docker**
   - **Space hardware**: **CPU basic (Free)**
   - **Visibility**: Public
4. Click **Create Space**.

## 3. Connect Your GitHub Repo

After the Space is created, you'll land on the Space page. Now link it to your GitHub repo:

### Option A — Push directly to HF (easiest)

```bash
# In your RAG project folder:
git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/rag-system
git push hf main
```

### Option B — Import from GitHub

On the Space page, click **Files** → **"Link a repository"** and paste your GitHub URL.

## 4. Add Your API Key

1. On your Space page, go to **Settings** → **Variables and secrets**.
2. Click **New secret**.
3. Name: `GROQ_API_KEY`  
   Value: *(your Groq API key)*
4. Click **Save**.

## 5. Wait for Build

Hugging Face will automatically:
1. Pull your code
2. Build the Docker image (install dependencies + ingest documents)
3. Start the FastAPI server on port 7860

The build takes ~5–10 minutes the first time. Once done, your Space will show **Running**.

## 6. Access Your Live App

Your app will be live at:

```
https://YOUR_HF_USERNAME-rag-system.hf.space
```

- Landing page: `https://YOUR_HF_USERNAME-rag-system.hf.space/`
- Main app: `https://YOUR_HF_USERNAME-rag-system.hf.space/app`
- API docs: `https://YOUR_HF_USERNAME-rag-system.hf.space/docs`

Share this link with your interviewers!

---

## Notes

- **Sleep on inactivity**: Free Spaces sleep after ~48 hours of inactivity. First visit after sleep takes ~2 minutes to wake up.
- **Your documents are baked in**: The Dockerfile runs your ingestion script during build, so all PDFs/documents are permanently embedded in the container.
- **No credit card required**: Everything is 100% free.
