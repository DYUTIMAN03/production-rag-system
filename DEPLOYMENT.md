# Public Deployment Guide

This guide explains how to deploy your "Ask My Docs" RAG system to [Render](https://render.com) for free! 

Since Render's free tier spins down on inactivity and uses ephemeral storage, we use a **Dockerfile** to run your ingestion script *during* the build phase. This means all the PDF/markdown files in `data/documents/` will be embedded directly into their own vector database permanently stored inside the container image, meaning it boots instantly with all your data accessible!

## 1. Push Code to GitHub
First, you need to push this local repository to GitHub. 

1. Create a new repository on [GitHub](https://github.com/new) (can be public or private).
2. Open your terminal in this repository and run:
   ```bash
   git add .
   git commit -m "Configure Dockerized Deployment"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```

*Note: The `.gitignore` is set up so it won't upload your API keys or local DB files to GitHub.*

## 2. Deploy on Render (Free)
1. Go to [Render.com](https://render.com/) and register/login with GitHub.
2. Click **New +** in the top right, and select **Web Service**.
3. Select "Build and deploy from a Git repository".
4. Connect the GitHub repository you just pushed.
5. In the configuration settings:
   - **Name**: `my-rag-system` (or any name you want)
   - **Region**: Any (e.g., US West)
   - **Branch**: `main`
   - **Environment**: Select **Docker** (very important!)
   - **Instance Type**: Select the **Free** tier
6. Under **Environment Variables**, click `Add Environment Variable`:
   - Key: `GOOGLE_API_KEY`
   - Value: `(Your Gemini API Key)`
7. Click **Create Web Service**.

## What happens next?
Render will do the following:
1. Pull your code from GitHub.
2. Run `docker build`, installing your python dependencies.
3. Your local PDFs/MD files (in `data/documents/`) will be ingested, and a new `chroma_db` and `bm25_index.pkl` will be baked into the image automatically!
4. Render will start the FastAPI server and wait for incoming requests.

Your system will be live on a public external URL like `https://my-rag-system.onrender.com`.

You can share this link with your interviewers, and they will immediately be able to query the documents via `https://my-rag-system.onrender.com/app`!
