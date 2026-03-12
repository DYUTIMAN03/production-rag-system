#!/usr/bin/env python3
"""
ML/AI/LLM Documentation Fetcher
================================
Fetches technical documentation from public sources and saves clean Markdown.
Output: data/documents/<source>/<page>.md

Usage:
    python scripts/fetch_docs.py                    # Fetch ALL sources
    python scripts/fetch_docs.py --source huggingface_transformers
    python scripts/fetch_docs.py --source openai --max-pages 50
    python scripts/fetch_docs.py --list              # List all sources
    python scripts/fetch_docs.py --category providers # Fetch a category
"""

import argparse
import hashlib
import os
import re
import sys
import time
import json
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

# ─── Configuration ───────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "documents"
CACHE_DIR = PROJECT_ROOT / ".cache" / "docs"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; RAG-DocFetcher/1.0; educational-project)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

REQUEST_TIMEOUT = 30
RATE_LIMIT_DELAY = 1.0          # seconds between requests to same domain
MAX_PAGES_PER_SOURCE = 200      # default cap per source
MAX_RETRIES = 3
MIN_CONTENT_LENGTH = 200        # skip pages with less text
MAX_WORKERS = 4                 # parallel threads across different domains


# ─── Data Structures ────────────────────────────────────────────────────────

@dataclass
class DocSource:
    """A documentation source to scrape."""
    name: str                       # unique identifier (folder name)
    display_name: str               # human-readable name
    category: str                   # grouping category
    base_url: str                   # root URL of docs
    index_url: str                  # sitemap, index, or overview page
    url_pattern: str = ""           # regex to filter relevant URLs
    exclude_pattern: str = ""       # regex to exclude URLs
    content_selector: str = ""      # CSS selector for main content
    remove_selectors: List[str] = field(default_factory=list)  # CSS selectors to remove
    max_pages: int = MAX_PAGES_PER_SOURCE
    sitemap: bool = False           # if index_url is an XML sitemap
    crawl: bool = False             # if True, crawl links from index page


@dataclass
class FetchResult:
    """Result of fetching a single page."""
    url: str
    source: str
    title: str
    markdown: str
    filepath: str
    success: bool
    error: str = ""


# ─── Source Registry ─────────────────────────────────────────────────────────

SOURCES: Dict[str, DocSource] = {}

def register(src: DocSource):
    SOURCES[src.name] = src

# ────────────────────────────── FOUNDATIONS ──────────────────────────────────

register(DocSource(
    name="huggingface_transformers",
    display_name="HuggingFace Transformers",
    category="foundations",
    base_url="https://huggingface.co/docs/transformers",
    index_url="https://huggingface.co/docs/transformers/index",
    url_pattern=r"/docs/transformers/",
    exclude_pattern=r"(/_|/ja/|/zh/|/ko/|/es/|/fr/|/de/|/it/|/pt/|/ar/|changelog|CONTRIBUTING)",
    content_selector="article, .doc-content, main",
    remove_selectors=["nav", "header", "footer", ".sidebar", ".toc", "script", "style"],
    crawl=True, max_pages=150,
))

register(DocSource(
    name="huggingface_hub",
    display_name="HuggingFace Hub",
    category="foundations",
    base_url="https://huggingface.co/docs/hub",
    index_url="https://huggingface.co/docs/hub/index",
    url_pattern=r"/docs/hub/",
    exclude_pattern=r"(/_|/ja/|/zh/|/ko/|/es/|changelog)",
    content_selector="article, .doc-content, main",
    remove_selectors=["nav", "header", "footer", ".sidebar", ".toc", "script", "style"],
    crawl=True, max_pages=80,
))

register(DocSource(
    name="huggingface_datasets",
    display_name="HuggingFace Datasets",
    category="data",
    base_url="https://huggingface.co/docs/datasets",
    index_url="https://huggingface.co/docs/datasets/index",
    url_pattern=r"/docs/datasets/",
    exclude_pattern=r"(/_|/ja/|/zh/|/ko/|/es/|changelog)",
    content_selector="article, .doc-content, main",
    remove_selectors=["nav", "header", "footer", ".sidebar", ".toc", "script", "style"],
    crawl=True, max_pages=80,
))

register(DocSource(
    name="huggingface_evaluate",
    display_name="HuggingFace Evaluate",
    category="data",
    base_url="https://huggingface.co/docs/evaluate",
    index_url="https://huggingface.co/docs/evaluate/index",
    url_pattern=r"/docs/evaluate/",
    exclude_pattern=r"(/_|/ja/|/zh/|changelog)",
    content_selector="article, .doc-content, main",
    remove_selectors=["nav", "header", "footer", ".sidebar", ".toc", "script", "style"],
    crawl=True, max_pages=60,
))

register(DocSource(
    name="pytorch",
    display_name="PyTorch",
    category="foundations",
    base_url="https://pytorch.org/docs/stable",
    index_url="https://pytorch.org/docs/stable/index.html",
    url_pattern=r"/docs/stable/",
    exclude_pattern=r"(/_modules/|/generated/|_sources|/ja/|\.txt$)",
    content_selector="article, .document, main, .body",
    remove_selectors=["nav", "header", "footer", ".sidebar", ".toc", "script", "style", ".sphinxsidebar"],
    crawl=True, max_pages=120,
))

register(DocSource(
    name="tensorflow",
    display_name="TensorFlow",
    category="foundations",
    base_url="https://www.tensorflow.org",
    index_url="https://www.tensorflow.org/guide",
    url_pattern=r"tensorflow\.org/(guide|tutorials)/",
    exclude_pattern=r"(/api_docs/|/versions/|/install|/js/|/lite/|/tfx/|\.ipynb)",
    content_selector="article, .devsite-article-body, main",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style", ".devsite-nav", "devsite-toc"],
    crawl=True, max_pages=100,
))

register(DocSource(
    name="keras",
    display_name="Keras",
    category="foundations",
    base_url="https://keras.io",
    index_url="https://keras.io/guides/",
    url_pattern=r"keras\.io/(guides|api)/",
    exclude_pattern=r"(/examples/|\.py$|\.ipynb)",
    content_selector="article, .k-content, main, .content",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=80,
))

# ────────────────────────────── TRAINING ────────────────────────────────────

register(DocSource(
    name="huggingface_peft",
    display_name="HuggingFace PEFT",
    category="training",
    base_url="https://huggingface.co/docs/peft",
    index_url="https://huggingface.co/docs/peft/index",
    url_pattern=r"/docs/peft/",
    exclude_pattern=r"(/_|/ja/|/zh/|changelog)",
    content_selector="article, .doc-content, main",
    remove_selectors=["nav", "header", "footer", ".sidebar", ".toc", "script", "style"],
    crawl=True, max_pages=80,
))

register(DocSource(
    name="huggingface_trl",
    display_name="HuggingFace TRL (RLHF)",
    category="training",
    base_url="https://huggingface.co/docs/trl",
    index_url="https://huggingface.co/docs/trl/index",
    url_pattern=r"/docs/trl/",
    exclude_pattern=r"(/_|/ja/|/zh/|changelog)",
    content_selector="article, .doc-content, main",
    remove_selectors=["nav", "header", "footer", ".sidebar", ".toc", "script", "style"],
    crawl=True, max_pages=60,
))

register(DocSource(
    name="huggingface_accelerate",
    display_name="HuggingFace Accelerate",
    category="training",
    base_url="https://huggingface.co/docs/accelerate",
    index_url="https://huggingface.co/docs/accelerate/index",
    url_pattern=r"/docs/accelerate/",
    exclude_pattern=r"(/_|/ja/|/zh/|changelog)",
    content_selector="article, .doc-content, main",
    remove_selectors=["nav", "header", "footer", ".sidebar", ".toc", "script", "style"],
    crawl=True, max_pages=60,
))

register(DocSource(
    name="pytorch_lightning",
    display_name="PyTorch Lightning",
    category="training",
    base_url="https://lightning.ai/docs/pytorch/stable",
    index_url="https://lightning.ai/docs/pytorch/stable/",
    url_pattern=r"lightning\.ai/docs/pytorch/stable/",
    exclude_pattern=r"(_modules/|_sources|\.txt$|/api/|changelog)",
    content_selector="article, .document, main, .body",
    remove_selectors=["nav", "header", "footer", ".sidebar", ".toc", "script", "style"],
    crawl=True, max_pages=80,
))

register(DocSource(
    name="deepspeed",
    display_name="DeepSpeed",
    category="training",
    base_url="https://www.deepspeed.ai",
    index_url="https://www.deepspeed.ai/tutorials/",
    url_pattern=r"deepspeed\.ai/(tutorials|training|getting-started|docs)/",
    exclude_pattern=r"(changelog|/ja/|/zh/)",
    content_selector="article, .post-content, main, .page-content",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=50,
))

register(DocSource(
    name="wandb",
    display_name="Weights & Biases",
    category="training",
    base_url="https://docs.wandb.ai",
    index_url="https://docs.wandb.ai/guides",
    url_pattern=r"docs\.wandb\.ai/guides/",
    exclude_pattern=r"(changelog|/ja/|/zh/|/ref/)",
    content_selector="article, main, .theme-doc-markdown",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style", ".table-of-contents"],
    crawl=True, max_pages=80,
))

register(DocSource(
    name="mlflow",
    display_name="MLflow",
    category="training",
    base_url="https://mlflow.org/docs/latest",
    index_url="https://mlflow.org/docs/latest/index.html",
    url_pattern=r"mlflow\.org/docs/latest/",
    exclude_pattern=r"(_modules/|_sources|\.txt$|/python_api/|changelog)",
    content_selector="article, .document, main, .body",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style", ".sphinxsidebar"],
    crawl=True, max_pages=60,
))

# ────────────────────────────── RAG & SEARCH ────────────────────────────────

register(DocSource(
    name="langchain",
    display_name="LangChain",
    category="rag_and_search",
    base_url="https://python.langchain.com/docs",
    index_url="https://python.langchain.com/docs/introduction/",
    url_pattern=r"python\.langchain\.com/docs/",
    exclude_pattern=r"(/api/|/versions/|changelog|/integrations/.*/.*/)",
    content_selector="article, main, .theme-doc-markdown",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style", ".table-of-contents"],
    crawl=True, max_pages=120,
))

register(DocSource(
    name="langgraph",
    display_name="LangGraph",
    category="agents",
    base_url="https://langchain-ai.github.io/langgraph",
    index_url="https://langchain-ai.github.io/langgraph/",
    url_pattern=r"langchain-ai\.github\.io/langgraph/",
    exclude_pattern=r"(/api/|changelog|\.ipynb)",
    content_selector="article, main, .md-content",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style", ".md-nav"],
    crawl=True, max_pages=80,
))

register(DocSource(
    name="llamaindex",
    display_name="LlamaIndex",
    category="rag_and_search",
    base_url="https://docs.llamaindex.ai/en/stable",
    index_url="https://docs.llamaindex.ai/en/stable/",
    url_pattern=r"docs\.llamaindex\.ai/en/stable/",
    exclude_pattern=r"(/api/|changelog|\.ipynb)",
    content_selector="article, main, .md-content",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style", ".md-nav"],
    crawl=True, max_pages=100,
))

register(DocSource(
    name="haystack",
    display_name="Haystack",
    category="rag_and_search",
    base_url="https://docs.haystack.deepset.ai/docs",
    index_url="https://docs.haystack.deepset.ai/docs/intro",
    url_pattern=r"docs\.haystack\.deepset\.ai/docs/",
    exclude_pattern=r"(/api/|changelog|/v1\./)",
    content_selector="article, main, .content",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=80,
))

register(DocSource(
    name="ragas",
    display_name="RAGAS",
    category="rag_and_search",
    base_url="https://docs.ragas.io/en/stable",
    index_url="https://docs.ragas.io/en/stable/",
    url_pattern=r"docs\.ragas\.io/en/stable/",
    exclude_pattern=r"(/_|changelog|/api/)",
    content_selector="article, main, .md-content, .document",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=50,
))

register(DocSource(
    name="deepeval",
    display_name="DeepEval",
    category="rag_and_search",
    base_url="https://docs.confident-ai.com",
    index_url="https://docs.confident-ai.com/docs/getting-started",
    url_pattern=r"docs\.confident-ai\.com/docs/",
    exclude_pattern=r"(changelog)",
    content_selector="article, main, .theme-doc-markdown",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=60,
))

register(DocSource(
    name="arize_phoenix",
    display_name="Arize Phoenix",
    category="rag_and_search",
    base_url="https://docs.arize.com/phoenix",
    index_url="https://docs.arize.com/phoenix",
    url_pattern=r"docs\.arize\.com/phoenix",
    exclude_pattern=r"(changelog|/api-reference/)",
    content_selector="article, main, .content",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=60,
))

register(DocSource(
    name="langsmith",
    display_name="LangSmith",
    category="rag_and_search",
    base_url="https://docs.smith.langchain.com",
    index_url="https://docs.smith.langchain.com/",
    url_pattern=r"docs\.smith\.langchain\.com/",
    exclude_pattern=r"(/api/|changelog)",
    content_selector="article, main, .theme-doc-markdown",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=60,
))

# ────────────────────────────── PROVIDERS ────────────────────────────────────

register(DocSource(
    name="openai",
    display_name="OpenAI",
    category="providers",
    base_url="https://platform.openai.com/docs",
    index_url="https://platform.openai.com/docs/overview",
    url_pattern=r"platform\.openai\.com/docs/",
    exclude_pattern=r"(/api-reference/|changelog|/plugins/)",
    content_selector="article, main, .docs-content, .content",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=80,
))

register(DocSource(
    name="anthropic",
    display_name="Anthropic Claude",
    category="providers",
    base_url="https://docs.anthropic.com",
    index_url="https://docs.anthropic.com/en/docs/welcome",
    url_pattern=r"docs\.anthropic\.com/en/docs/",
    exclude_pattern=r"(/api-reference/|changelog)",
    content_selector="article, main, .content",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=80,
))

register(DocSource(
    name="google_gemini",
    display_name="Google Gemini",
    category="providers",
    base_url="https://ai.google.dev/gemini-api/docs",
    index_url="https://ai.google.dev/gemini-api/docs",
    url_pattern=r"ai\.google\.dev/gemini-api/docs",
    exclude_pattern=r"(/api/|changelog|/rest/)",
    content_selector="article, main, .devsite-article-body",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style", "devsite-toc"],
    crawl=True, max_pages=60,
))

register(DocSource(
    name="groq",
    display_name="Groq",
    category="providers",
    base_url="https://console.groq.com/docs",
    index_url="https://console.groq.com/docs/overview",
    url_pattern=r"console\.groq\.com/docs/",
    exclude_pattern=r"(changelog)",
    content_selector="article, main, .content",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=40,
))

register(DocSource(
    name="mistral",
    display_name="Mistral AI",
    category="providers",
    base_url="https://docs.mistral.ai",
    index_url="https://docs.mistral.ai/",
    url_pattern=r"docs\.mistral\.ai/",
    exclude_pattern=r"(/api/|changelog)",
    content_selector="article, main, .md-content",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style", ".md-nav"],
    crawl=True, max_pages=60,
))

register(DocSource(
    name="cohere",
    display_name="Cohere",
    category="providers",
    base_url="https://docs.cohere.com/docs",
    index_url="https://docs.cohere.com/docs/the-cohere-platform",
    url_pattern=r"docs\.cohere\.com/docs/",
    exclude_pattern=r"(/api-reference/|changelog|/v1/)",
    content_selector="article, main, .content",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=60,
))

register(DocSource(
    name="together_ai",
    display_name="Together AI",
    category="providers",
    base_url="https://docs.together.ai",
    index_url="https://docs.together.ai/docs/introduction",
    url_pattern=r"docs\.together\.ai/docs/",
    exclude_pattern=r"(/api-reference/|changelog)",
    content_selector="article, main, .content",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=50,
))

register(DocSource(
    name="fireworks_ai",
    display_name="Fireworks AI",
    category="providers",
    base_url="https://docs.fireworks.ai",
    index_url="https://docs.fireworks.ai/getting-started/introduction",
    url_pattern=r"docs\.fireworks\.ai/",
    exclude_pattern=r"(/api-reference/|changelog)",
    content_selector="article, main, .content",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=50,
))

register(DocSource(
    name="replicate",
    display_name="Replicate",
    category="providers",
    base_url="https://replicate.com/docs",
    index_url="https://replicate.com/docs",
    url_pattern=r"replicate\.com/docs/",
    exclude_pattern=r"(/api/|changelog)",
    content_selector="article, main, .content",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=40,
))

# ────────────────────────────── VECTOR STORES ───────────────────────────────

register(DocSource(
    name="chromadb",
    display_name="ChromaDB",
    category="vector_stores",
    base_url="https://docs.trychroma.com",
    index_url="https://docs.trychroma.com/docs/overview/introduction",
    url_pattern=r"docs\.trychroma\.com/docs/",
    exclude_pattern=r"(changelog|/api/)",
    content_selector="article, main, .theme-doc-markdown",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=50,
))

register(DocSource(
    name="qdrant",
    display_name="Qdrant",
    category="vector_stores",
    base_url="https://qdrant.tech/documentation",
    index_url="https://qdrant.tech/documentation/",
    url_pattern=r"qdrant\.tech/documentation/",
    exclude_pattern=r"(changelog|/api-reference/)",
    content_selector="article, main, .content",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=60,
))

register(DocSource(
    name="weaviate",
    display_name="Weaviate",
    category="vector_stores",
    base_url="https://weaviate.io/developers/weaviate",
    index_url="https://weaviate.io/developers/weaviate",
    url_pattern=r"weaviate\.io/developers/weaviate",
    exclude_pattern=r"(changelog|/api/)",
    content_selector="article, main, .theme-doc-markdown",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=60,
))

register(DocSource(
    name="pinecone",
    display_name="Pinecone",
    category="vector_stores",
    base_url="https://docs.pinecone.io/guides",
    index_url="https://docs.pinecone.io/guides/get-started/overview",
    url_pattern=r"docs\.pinecone\.io/(guides|reference)/",
    exclude_pattern=r"(changelog)",
    content_selector="article, main, .content",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=50,
))

register(DocSource(
    name="milvus",
    display_name="Milvus",
    category="vector_stores",
    base_url="https://milvus.io/docs",
    index_url="https://milvus.io/docs",
    url_pattern=r"milvus\.io/docs/",
    exclude_pattern=r"(changelog|/api-reference/)",
    content_selector="article, main, .doc-content, .content",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=60,
))

# ────────────────────────────── INFERENCE ───────────────────────────────────

register(DocSource(
    name="vllm",
    display_name="vLLM",
    category="inference",
    base_url="https://docs.vllm.ai/en/stable",
    index_url="https://docs.vllm.ai/en/stable/",
    url_pattern=r"docs\.vllm\.ai/en/stable/",
    exclude_pattern=r"(_modules/|_sources|changelog|/api/)",
    content_selector="article, main, .document, .body",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=50,
))

register(DocSource(
    name="ollama",
    display_name="Ollama",
    category="inference",
    base_url="https://github.com/ollama/ollama/blob/main",
    index_url="https://github.com/ollama/ollama/blob/main/README.md",
    url_pattern=r"github\.com/ollama/ollama/blob/main/docs/",
    exclude_pattern=r"(changelog)",
    content_selector="article, main, .markdown-body",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=30,
))

register(DocSource(
    name="litellm",
    display_name="LiteLLM",
    category="inference",
    base_url="https://docs.litellm.ai",
    index_url="https://docs.litellm.ai/docs/",
    url_pattern=r"docs\.litellm\.ai/docs/",
    exclude_pattern=r"(changelog|/api/)",
    content_selector="article, main, .theme-doc-markdown",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=60,
))

register(DocSource(
    name="bentoml",
    display_name="BentoML",
    category="inference",
    base_url="https://docs.bentoml.com/en/latest",
    index_url="https://docs.bentoml.com/en/latest/",
    url_pattern=r"docs\.bentoml\.com/en/latest/",
    exclude_pattern=r"(_modules/|_sources|changelog|/api/)",
    content_selector="article, main, .document, .body",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=50,
))

# ────────────────────────────── AGENTS ──────────────────────────────────────

register(DocSource(
    name="autogen",
    display_name="AutoGen",
    category="agents",
    base_url="https://microsoft.github.io/autogen/stable",
    index_url="https://microsoft.github.io/autogen/stable/",
    url_pattern=r"microsoft\.github\.io/autogen/stable/",
    exclude_pattern=r"(_modules/|changelog|/api/)",
    content_selector="article, main, .md-content",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=50,
))

register(DocSource(
    name="crewai",
    display_name="CrewAI",
    category="agents",
    base_url="https://docs.crewai.com",
    index_url="https://docs.crewai.com/introduction",
    url_pattern=r"docs\.crewai\.com/",
    exclude_pattern=r"(changelog|/api/)",
    content_selector="article, main, .content",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=50,
))

# ────────────────────────────── PROMPTS ─────────────────────────────────────

register(DocSource(
    name="dspy",
    display_name="DSPy",
    category="prompts",
    base_url="https://dspy.ai",
    index_url="https://dspy.ai/learn/",
    url_pattern=r"dspy\.ai/(learn|tutorials|api)/",
    exclude_pattern=r"(changelog)",
    content_selector="article, main, .md-content",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=50,
))

register(DocSource(
    name="instructor",
    display_name="Instructor",
    category="prompts",
    base_url="https://python.useinstructor.com",
    index_url="https://python.useinstructor.com/",
    url_pattern=r"python\.useinstructor\.com/",
    exclude_pattern=r"(changelog|/api/)",
    content_selector="article, main, .md-content",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=50,
))

# ────────────────────────────── SERVING ─────────────────────────────────────

register(DocSource(
    name="fastapi",
    display_name="FastAPI",
    category="serving",
    base_url="https://fastapi.tiangolo.com",
    index_url="https://fastapi.tiangolo.com/tutorial/",
    url_pattern=r"fastapi\.tiangolo\.com/(tutorial|advanced|how-to)/",
    exclude_pattern=r"(/ja/|/zh/|/ko/|/es/|/fr/|/de/|/pt/|/ru/|/it/|/tr/|/vi/|/pl/|/uk/|changelog)",
    content_selector="article, main, .md-content",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=60,
))

register(DocSource(
    name="pydantic",
    display_name="Pydantic",
    category="serving",
    base_url="https://docs.pydantic.dev/latest",
    index_url="https://docs.pydantic.dev/latest/",
    url_pattern=r"docs\.pydantic\.dev/latest/(concepts|why)/",
    exclude_pattern=r"(/api/|changelog)",
    content_selector="article, main, .md-content",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=40,
))

# ────────────────────────────── SAFETY ──────────────────────────────────────

register(DocSource(
    name="guardrails_ai",
    display_name="Guardrails AI",
    category="safety",
    base_url="https://docs.guardrailsai.com",
    index_url="https://docs.guardrailsai.com/getting_started/",
    url_pattern=r"docs\.guardrailsai\.com/",
    exclude_pattern=r"(changelog|/api/)",
    content_selector="article, main, .md-content",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=40,
))

register(DocSource(
    name="nemo_guardrails",
    display_name="NeMo Guardrails",
    category="safety",
    base_url="https://docs.nvidia.com/nemo/guardrails",
    index_url="https://docs.nvidia.com/nemo/guardrails/",
    url_pattern=r"docs\.nvidia\.com/nemo/guardrails/",
    exclude_pattern=r"(changelog|/api/)",
    content_selector="article, main, .content",
    remove_selectors=["nav", "header", "footer", ".sidebar", "script", "style"],
    crawl=True, max_pages=40,
))


# ─── HTML to Markdown Converter ──────────────────────────────────────────────

def html_to_markdown(element: Tag) -> str:
    """Convert an HTML element to clean Markdown text."""
    lines = []
    _walk(element, lines, indent=0)
    text = "\n".join(lines)
    # Clean up excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _walk(el, lines, indent=0):
    """Recursively walk HTML nodes and produce markdown lines."""
    if isinstance(el, NavigableString):
        text = str(el)
        if text.strip():
            lines.append(text.strip())
        return

    if not isinstance(el, Tag):
        return

    tag = el.name

    if tag in ("script", "style", "nav", "footer", "header", "noscript", "svg", "iframe"):
        return

    # Headings
    if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
        level = int(tag[1])
        text = el.get_text(strip=True)
        if text:
            lines.append("")
            lines.append(f"{'#' * level} {text}")
            lines.append("")
        return

    # Paragraphs
    if tag == "p":
        text = _inline_text(el)
        if text:
            lines.append("")
            lines.append(text)
        return

    # Code blocks
    if tag == "pre":
        code_el = el.find("code")
        if code_el:
            lang = ""
            classes = code_el.get("class", [])
            for c in (classes if isinstance(classes, list) else [classes]):
                if c and ("language-" in str(c) or "lang-" in str(c)):
                    lang = str(c).replace("language-", "").replace("lang-", "").split()[0]
                    break
            lines.append("")
            lines.append(f"```{lang}")
            lines.append(code_el.get_text())
            lines.append("```")
            lines.append("")
        else:
            lines.append("")
            lines.append(f"```")
            lines.append(el.get_text())
            lines.append("```")
            lines.append("")
        return

    # Inline code
    if tag == "code" and el.parent and el.parent.name != "pre":
        text = el.get_text(strip=True)
        if text:
            lines.append(f"`{text}`")
        return

    # Lists
    if tag in ("ul", "ol"):
        lines.append("")
        for i, li in enumerate(el.find_all("li", recursive=False)):
            prefix = f"{i+1}." if tag == "ol" else "-"
            text = _inline_text(li)
            if text:
                lines.append(f"{' ' * indent}{prefix} {text}")
            # Nested lists
            for nested in li.find_all(["ul", "ol"], recursive=False):
                _walk(nested, lines, indent=indent + 2)
        lines.append("")
        return

    # Tables
    if tag == "table":
        _convert_table(el, lines)
        return

    # Blockquotes
    if tag == "blockquote":
        text = el.get_text(strip=True)
        if text:
            lines.append("")
            for line in text.split("\n"):
                lines.append(f"> {line.strip()}")
            lines.append("")
        return

    # Links
    if tag == "a":
        text = el.get_text(strip=True)
        href = el.get("href", "")
        if text and href and not href.startswith("#"):
            lines.append(f"[{text}]({href})")
        elif text:
            lines.append(text)
        return

    # Bold / Italic
    if tag in ("strong", "b"):
        text = el.get_text(strip=True)
        if text:
            lines.append(f"**{text}**")
        return
    if tag in ("em", "i"):
        text = el.get_text(strip=True)
        if text:
            lines.append(f"*{text}*")
        return

    # Divs and other containers — recurse
    for child in el.children:
        _walk(child, lines, indent)


def _inline_text(el) -> str:
    """Get inline markdown text from an element."""
    parts = []
    for child in el.children:
        if isinstance(child, NavigableString):
            parts.append(str(child).strip())
        elif isinstance(child, Tag):
            if child.name == "code":
                parts.append(f"`{child.get_text(strip=True)}`")
            elif child.name in ("strong", "b"):
                parts.append(f"**{child.get_text(strip=True)}**")
            elif child.name in ("em", "i"):
                parts.append(f"*{child.get_text(strip=True)}*")
            elif child.name == "a":
                text = child.get_text(strip=True)
                href = child.get("href", "")
                if text and href and not href.startswith("#"):
                    parts.append(f"[{text}]({href})")
                elif text:
                    parts.append(text)
            elif child.name == "br":
                parts.append("\n")
            else:
                parts.append(child.get_text(strip=True))
    return " ".join(p for p in parts if p)


def _convert_table(table: Tag, lines: list):
    """Convert an HTML table to markdown."""
    rows = table.find_all("tr")
    if not rows:
        return

    lines.append("")
    for i, row in enumerate(rows):
        cells = row.find_all(["th", "td"])
        row_text = "| " + " | ".join(c.get_text(strip=True).replace("|", "\\|") for c in cells) + " |"
        lines.append(row_text)
        if i == 0:
            lines.append("| " + " | ".join("---" for _ in cells) + " |")
    lines.append("")


# ─── Core Fetcher Logic ─────────────────────────────────────────────────────

class DocFetcher:
    """Fetches and converts documentation pages to Markdown."""

    def __init__(self, output_dir: Path = OUTPUT_DIR, cache_dir: Path = CACHE_DIR):
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self._domain_timestamps: Dict[str, float] = {}
        self.stats = {"fetched": 0, "cached": 0, "skipped": 0, "errors": 0}

    def _rate_limit(self, url: str):
        """Enforce rate limiting per domain."""
        domain = urllib.parse.urlparse(url).netloc
        last = self._domain_timestamps.get(domain, 0)
        elapsed = time.time() - last
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self._domain_timestamps[domain] = time.time()

    def _get_cache_path(self, url: str) -> Path:
        """Get cache file path for a URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"{url_hash}.html"

    def _fetch_html(self, url: str) -> Optional[str]:
        """Fetch HTML content with caching and retries."""
        cache_path = self._get_cache_path(url)

        # Check cache (24h TTL)
        if cache_path.exists():
            age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
            if age_hours < 24:
                self.stats["cached"] += 1
                return cache_path.read_text(encoding="utf-8", errors="replace")

        self._rate_limit(url)

        for attempt in range(MAX_RETRIES):
            try:
                resp = self.session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
                if resp.status_code == 429:
                    wait = min(2 ** attempt * 5, 60)
                    print(f"    [!] Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()

                html = resp.text
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(html, encoding="utf-8")
                self.stats["fetched"] += 1
                return html

            except requests.exceptions.RequestException as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                else:
                    self.stats["errors"] += 1
                    print(f"    [FAIL] {url} - {e}")
                    return None

        return None

    def _extract_content(self, html: str, source: DocSource) -> Tuple[str, str]:
        """Extract title and main content markdown from HTML."""
        soup = BeautifulSoup(html, "html.parser")

        # Extract title
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
        if not title:
            h1 = soup.find("h1")
            if h1:
                title = h1.get_text(strip=True)

        # Remove unwanted elements — expanded list for better noise removal
        remove_sels = source.remove_selectors or ["nav", "header", "footer", "script", "style"]
        # Always remove these regardless of source config
        always_remove = [
            "nav", "script", "style", "noscript", "iframe", "svg",
            ".sidebar", ".toc", ".table-of-contents", ".breadcrumb",
            ".pagination", ".edit-page", ".page-nav", ".prev-next",
            "[role='navigation']", "[role='banner']", "[role='contentinfo']",
            ".feedback", ".thumbs-up", ".thumbs-down",
        ]
        all_sels = list(set(remove_sels + always_remove))
        for sel in all_sels:
            try:
                for el in soup.select(sel):
                    el.decompose()
            except Exception:
                pass  # skip invalid selectors

        # Find main content
        content = None
        if source.content_selector:
            for selector in source.content_selector.split(","):
                selector = selector.strip()
                content = soup.select_one(selector)
                if content:
                    break

        if not content:
            content = soup.find("body") or soup

        markdown = html_to_markdown(content)
        markdown = self._clean_markdown(markdown)
        return title, markdown

    @staticmethod
    def _clean_markdown(text: str) -> str:
        """Post-process markdown to remove navigation noise and boilerplate."""
        lines = text.split("\n")
        cleaned = []

        # Patterns to skip entirely
        skip_patterns = [
            r"^\s*Skip to (main |)content\s*$",
            r"^\s*(Was this page helpful|Suggest edits|Report incorrect)",
            r"^\s*(Yes|No)\s*$",
            r"^\s*On this page\s*$",
            r"^\s*Search\.\.\.\s*$",
            r"^\s*Ask AI\s*$",
            r"^\s*(Copy|Report incorrect code)\s*$",
            r"^\s*Navigation\s*$",
            r"^\s*Previous\s*$",
            r"^\s*Next\s*$",
            r"^\s*(Contact support|Responses are generated using AI)",
            r"^\s*\$\s*$",        # stray $ signs
            r"^\s*/\$\s*$",
            r"^\s*Assistant\s*$",
        ]

        for line in lines:
            # Skip boilerplate lines
            if any(re.match(p, line, re.IGNORECASE) for p in skip_patterns):
                continue
            # Skip keyboard shortcut indicators
            if re.match(r"^\s*[A-Z]\s*$", line):
                continue
            cleaned.append(line)

        text = "\n".join(cleaned)
        # Clean up excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _discover_urls(self, source: DocSource) -> List[str]:
        """Discover documentation URLs from a source."""
        urls = set()

        html = self._fetch_html(source.index_url)
        if not html:
            print(f"  [ERR] Could not fetch index: {source.index_url}")
            return []

        if source.sitemap:
            # Parse XML sitemap
            soup = BeautifulSoup(html, "html.parser")
            for loc in soup.find_all("loc"):
                url = loc.get_text(strip=True)
                if self._url_matches(url, source):
                    urls.add(url)
        elif source.crawl:
            # Crawl links from the index page
            soup = BeautifulSoup(html, "html.parser")
            urls.add(source.index_url)

            for a in soup.find_all("a", href=True):
                href = a["href"]
                # Resolve relative URLs
                full_url = urllib.parse.urljoin(source.index_url, href)
                # Remove fragments
                full_url = full_url.split("#")[0]
                # Remove trailing slashes for consistency
                full_url = full_url.rstrip("/")
                if full_url and self._url_matches(full_url, source):
                    urls.add(full_url)

            # Second pass: crawl discovered pages for more links (depth 2)
            second_pass = set()
            for url in list(urls)[:30]:  # limit second-pass crawling
                html2 = self._fetch_html(url)
                if html2:
                    soup2 = BeautifulSoup(html2, "html.parser")
                    for a in soup2.find_all("a", href=True):
                        href = a["href"]
                        full_url = urllib.parse.urljoin(url, href)
                        full_url = full_url.split("#")[0].rstrip("/")
                        if full_url and self._url_matches(full_url, source):
                            second_pass.add(full_url)

            urls.update(second_pass)

        filtered = sorted(urls)[:source.max_pages]
        return filtered

    def _url_matches(self, url: str, source: DocSource) -> bool:
        """Check if a URL matches the source's url_pattern and doesn't match exclude_pattern."""
        if source.url_pattern and not re.search(source.url_pattern, url):
            return False
        if source.exclude_pattern and re.search(source.exclude_pattern, url):
            return False
        # Must be HTTP(S)
        if not url.startswith("http"):
            return False
        # Skip common non-doc pages
        if any(ext in url for ext in [".png", ".jpg", ".gif", ".svg", ".css", ".js", ".ico", ".pdf", ".zip"]):
            return False
        return True

    def _url_to_filename(self, url: str, source: DocSource) -> str:
        """Convert a URL to a safe filename."""
        parsed = urllib.parse.urlparse(url)
        path = parsed.path.strip("/")
        # Remove common prefixes
        for prefix in ["docs/", "en/stable/", "en/latest/", "docs/stable/", "latest/"]:
            if path.startswith(prefix):
                path = path[len(prefix):]
        # Remove extension
        path = re.sub(r"\.(html?|md)$", "", path)
        # Replace slashes and special chars with underscores
        safe = re.sub(r"[^\w\-]", "_", path)
        safe = re.sub(r"_+", "_", safe).strip("_")
        if not safe:
            safe = "index"
        return safe[:120] + ".md"

    def fetch_source(self, source: DocSource) -> List[FetchResult]:
        """Fetch all pages from a documentation source."""
        print(f"\n{'='*60}")
        print(f"[DOC] {source.display_name} ({source.category})")
        print(f"      {source.base_url}")
        print(f"{'='*60}")

        # Discover URLs
        print(f"  [*] Discovering pages...")
        urls = self._discover_urls(source)
        print(f"  [i] Found {len(urls)} pages")

        if not urls:
            return []

        results = []
        source_dir = self.output_dir / source.name
        source_dir.mkdir(parents=True, exist_ok=True)

        for i, url in enumerate(urls):
            print(f"  [{i+1}/{len(urls)}] {url[:80]}...", end=" ")

            html = self._fetch_html(url)
            if not html:
                print("[SKIP]")
                self.stats["skipped"] += 1
                continue

            title, markdown = self._extract_content(html, source)

            if len(markdown) < MIN_CONTENT_LENGTH:
                print("[SHORT]")
                self.stats["skipped"] += 1
                continue

            filename = self._url_to_filename(url, source)
            filepath = source_dir / filename

            # Add metadata header
            full_content = f"# {title}\n\n" if title else ""
            full_content += f"> Source: {url}\n\n"
            full_content += markdown

            filepath.write_text(full_content, encoding="utf-8")
            print(f"[OK] {filename} ({len(markdown)} chars)")

            results.append(FetchResult(
                url=url, source=source.name, title=title,
                markdown=markdown, filepath=str(filepath), success=True,
            ))

        return results

    def fetch_all(self, sources: Optional[List[str]] = None,
                  category: Optional[str] = None,
                  max_pages: Optional[int] = None) -> Dict[str, List[FetchResult]]:
        """Fetch documentation from all (or filtered) sources."""
        targets = SOURCES

        if sources:
            targets = {k: v for k, v in SOURCES.items() if k in sources}
        elif category:
            targets = {k: v for k, v in SOURCES.items() if v.category == category}

        if max_pages:
            for src in targets.values():
                src.max_pages = min(src.max_pages, max_pages)

        all_results = {}
        total_sources = len(targets)

        print(f"\n{'#'*60}")
        print(f"  ML/AI/LLM Documentation Fetcher")
        print(f"  Sources: {total_sources}")
        print(f"  Output:  {self.output_dir}")
        print(f"{'#'*60}")

        for i, (name, source) in enumerate(targets.items()):
            print(f"\n[{i+1}/{total_sources}] Processing {source.display_name}...")
            try:
                results = self.fetch_source(source)
                all_results[name] = results
            except Exception as e:
                print(f"  [ERR] Error processing {name}: {e}")
                all_results[name] = []

        # Print summary
        total_pages = sum(len(r) for r in all_results.values())
        print(f"\n{'#'*60}")
        print(f"  FETCH COMPLETE")
        print(f"  Total pages saved:   {total_pages}")
        print(f"  HTTP fetches:        {self.stats['fetched']}")
        print(f"  From cache:          {self.stats['cached']}")
        print(f"  Skipped:             {self.stats['skipped']}")
        print(f"  Errors:              {self.stats['errors']}")
        print(f"{'#'*60}")

        # Save manifest
        manifest = {
            name: [{
                "url": r.url,
                "title": r.title,
                "filepath": r.filepath,
                "chars": len(r.markdown),
            } for r in results]
            for name, results in all_results.items()
        }
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"\n  Manifest: {manifest_path}")

        return all_results


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch ML/AI/LLM documentation to Markdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/fetch_docs.py                          # Fetch ALL sources
  python scripts/fetch_docs.py --source langchain       # Single source
  python scripts/fetch_docs.py --category providers     # All providers
  python scripts/fetch_docs.py --list                   # List sources
  python scripts/fetch_docs.py --max-pages 20           # Limit pages
        """,
    )
    parser.add_argument("--source", "-s", action="append",
                        help="Source name(s) to fetch (repeatable)")
    parser.add_argument("--category", "-c",
                        help="Fetch all sources in a category")
    parser.add_argument("--max-pages", "-m", type=int,
                        help="Max pages per source")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List all available sources")
    parser.add_argument("--output", "-o", default=str(OUTPUT_DIR),
                        help="Output directory")

    args = parser.parse_args()

    if args.list:
        categories = {}
        for name, src in sorted(SOURCES.items()):
            categories.setdefault(src.category, []).append(src)

        print(f"\n{'='*60}")
        print(f"  Available Sources ({len(SOURCES)} total)")
        print(f"{'='*60}")

        for cat, srcs in sorted(categories.items()):
            print(f"\n  {cat.upper().replace('_', ' ')} ({len(srcs)} sources):")
            for s in srcs:
                print(f"    {s.name:<30} {s.display_name}")

        print(f"\n  Categories: {', '.join(sorted(categories.keys()))}")
        return

    fetcher = DocFetcher(
        output_dir=Path(args.output),
    )

    fetcher.fetch_all(
        sources=args.source,
        category=args.category,
        max_pages=args.max_pages,
    )


if __name__ == "__main__":
    main()
