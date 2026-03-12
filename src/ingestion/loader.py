"""
Document Loaders — PDF, Markdown, and Web page ingestion.
Each loader extracts text + metadata and returns Document objects.
"""

import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

import fitz  # PyMuPDF
import markdown
import requests
from bs4 import BeautifulSoup


@dataclass
class Document:
    """Represents a loaded document with metadata."""
    text: str
    source: str
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class PDFLoader:
    """Load and extract text from PDF files."""

    def load(self, file_path: str) -> List[Document]:
        """Extract text and metadata from each page of a PDF."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF not found: {file_path}")

        documents = []
        filename = os.path.basename(file_path)

        doc = fitz.open(file_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text").strip()

            if not text:
                continue

            # Try to extract first heading-like line as section heading
            lines = text.split("\n")
            section = None
            for line in lines[:5]:  # Check first 5 lines
                line = line.strip()
                if line and len(line) < 100 and line[0].isupper():
                    section = line
                    break

            documents.append(Document(
                text=text,
                source=filename,
                page_number=page_num + 1,
                section_heading=section,
                metadata={
                    "file_path": file_path,
                    "file_type": "pdf",
                    "total_pages": len(doc),
                }
            ))

        doc.close()
        return documents


class MarkdownLoader:
    """Load and extract text from Markdown files."""

    def load(self, file_path: str) -> List[Document]:
        """Extract text and section headings from a Markdown file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Markdown file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        filename = os.path.basename(file_path)

        # Split by headings to create section-aware documents
        sections = re.split(r"(^#{1,3}\s+.+$)", content, flags=re.MULTILINE)

        documents = []
        current_heading = None

        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue

            if re.match(r"^#{1,3}\s+", section):
                current_heading = re.sub(r"^#{1,3}\s+", "", section).strip()
                continue

            # Convert markdown to plain text
            html = markdown.markdown(section)
            text = BeautifulSoup(html, "html.parser").get_text()

            if text.strip():
                documents.append(Document(
                    text=text.strip(),
                    source=filename,
                    section_heading=current_heading,
                    metadata={
                        "file_path": file_path,
                        "file_type": "markdown",
                    }
                ))

        # If no sections found, return whole file as one document
        if not documents and content.strip():
            html = markdown.markdown(content)
            text = BeautifulSoup(html, "html.parser").get_text()
            documents.append(Document(
                text=text.strip(),
                source=filename,
                metadata={
                    "file_path": file_path,
                    "file_type": "markdown",
                }
            ))

        return documents


class WebLoader:
    """Load and extract text from web pages."""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "Mozilla/5.0 (RAG System Document Loader)"
        }

    def load(self, url: str) -> List[Document]:
        """Fetch and extract text from a web page."""
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to fetch {url}: {e}")

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove scripts, styles, nav elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Extract title
        title = soup.find("title")
        title_text = title.get_text().strip() if title else url

        # Extract main content
        main = soup.find("main") or soup.find("article") or soup.find("body")
        if not main:
            return []

        text = main.get_text(separator="\n", strip=True)

        if text:
            return [Document(
                text=text,
                source=url,
                section_heading=title_text,
                metadata={
                    "file_type": "web",
                    "url": url,
                }
            )]

        return []


def load_documents(path: str) -> List[Document]:
    """Auto-detect file type and load documents from a file or directory."""
    documents = []

    if os.path.isfile(path):
        documents.extend(_load_single_file(path))
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in sorted(files):
                file_path = os.path.join(root, file)
                try:
                    docs = _load_single_file(file_path)
                    documents.extend(docs)
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")
    elif path.startswith("http"):
        loader = WebLoader()
        documents.extend(loader.load(path))
    else:
        raise FileNotFoundError(f"Path not found: {path}")

    return documents


def _load_single_file(file_path: str) -> List[Document]:
    """Load a single file based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return PDFLoader().load(file_path)
    elif ext in (".md", ".markdown"):
        return MarkdownLoader().load(file_path)
    elif ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            return [Document(
                text=text,
                source=os.path.basename(file_path),
                metadata={"file_path": file_path, "file_type": "text"}
            )]
    return []
