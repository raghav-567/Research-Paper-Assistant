# 🔬 Agentic Research Assistant

An intelligent, multi-agent system for automated academic literature review generation. This tool searches arXiv, extracts content from PDFs, builds a RAG (Retrieval-Augmented Generation) pipeline, and synthesizes comprehensive literature reviews.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [API Reference](#-api-reference)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## ✨ Features

- **🔍 Automated Paper Discovery**: Search and retrieve papers from arXiv based on your research query
- **📄 PDF Processing**: Intelligent extraction and cleaning of academic PDFs
- **🧠 RAG Pipeline**: Vector-based retrieval using FAISS for semantic search
- **📝 AI Summarization**: Powered by Google Gemini for high-quality summaries
- **📚 Literature Synthesis**: Automatic generation of structured literature reviews
- **🎯 Multi-Agent Architecture**: Specialized agents for different tasks
- **🖥️ Multiple Interfaces**: CLI, Streamlit UI, and FastAPI backend

---

## 🏗️ Architecture

The system uses a multi-agent architecture with specialized components:

```
┌─────────────────────────────────────────────────────────┐
│                    Research Query                        │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  SEARCH AGENT                                            │
│  • Queries arXiv API                                     │
│  • Downloads PDFs                                        │
│  • Extracts metadata                                     │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  EXTRACTION AGENT                                        │
│  • Parses PDF content                                    │
│  • Cleans text (removes refs, figures, equations)       │
│  • Extracts abstract                                     │
│  • Chunks body text                                      │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  RAG PIPELINE                                            │
│  • Generates embeddings (SentenceTransformer)           │
│  • Builds FAISS index                                    │
│  • Semantic similarity search                            │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  SUMMARIZER AGENT                                        │
│  • Summarizes abstract + body chunks                     │
│  • Query-based chunk selection                           │
│  • Google Gemini API integration                         │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│  SYNTHESIZER AGENT                                       │
│  • Aggregates paper summaries                            │
│  • Generates structured literature review                │
│  • Markdown formatting                                   │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Literature Review (Markdown)                │
└─────────────────────────────────────────────────────────┘
```

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/agentic-research-assistant.git
cd agentic-research-assistant
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up API Keys

Create a `.env` file in the root directory:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

Or export as environment variable:

```bash
# On Windows (Command Prompt):
set GEMINI_API_KEY=your_gemini_api_key_here

# On Windows (PowerShell):
$env:GEMINI_API_KEY="your_gemini_api_key_here"

# On macOS/Linux:
export GEMINI_API_KEY="your_gemini_api_key_here"
```

---

## 🚀 Quick Start

### Basic Usage (CLI)

```python
from src.main import run

# Generate literature review for your research topic
review = run("machine learning optimization")
```

The generated review will be saved in `outputs/sample_review_optimized.md`

### Using Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

Then open your browser to `http://localhost:8501`

### Using FastAPI Backend

```bash
uvicorn app.fastapi_app:app --reload
```

API will be available at `http://localhost:8000`
API documentation at `http://localhost:8000/docs`

---

## 📁 Project Structure

```
Agentic-Research-Assistant/
│
├── src/
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── search_agent.py        # arXiv search & PDF download
│   │   ├── extraction_agent.py    # PDF parsing & cleaning
│   │   ├── summarizer_agent.py    # AI-powered summarization
│   │   └── synthesizer_agent.py   # Literature review generation
│   │
│   ├── rag_pipeline.py             # FAISS + embeddings + retrieval
│   ├── utils.py                    # Logging, config helpers
│   └── main.py                     # Orchestrator (ties all agents)
│
├── app/
│   ├── streamlit_app.py            # Frontend UI
│   └── fastapi_app.py              # REST API backend
│
├── outputs/
│   ├── sample_review_optimized.md  # Generated literature reviews
│   └── references.bib              # BibTeX references
│
├── pdfs/                           # Downloaded PDFs (auto-created)
├── logs/                           # Application logs (auto-created)
│
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── .gitignore
└── README.md                       # This file
```

---

## 💻 Usage

### Command Line Interface

```python
from src.main import run

# Basic usage
review = run("deep learning")

# The function will:
# 1. Search arXiv for relevant papers
# 2. Download and parse PDFs
# 3. Extract and chunk content
# 4. Generate summaries using AI
# 5. Create a structured literature review
# 6. Save to outputs/sample_review_optimized.md
```

### Programmatic Usage

```python
from src.agents.search_agent import SearchAgent
from src.agents.extraction_agent import ExtractionAgent
from src.agents.summarizer_agent import SummarizerAgent
from src.rag_pipeline import RAGPipeline
import faiss

# Initialize agents
search = SearchAgent(pdf_dir="pdfs")
extraction = ExtractionAgent()
summarizer = SummarizerAgent(api_key="your_api_key")

# Search for papers
papers = search.search_arxiv("neural networks", max_results=5)

# Process a single paper
for paper in papers:
    if paper.get("pdf_path"):
        # Extract content
        parsed = extraction.parse_pdf(paper["pdf_path"], paper["id"])
        
        # Generate summary
        summary = summarizer.summarize_chunks(parsed, query="neural networks")
        
        print(f"Title: {paper['title']}")
        print(f"Summary: {summary}\n")
```

### Customization Options

```python
# Customize number of papers
papers = search.search_arxiv("quantum computing", max_results=10)

# Customize chunk size
chunks = extraction.chunk_text(text, chunk_size=1500)

# Customize summary length
summary = summarizer._summarize_text(text, max_output_tokens=500)

# Customize RAG retrieval
papers, summaries = rag.query(
    query="machine learning",
    top_k_chunks=200,
    top_k_papers=5,
    chunks_per_paper=15
)
```

### Batch Processing

```python
from src.main import run

queries = [
    "machine learning optimization",
    "deep learning computer vision",
    "natural language processing transformers"
]

for query in queries:
    print(f"Processing: {query}")
    review = run(query)
    print(f"Review saved for: {query}\n")
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file:

```bash
# Required
GEMINI_API_KEY=your_gemini_api_key

# Optional
PDF_DIR=pdfs
OUTPUT_DIR=outputs
LOG_LEVEL=INFO
MAX_PAPERS=5
CHUNK_SIZE=2000
```

### Agent Configuration

Edit parameters in `src/main.py`:

```python
# Search configuration
papers = search.search_arxiv(query, max_results=3)  # Number of papers

# Chunking configuration
chunks = extraction.chunk_text(text, chunk_size=2000)  # Chunk size

# Summarization configuration
summary = summarizer._summarize_text(
    text,
    max_output_tokens=300  # Summary length
)

# RAG configuration
papers, summaries = rag.query(
    query=query,
    top_k_chunks=200,      # Chunks to retrieve
    top_k_papers=3,        # Papers to include
    chunks_per_paper=10    # Chunks per paper
)
```

### Logging Configuration

In `src/utils.py`:

```python
def get_logger(name):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)  # Change to DEBUG for verbose output
    return logger
```

---

## 📚 API Reference

### SearchAgent

**Initialize:**
```python
search = SearchAgent(pdf_dir="pdfs")
```

**Methods:**

#### `search_arxiv(query, max_results=3)`
Search arXiv for papers matching the query.

**Parameters:**
- `query` (str): Search query
- `max_results` (int): Maximum number of papers to retrieve

**Returns:**
- List of paper dictionaries with keys: `id`, `title`, `summary`, `authors`, `pdf_path`, `url`, `published`

**Example:**
```python
papers = search.search_arxiv("machine learning", max_results=5)
for paper in papers:
    print(f"Title: {paper['title']}")
    print(f"Authors: {', '.join(paper['authors'])}")
```

#### `download_pdf(pdf_url, paper_id)`
Download a PDF from the given URL.

**Parameters:**
- `pdf_url` (str): URL of the PDF
- `paper_id` (str): Unique identifier for the paper

**Returns:**
- str: Path to downloaded PDF, or None if download failed

---

### ExtractionAgent

**Initialize:**
```python
extraction = ExtractionAgent()
```

**Methods:**

#### `parse_pdf(pdf_path, paper_id=None)`
Extract and clean text from a PDF file.

**Parameters:**
- `pdf_path` (str): Path to PDF file
- `paper_id` (str, optional): Paper identifier for metadata

**Returns:**
- dict: `{"abstract": str, "chunks": List[dict]}`

**Example:**
```python
parsed = extraction.parse_pdf("pdfs/paper123.pdf", paper_id="paper123")
print(f"Abstract: {parsed['abstract'][:200]}...")
print(f"Number of chunks: {len(parsed['chunks'])}")
```

#### `chunk_text(text, paper_id=None, chunk_size=2000)`
Split text into chunks for processing.

**Parameters:**
- `text` (str): Text to chunk
- `paper_id` (str, optional): Paper identifier
- `chunk_size` (int): Number of words per chunk

**Returns:**
- List[dict]: Chunks with metadata

#### `clean_text(text)`
Remove equations, citations, figures, and other noise from text.

**Parameters:**
- `text` (str): Raw text

**Returns:**
- str: Cleaned text

---

### SummarizerAgent

**Initialize:**
```python
summarizer = SummarizerAgent(api_key="your_gemini_api_key")
```

**Methods:**

#### `summarize_chunks(paper_data, query=None, k=10)`
Generate a summary from paper content.

**Parameters:**
- `paper_data` (dict): `{"abstract": str, "chunks": List[dict]}`
- `query` (str, optional): Query for relevance-based chunk selection
- `k` (int): Number of top chunks to use

**Returns:**
- str: Generated summary

**Example:**
```python
summary = summarizer.summarize_chunks(
    paper_data=parsed,
    query="machine learning",
    k=5
)
print(summary)
```

#### `embed_chunks(chunks, normalize=True)`
Generate embeddings for text chunks.

**Parameters:**
- `chunks` (List[str]): List of text chunks
- `normalize` (bool): Whether to normalize embeddings

**Returns:**
- np.ndarray: Array of embeddings

---

### RAGPipeline

**Initialize:**
```python
from src.rag_pipeline import RAGPipeline
import faiss

dim = summarizer.embedding_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(dim)

rag = RAGPipeline(
    search_agent=search,
    extraction_agent=extraction,
    summarizer_agent=summarizer,
    index=index,
    id_to_metadata={}
)
```

**Methods:**

#### `build_index(chunks, paper_info=None)`
Add chunks to the FAISS index.

**Parameters:**
- `chunks` (List[dict]): Chunks with text and metadata
- `paper_info` (dict, optional): Paper metadata

**Example:**
```python
rag.build_index(chunks, paper_info=paper)
```

#### `query(query, top_k_chunks=200, top_k_papers=3, chunks_per_paper=10)`
Retrieve and summarize relevant papers.

**Parameters:**
- `query` (str): Search query
- `top_k_chunks` (int): Total chunks to retrieve
- `top_k_papers` (int): Number of papers to return
- `chunks_per_paper` (int): Chunks per paper for summarization

**Returns:**
- Tuple[List[dict], List[str]]: (papers, summaries)

---

### SynthesizerAgent

**Initialize:**
```python
from src.agents.synthesizer_agent import SynthesizerAgent

synthesizer = SynthesizerAgent()
```

**Methods:**

#### `synthesize(papers, summaries)`
Generate a structured literature review.

**Parameters:**
- `papers` (List[dict]): List of paper metadata
- `summaries` (List[str]): List of paper summaries

**Returns:**
- str: Markdown-formatted literature review

---

## 🔧 Troubleshooting

### Common Issues

#### 1. **"Unable to generate summary after 3 attempts"**

**Cause**: Gemini API rate limiting or quota exceeded

**Solutions:**
```python
# Solution 1: Reduce number of papers
papers = search.search_arxiv(query, max_results=2)

# Solution 2: Use optimized main.py (fewer API calls)
from src.main import run  # Uses optimized version

# Solution 3: Check your API quota
# Visit: https://makersuite.google.com/app/apikey

# Solution 4: Add manual delays
import time
time.sleep(2)  # Between API calls
```

#### 2. **"No PDF for [paper], using arXiv summary"**

**Cause**: PDF download failed or paper has no PDF available

**Solutions:**
- Check your internet connection
- Some papers don't have PDFs (will use abstract instead)
- Check `pdfs/` directory permissions:
  ```bash
  # On macOS/Linux:
  chmod 755 pdfs/
  
  # On Windows:
  # Right-click pdfs folder → Properties → Security → Edit
  ```

#### 3. **"Empty embeddings" or FAISS errors**

**Cause**: No valid text chunks extracted

**Solutions:**
```python
# Check if PDFs downloaded
import os
print(os.listdir("pdfs"))

# Check extraction
parsed = extraction.parse_pdf(pdf_path, paper_id)
print(f"Abstract: {bool(parsed['abstract'])}")
print(f"Chunks: {len(parsed['chunks'])}")

# Debug extraction
if not parsed['chunks']:
    print("No chunks extracted - PDF might be image-based or corrupted")
```

#### 4. **Memory errors with large PDFs**

**Solutions:**
```python
# Solution 1: Reduce chunk size
chunks = extraction.chunk_text(text, chunk_size=1000)

# Solution 2: Limit number of chunks
chunks = chunks[:50]

# Solution 3: Process papers one at a time
for paper in papers:
    # Process and clear memory
    del parsed, summary
    import gc
    gc.collect()
```

#### 5. **Import errors**

**Cause**: Missing dependencies or wrong Python version

**Solutions:**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check Python version
python --version  # Should be 3.8+

# Create fresh virtual environment
deactivate
rm -rf venv
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

#### 6. **"ModuleNotFoundError: No module named 'src'"**

**Cause**: Python path not set correctly

**Solutions:**
```python
# Solution 1: Add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Solution 2: Run from project root
cd Agentic-Research-Assistant/
python -c "from src.main import run; run('test')"

# Solution 3: Install as package
pip install -e .
```

### Debug Mode

Enable detailed logging:

```python
# In your script
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Or set in environment
export LOG_LEVEL=DEBUG  # macOS/Linux
set LOG_LEVEL=DEBUG     # Windows
```

### Getting Help

If you encounter issues:

1. **Check logs**: Look in the console output for error messages
2. **Enable debug mode**: Set `LOG_LEVEL=DEBUG`
3. **Check API quota**: Visit [Google AI Studio](https://makersuite.google.com)
4. **Open an issue**: [GitHub Issues](https://github.com/yourusername/agentic-research-assistant/issues)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

### How to Contribute

1. **Fork the repository**
   ```bash
   # Click "Fork" on GitHub
   git clone https://github.com/your-username/agentic-research-assistant.git
   cd agentic-research-assistant
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make your changes**
   - Write code
   - Add tests
   - Update documentation

4. **Commit your changes**
   ```bash
   git add .
   git commit -m 'Add amazing feature'
   ```

5. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```

6. **Open a Pull Request**
   - Go to GitHub
   - Click "New Pull Request"
   - Describe your changes

### Development Guidelines

- Follow PEP 8 style guide
- Add docstrings to all functions
- Write unit tests for new features
- Update README if needed

### Development Setup

```bash
# Install development dependencies
pip install pytest flake8 black mypy

# Run tests
pytest tests/

# Run linting
flake8 src/

# Format code
black src/

# Type checking
mypy src/
```

---

## 📝 License

This project is licensed under the MIT License - see below for details.

```
MIT License

Copyright (c) 2024 [Raghav Agarwal]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- **[arXiv](https://arxiv.org/)** for providing open access to research papers
- **[Google Gemini](https://ai.google.dev/)** for AI summarization capabilities
- **[Sentence Transformers](https://www.sbert.net/)** for embedding models
- **[FAISS](https://github.com/facebookresearch/faiss)** for efficient similarity search
- **[PyMuPDF](https://pymupdf.readthedocs.io/)** for PDF processing

---

## 📧 Contact

For questions, suggestions, or collaboration:

- **Email**: agarwal1996raghav@gmail.com
- **GitHub**: [@raghav-567](https://github.com/raghav-567)
- **Issues**: [Report a bug](https://github.com/yourusername/agentic-research-assistant/issues)
- **Discussions**: [Start a discussion](https://github.com/yourusername/agentic-research-assistant/discussions)

---

## 🔮 Roadmap

### Current Version (v1.0)
- ✅ arXiv integration
- ✅ PDF extraction
- ✅ RAG pipeline with FAISS
- ✅ AI summarization
- ✅ Literature review generation

### Upcoming Features (v1.1)
- [ ] Support for PubMed and Google Scholar
- [ ] Citation graph analysis
- [ ] Interactive web UI improvements
- [ ] Multi-language support
- [ ] Custom prompt templates

### Future Vision (v2.0)
- [ ] Export to LaTeX/Word
- [ ] Collaborative filtering
- [ ] Paper recommendation system
- [ ] Knowledge graph visualization
- [ ] Real-time collaboration features
- [ ] Integration with reference managers (Zotero, Mendeley)

---

## 📊 Performance

Typical performance metrics:

| Operation | Time | API Calls |
|-----------|------|-----------|
| Search 5 papers | ~5s | 1 |
| Download PDFs | ~10s | 0 |
| Extract & Chunk | ~15s | 0 |
| Generate Summaries | ~30s | 5 |
| Build RAG Index | ~5s | 0 |
| Total Pipeline | ~65s | 6 |

*Note: Times vary based on paper length and network speed*

---
