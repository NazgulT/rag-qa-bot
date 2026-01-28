# RAG Document QA Bot with Gemma 3, LangChain, ChromaDB

A Retrieval-Augmented Generation (RAG) chatbot that uses LangChain, ChromaDB, HuggingFace Gemma 3, and Gradio to answer questions about uploaded PDF documents.

## Features

- **PDF Document Upload**: Upload any PDF document for querying
- **Semantic Search**: Uses ChromaDB for vector-based document retrieval
- **Advanced LLM**: Powered by Google's Gemma 3 (2B) model
- **HuggingFace Embeddings**: Uses `sentence-transformers/all-mpnet-base-v2` for semantic understanding
- **Web Interface**: Easy-to-use Gradio interface
- **Smart Caching**: Reuses vector stores for the same document to save computation

## Requirements

- Python 3.11+
- Virtual environment (recommended)
- GPU support not used

## Setup & Installation

### 1. Activate Virtual Environment
```bash
source qa_doc_env/bin/activate
```

### 2. Install Dependencies
```bash
pip install langchain-huggingface langchain-community chromadb pypdf gradio transformers torch sentence-transformers
```

### 3. Configure HuggingFace (if needed)
If you need to use a private model, set your HuggingFace token:
```bash
huggingface-cli login
```

## Running the Application

```bash
python qabot.py
```

The application will:
1. Load the Gemma 3 model (this may take a few minutes on first run)
2. Start a local server at `http://127.0.0.1:7860`
3. Open automatically in your default browser

## Usage

1. **Upload a PDF**: Click the upload area and select a PDF file
2. **Ask a Question**: Type your question in the text box
3. **Get Answer**: The bot will retrieve relevant sections and generate an answer

## How It Works

1. **Document Processing**: PDF is loaded and split into chunks (1000 chars with 100 overlap)
2. **Embedding**: Text chunks are converted to embeddings using HuggingFace embeddings
3. **Vector Store**: Embeddings stored in ChromaDB for fast retrieval
4. **Retrieval**: When you ask a question, the top 3 most relevant chunks are retrieved
5. **Generation**: Gemma 3 model generates an answer based on retrieved context

## Performance Tips

- **First Run**: The Gemma 3 model (~18GB) will download automatically. This may take several minutes.
- **Document Caching**: The same document isn't re-processed if uploaded again
- **Batch Processing**: Consider splitting very large PDFs into smaller documents for better results

## Troubleshooting

### "Module not found" errors
```bash
pip install --upgrade langchain langchain-huggingface langchain-community
```

### Out of Memory errors
- Use a smaller model or enable 8-bit quantization
- Reduce `chunk_size` from 1000 to 500
- Process smaller PDFs


## Model Information

- **LLM**: `google/gemma-1.1-2b-it`
- **Embeddings**: `sentence-transformers/all-mpnet-base-v2`
- **Vector Database**: ChromaDB (in-memory, ephemeral)
- **Text Splitter**: RecursiveCharacterTextSplitter (1000 tokens, 100 overlap)

## Architecture

```
PDF Upload
    ↓
PyPDFLoader
    ↓
RecursiveCharacterTextSplitter
    ↓
HuggingFaceEmbeddings
    ↓
ChromaDB Vector Store
    ↓
Retrieval (top-3 chunks)
    ↓
Gemma 3 LLM + Prompt Chain
    ↓
Generated Answer
```

## Customization

Edit [qabot.py](qabot.py) to:
- Change the LLM model
- Adjust chunk size/overlap for document splitting
- Modify prompt template for different answer styles
- Adjust the number of retrieved chunks (default: 3)
- Change temperature and max tokens for generation

## Notes

- Vector stores are ephemeral (reset when app restarts)
- For production, consider using persistent storage (e.g., Chroma with persistent backend)
- The model requires significant RAM (18GB+)
- Internet connection required for first-time model downloads
