# Local RAG System for scikit-learn Documentation

A local Retrieval-Augmented Generation (RAG) system that answers questions about scikit-learn documentation using semantic search and a locally running LLM.

The project demonstrates a complete RAG pipeline:

**Document → Chunking → Embeddings → Retrieval → Context → Local LLM → Answer**

## Project Overview

The system uses the scikit-learn "Getting Started" documentation as its knowledge base.

For a user question, the system:

1. Loads the document chunks.
2. Creates an embedding for the question.
3. Compares the question embedding with the document embeddings.
4. Retrieves the most relevant chunks.
5. Sends the retrieved context to a local LLM.
6. Generates an answer based only on the retrieved documentation.

The system is designed to run locally without paid API services.

## Architecture

```text
                    ┌─────────────────────┐
                    │  Source Document    │
                    │  scikit-learn docs  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Chunking       │
                    │  1000 chars / 150   │
                    │      overlap        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Embeddings       │
                    │ all-MiniLM-L6-v2    │
                    │    384 dimensions   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Vector Storage    │
                    │       .npy          │
                    └──────────┬──────────┘
                               │
                               │
User Question ────────► Semantic Retrieval
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Top-K Context      │
                    │   Top 3 chunks       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Ollama         │
                    │    llama3.2:3b      │
                    └──────────┬──────────┘
                               │
                               ▼
                         Final Answer
```

## Technologies

* Python 3.13
* Sentence Transformers
* `sentence-transformers/all-MiniLM-L6-v2`
* NumPy
* scikit-learn
* Ollama
* Llama 3.2 3B
* Requests
* Pytest

All components are available for local use without paid API calls.

## Project Structure

```text
genAI_RAG/
│
├── data/
│   ├── raw/
│   │   └── scikit_learn_getting_started.txt
│   │
│   └── processed/
│       ├── scikit_learn_chunks.txt
│       └── scikit_learn_embeddings.npy
│
├── src/
│   ├── ingestion/
│   │   ├── load_documents.py
│   │   └── chunk_documents.py
│   │
│   ├── retrieval/
│   │   ├── create_embeddings.py
│   │   ├── search.py
│   │   └── rag_pipeline.py
│   │
│   └── generation/
│       └── generate_answer.py
│
├── tests/
│   └── test_retrieval.py
│
├── requirements.txt
└── .gitignore
```

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Install Ollama separately and make sure it is running.

Pull the local model:

```powershell
ollama pull llama3.2:3b
```

Verify that the model is available:

```powershell
ollama list
```

## Running the Pipeline

### 1. Load the document

```powershell
python src\ingestion\load_documents.py
```

### 2. Create document chunks

```powershell
python src\ingestion\chunk_documents.py
```

The current configuration creates chunks of approximately 1000 characters with 150 characters of overlap.

### 3. Create embeddings

```powershell
python src\retrieval\create_embeddings.py
```

The embeddings are generated using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Each embedding has 384 dimensions.

### 4. Run the RAG pipeline

```powershell
python src\retrieval\rag_pipeline.py
```

The system will ask:

```text
Enter your question:
```

Example:

```text
What is a pipeline in scikit-learn?
```

The pipeline retrieves the three most relevant chunks and sends them to the local LLM.

## Example

### Question

```text
What is a pipeline in scikit-learn?
```

### Retrieved Context

The system retrieves documentation describing pipelines as a combination of transformers and estimators.

### Generated Answer

```text
A pipeline in scikit-learn is a single unifying object that combines
transformers (pre-processors) and estimators (predictors) into a single unit,
allowing the same API as a regular estimator to be used.
```

The answer is generated using the retrieved documentation as context.

## Retrieval

The current implementation uses cosine similarity through normalized embeddings:

```text
similarity = embedding_matrix · query_embedding
```

The three highest-scoring chunks are retrieved.

The retrieval process therefore consists of:

```text
Query
  ↓
Query Embedding
  ↓
Similarity Calculation
  ↓
Ranking
  ↓
Top 3 Chunks
```

## Generation

The retrieved chunks are inserted into a prompt containing the instruction:

```text
Use ONLY the provided context to answer the question.
Do not use outside knowledge.
```

This reduces the likelihood of the LLM answering from its general knowledge instead of the retrieved documentation.

If the answer cannot be found in the provided context, the model is instructed to respond:

```text
I cannot answer this based on the provided documentation.
```

## Testing

Run the automated tests with:

```powershell
python -m pytest tests
```

Current tests verify:

* The number of chunks matches the number of embeddings.
* Embeddings have the expected 384-dimensional representation.

Expected result:

```text
2 passed
```

## Current Limitations

This is a deliberately small RAG MVP.

Current limitations include:

* Only one documentation source is indexed.
* Embeddings are stored in a NumPy file rather than a dedicated vector database.
* Retrieval uses brute-force similarity search.
* The number of retrieved chunks is fixed to 3.
* There is no conversational memory.
* There is no web search.
* There is no graphical user interface.
* The local LLM can still produce an answer that is more general than the retrieved context.

These limitations provide potential directions for future development.

## Future Improvements

Possible extensions include:

* Support for multiple documents.
* Metadata-aware retrieval.
* Better chunking strategies.
* Hybrid keyword + semantic search.
* Vector databases such as FAISS or Chroma.
* Retrieval evaluation metrics.
* Answer evaluation.
* Source citations in generated answers.
* Streamlit user interface.
* Conversation history.
* Improved prompt engineering.

## Privacy and Cost

The system is designed as a local and free RAG implementation.

The embedding model runs locally after downloading it from Hugging Face.

The LLM runs locally through Ollama.

No paid LLM API is required for the RAG pipeline.

API keys and tokens are intentionally excluded from the repository.

## Status

**MVP completed**

The current implementation successfully performs:

```text
Document ingestion
        ↓
Chunking
        ↓
Embedding generation
        ↓
Semantic retrieval
        ↓
Context construction
        ↓
Local LLM generation
        ↓
Interactive answer
        ↓
Automated tests
```
