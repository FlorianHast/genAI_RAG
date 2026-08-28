# Local RAG Chatbot with Conversation Memory

A local Retrieval-Augmented Generation (RAG) chatbot that answers questions about scikit-learn documentation using semantic search, a locally running LLM, and conversation memory.

The project demonstrates a complete local RAG chatbot pipeline:

```text
Document
   ↓
Chunking
   ↓
Embeddings
   ↓
Vector Storage
   ↓
Semantic Retrieval
   ↓
Conversation Context
   ↓
Local LLM
   ↓
Answer
```

## Project Overview

The system uses the scikit-learn "Getting Started" documentation as its knowledge base.

For each user question, the system:

1. Loads the document chunks and embeddings.
2. Creates an embedding for the user question.
3. Incorporates recent conversation history when available.
4. Retrieves the most relevant document chunks using semantic similarity.
5. Combines the retrieved context with the conversation history.
6. Sends the context to a locally running LLM through Ollama.
7. Generates an answer based only on the provided documentation and conversation context.
8. Stores the question and answer in conversation memory.

The system is designed to run locally without paid API services.

## Architecture

```text
                    ┌─────────────────────────────┐
                    │      Source Document        │
                    │   scikit-learn docs         │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │          Chunking           │
                    │    1000 chars / 150         │
                    │         overlap             │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │         Embeddings          │
                    │ all-MiniLM-L6-v2            │
                    │      384 dimensions         │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │      Vector Storage         │
                    │           .npy              │
                    └──────────────┬──────────────┘
                                   │
                                   │
User Question ────────────────► Query
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │   Context-aware Retrieval  │
                    │                             │
                    │ Question + recent memory   │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │       Top-K Context         │
                    │        Top 3 chunks         │
                    └──────────────┬──────────────┘
                                   │
                                   │
Conversation Memory ────────────────┤
                                   ▼
                    ┌─────────────────────────────┐
                    │           Ollama            │
                    │       llama3.2:3b           │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                              Answer
                                   │
                                   ▼
                         Conversation Memory
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

## Preparing the Knowledge Base

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

## Running the Chatbot

Start the chatbot with:

```powershell
python src\retrieval\rag_pipeline.py
```

The system starts an interactive conversation:

```text
Local RAG Chatbot
Type 'end' to exit.

You: What is a pipeline in scikit-learn?
```

Follow-up questions can refer to previous messages:

```text
You: What is a pipeline in scikit-learn?

Assistant:
A pipeline in scikit-learn is a single unifying object that combines
transformers and estimators.

You: Why is it useful?

Assistant:
Pipelines are useful because they help prevent data leakage and provide
a consistent API for fitting and prediction.
```

Type `end` to terminate the conversation.

## Conversation Memory

The chatbot maintains a limited conversation history.

The current configuration stores the most recent five conversation turns:

```python
MAX_MEMORY_TURNS = 5
```

Conversation memory is used in two places.

### 1. Context-aware retrieval

For a follow-up question such as:

```text
Why is it useful?
```

the system combines the question with recent conversation history before performing semantic retrieval.

This allows the retrieval system to understand what "it" refers to.

For example:

```text
Previous conversation:
User: What is a pipeline in scikit-learn?
Assistant: A pipeline combines transformers and estimators.

Current question:
Why is it useful?
```

This significantly improves retrieval for follow-up questions.

### 2. Answer generation

The conversation history is also included in the prompt sent to Ollama.

The model receives:

```text
Previous conversation
        +
Retrieved documentation
        +
Current question
```

The model is instructed to use only this information when generating the answer.

## Retrieval

The system uses semantic similarity with normalized embeddings:

```text
similarity = embedding_matrix · query_embedding
```

The three highest-scoring chunks are retrieved.

The retrieval process is:

```text
User Question
      ↓
Context-aware Query
      ↓
Query Embedding
      ↓
Similarity Calculation
      ↓
Ranking
      ↓
Top 3 Chunks
```

The system uses brute-force similarity search with NumPy.

## Generation

The retrieved chunks and recent conversation history are inserted into a prompt containing the instruction:

```text
Use ONLY the provided context and previous conversation to answer the question.
Do not use outside knowledge.
```

If the answer cannot be found in the provided information, the model is instructed to respond:

```text
I cannot answer this based on the provided documentation.
```

The answer is generated locally through Ollama using:

```text
llama3.2:3b
```

## Testing

Run the automated tests with:

```powershell
python -m pytest tests
```

The current test suite verifies:

* The number of chunks matches the number of embeddings.
* Embeddings have the expected 384-dimensional representation.
* Retrieval queries work without conversation memory.
* Retrieval queries incorporate conversation memory.

Expected result:

```text
4 passed
```

## Current Limitations

This is a deliberately small RAG MVP.

Current limitations include:

* Only one documentation source is indexed.
* Embeddings are stored in a NumPy file rather than a dedicated vector database.
* Retrieval uses brute-force similarity search.
* The number of retrieved chunks is fixed to 3.
* Conversation memory is limited to the current runtime session.
* Conversation history is not persisted between sessions.
* There is no graphical user interface.
* There is no source citation in the generated answer.
* The local LLM can still produce an answer that is more general than the retrieved context.

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
* Persistent conversation memory.
* Streamlit user interface.
* Conversation history storage.
* Improved prompt engineering.
* Retrieval score thresholds for detecting insufficient context.

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
Context-aware retrieval
        ↓
Conversation memory
        ↓
Context construction
        ↓
Local LLM generation
        ↓
Interactive chatbot
        ↓
Automated tests
```
