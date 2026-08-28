from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_NAME = "llama3.2:3b"

CHUNKS_PATH = Path("data/processed/scikit_learn_chunks.txt")
EMBEDDINGS_PATH = Path("data/processed/scikit_learn_embeddings.npy")

TOP_K = 5
MAX_MEMORY_TURNS = 5
SIMILARITY_THRESHOLD = 0.50


def load_chunks():
    text = CHUNKS_PATH.read_text(encoding="utf-8")

    chunks = []
    current_chunk = []

    for line in text.splitlines():
        if line.startswith("--- CHUNK "):
            if current_chunk:
                chunks.append("\n".join(current_chunk).strip())
                current_chunk = []
        else:
            current_chunk.append(line)

    if current_chunk:
        chunks.append("\n".join(current_chunk).strip())

    return [chunk for chunk in chunks if chunk]


def retrieve(query, model, embeddings, chunks):
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )[0]

    scores = embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:TOP_K]

    results = [
        (int(index), float(scores[index]), chunks[index])
        for index in top_indices
        if scores[index] >= SIMILARITY_THRESHOLD
    ]

    return results


def build_retrieval_query(query, memory):
    """Create a context-aware query for semantic retrieval."""

    if not memory:
        return query

    previous_context = "\n".join(
        f"User: {user_message}\nAssistant: {assistant_message}"
        for user_message, assistant_message in memory[-2:]
    )

    return f"""Previous conversation:
{previous_context}

Current question:
{query}"""


def format_memory(memory):
    if not memory:
        return "No previous conversation."

    return "\n\n".join(
        f"User: {user_message}\nAssistant: {assistant_message}"
        for user_message, assistant_message in memory[-MAX_MEMORY_TURNS:]
    )


def extract_source(chunk):
    """Extract the source filename from a chunk."""

    for line in chunk.splitlines():
        if line.startswith("SOURCE:"):
            return line.replace("SOURCE:", "", 1).strip()

    return "Unknown source"


def generate_answer(query, retrieved_chunks, memory):
    context = "\n\n".join(
        f"Context {i + 1}:\n{chunk}"
        for i, (_, _, chunk) in enumerate(retrieved_chunks)
    )

    conversation_history = format_memory(memory)

    prompt = f"""You are a helpful assistant answering questions about scikit-learn.

Use ONLY the provided context and previous conversation to answer the question.
Do not use outside knowledge.

If the answer cannot be found in the provided context or previous conversation, say:
"I cannot answer this based on the provided documentation."

Keep your answer short and clear.

Previous conversation:
{conversation_history}

Retrieved context:
{context}

Current question:
{query}

Answer:"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": LLM_NAME,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()["response"].strip()


def main():
    print("\nLocal RAG Chatbot")
    print("Type 'end' to exit.")

    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    chunks = load_chunks()
    embeddings = np.load(EMBEDDINGS_PATH)

    print(f"Loaded chunks: {len(chunks)}")
    print(f"Loaded embeddings: {embeddings.shape}")

    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(chunks)} chunks but "
            f"{len(embeddings)} embeddings."
        )

    memory = []

    while True:
        query = input("\nYou: ").strip()

        if query.lower() == "end":
            print("\nEnding conversation.")
            break

        if not query:
            print("Please enter a question.")
            continue

        retrieval_query = build_retrieval_query(
            query,
            memory,
        )

        retrieved_chunks = retrieve(
            retrieval_query,
            model,
            embeddings,
            chunks,
        )

        print("\n--- Retrieved Context ---")

        for index, score, chunk in retrieved_chunks:
            print(f"\nChunk {index + 1} | Score: {score:.4f}")
            print(chunk[:500])

        answer = generate_answer(
            query,
            retrieved_chunks,
            memory,
        )

        sources = []
        for _, _, chunk in retrieved_chunks:
            source = extract_source(chunk)
            if source not in sources:
                sources.append(source)

        print("\nAssistant:")
        print(answer)

        print("\nSources:")
        for source in sources:
            print(f"- {source}")

        memory.append((query, answer))


if __name__ == "__main__":
    main()
