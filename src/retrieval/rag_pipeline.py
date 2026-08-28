from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_NAME = "llama3.2:3b"

CHUNKS_PATH = Path("data/processed/scikit_learn_chunks.txt")
EMBEDDINGS_PATH = Path("data/processed/scikit_learn_embeddings.npy")

TOP_K = 3


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

    return [
        (int(index), float(scores[index]), chunks[index])
        for index in top_indices
    ]


def generate_answer(query, retrieved_chunks):
    context = "\n\n".join(
        f"Context {i + 1}:\n{chunk}"
        for i, (_, _, chunk) in enumerate(retrieved_chunks)
    )

    prompt = f"""You are a helpful assistant answering questions about scikit-learn.

Use ONLY the provided context to answer the question.
Do not use outside knowledge.
If the answer cannot be found in the context, say:
"I cannot answer this based on the provided documentation."

Context:
{context}

Question:
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
    query = input("\nEnter your question: ").strip()

    if not query:
        raise ValueError("Question cannot be empty.")

    print(f"\nQuery: {query}")

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

    retrieved_chunks = retrieve(
        query,
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
    )

    print("\n--- Answer ---")
    print(answer)


if __name__ == "__main__":
    main()