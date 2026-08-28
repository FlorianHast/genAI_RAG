from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNKS_PATH = Path("data/processed/scikit_learn_chunks.txt")
EMBEDDINGS_PATH = Path("data/processed/scikit_learn_embeddings.npy")


def load_chunks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")

    raw_chunks = text.split("--- CHUNK ")
    chunks = []

    for raw_chunk in raw_chunks[1:]:
        parts = raw_chunk.split("---", 1)

        if len(parts) == 2:
            chunk = parts[1].strip()

            if chunk:
                chunks.append(chunk)

    return chunks


def keyword_score(query: str, chunk: str) -> float:
    """Calculate keyword overlap after basic text normalization."""

    def normalize(text: str) -> set[str]:
        cleaned = ""

        for char in text.lower():
            if char.isalnum() or char.isspace():
                cleaned += char
            else:
                cleaned += " "

        return set(cleaned.split())

    query_words = normalize(query)
    chunk_words = normalize(chunk)

    if not query_words:
        return 0.0

    return len(query_words & chunk_words) / len(query_words)


def search(
    query: str,
    model: SentenceTransformer,
    chunks: list[str],
    embeddings: np.ndarray,
    top_k: int = 3,
) -> list[tuple[int, float, str]]:
    """Perform semantic retrieval with exact-term prioritization."""

    query_embedding = model.encode([query])

    semantic_scores = cosine_similarity(
        query_embedding,
        embeddings,
    )[0]

    normalized_query = query.lower().strip(" ?!.,:;")

    final_scores = semantic_scores.copy()

    for index, chunk in enumerate(chunks):
        normalized_chunk = chunk.lower()

        # Strongly prioritize an exact query phrase.
        if normalized_query in normalized_chunk:
            final_scores[index] += 0.3

    top_indices = np.argsort(final_scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append(
            (
                int(index) + 1,
                float(final_scores[index]),
                chunks[index],
            )
        )

    return results

def main() -> None:
    model = SentenceTransformer(MODEL_NAME)

    chunks = load_chunks(CHUNKS_PATH)
    embeddings = np.load(EMBEDDINGS_PATH)

    query = "What is RandomForestClassifier?"

    results = search(
        query=query,
        model=model,
        chunks=chunks,
        embeddings=embeddings,
        top_k=3,
    )

    print(f"\nQuery: {query}\n")

    for chunk_number, score, chunk in results:
        print(f"--- Chunk {chunk_number} | Hybrid Score: {score:.4f} ---")
        print(chunk[:500])
        print()


if __name__ == "__main__":
    main()
