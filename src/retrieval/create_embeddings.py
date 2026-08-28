from pathlib import Path

from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

INPUT_PATH = Path("data/processed/scikit_learn_chunks.txt")
OUTPUT_PATH = Path("data/processed/scikit_learn_embeddings.npy")


def load_chunks(path: Path) -> list[str]:
    """Load chunks from the processed text file."""
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


def main() -> None:
    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    chunks = load_chunks(INPUT_PATH)

    print(f"Loaded chunks: {len(chunks)}")

    embeddings = model.encode(
        chunks,
        show_progress_bar=True,
    )

    import numpy as np

    np.save(OUTPUT_PATH, embeddings)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Saved embeddings: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()