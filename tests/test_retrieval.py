from pathlib import Path

import numpy as np


CHUNKS_PATH = Path("data/processed/scikit_learn_chunks.txt")
EMBEDDINGS_PATH = Path("data/processed/scikit_learn_embeddings.npy")


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


def test_chunks_and_embeddings_match():
    chunks = load_chunks()
    embeddings = np.load(EMBEDDINGS_PATH)

    assert len(chunks) == len(embeddings)


def test_embedding_dimension():
    embeddings = np.load(EMBEDDINGS_PATH)

    assert embeddings.shape[1] == 384