from pathlib import Path


INPUT_PATH = Path("data/raw/scikit_learn_getting_started.txt")
OUTPUT_PATH = Path("data/processed/scikit_learn_chunks.txt")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_document(path: Path) -> str:
    """Load a text document."""
    return path.read_text(encoding="utf-8")


def split_large_paragraph(
    paragraph: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """Split an oversized paragraph using sentence/line boundaries where possible."""

    if len(paragraph) <= chunk_size:
        return [paragraph]

    parts = []
    current = ""

    # Prefer line boundaries, which helps preserve code blocks.
    segments = [
        segment.strip()
        for segment in paragraph.splitlines()
        if segment.strip()
    ]

    for segment in segments:
        candidate = segment if not current else current + "\n" + segment

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                parts.append(current)

            if len(segment) <= chunk_size:
                current = segment
            else:
                # Final fallback for a single line that is too large.
                start = 0

                while start < len(segment):
                    end = min(start + chunk_size, len(segment))
                    parts.append(segment[start:end].strip())

                    if end >= len(segment):
                        current = ""
                        break

                    start = end - chunk_overlap

    if current:
        parts.append(current)

    return parts


def create_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Create chunks while preserving paragraph and line boundaries."""

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []
    current_parts = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph_parts = split_large_paragraph(
            paragraph,
            chunk_size,
            chunk_overlap,
        )

        for part in paragraph_parts:
            part_length = len(part)

            if not current_parts:
                current_parts = [part]
                current_length = part_length
                continue

            candidate_length = current_length + 2 + part_length

            if candidate_length <= chunk_size:
                current_parts.append(part)
                current_length = candidate_length
                continue

            chunks.append("\n\n".join(current_parts))

            # Keep a small number of complete previous lines as overlap.
            overlap_parts = []
            overlap_length = 0

            for previous in reversed(current_parts):
                additional_length = len(previous) + (
                    2 if overlap_parts else 0
                )

                if overlap_length + additional_length > chunk_overlap:
                    break

                overlap_parts.insert(0, previous)
                overlap_length += additional_length

            current_parts = overlap_parts + [part]
            current_length = sum(len(p) for p in current_parts) + (
                2 * (len(current_parts) - 1)
            )

            # If overlap + new part is still too large, keep only the new part.
            if current_length > chunk_size:
                current_parts = [part]
                current_length = part_length

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def save_chunks(chunks: list[str], path: Path) -> None:
    """Save chunks to a text file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for index, chunk in enumerate(chunks, start=1):
            file.write(f"--- CHUNK {index} ---\n")
            file.write(chunk)
            file.write("\n\n")


if __name__ == "__main__":
    text = load_document(INPUT_PATH)

    chunks = create_chunks(text)

    save_chunks(chunks, OUTPUT_PATH)

    print(f"Document characters: {len(text)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Chunk size: {CHUNK_SIZE} characters")
    print(f"Chunk overlap: {CHUNK_OVERLAP} characters")
    print(f"Saved chunks: {OUTPUT_PATH}")

    for index, chunk in enumerate(chunks, start=1):
        print(f"\n--- Chunk {index} ({len(chunk)} characters) ---")
        print(chunk[:150])

        