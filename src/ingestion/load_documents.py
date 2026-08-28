from pathlib import Path

import requests
from bs4 import BeautifulSoup


URL = "https://scikit-learn.org/stable/getting_started.html"
OUTPUT_PATH = Path("data/raw/scikit_learn_getting_started.txt")


def load_webpage(url: str) -> str:
    """Download a scikit-learn documentation page and preserve its structure."""
    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    main = soup.select_one("main article")

    if main is None:
        raise ValueError("Could not find the main documentation article.")

    for element in main.select("script, style, .headerlink"):
        element.decompose()

    parts = []

    for element in main.find_all(["h1", "h2", "h3", "h4", "p", "pre", "li"]):

        if element.name == "pre":
            text = element.get_text("", strip=False).strip()

            if text:
                parts.append(f"CODE:\n{text}")

        elif element.name.startswith("h"):
            text = element.get_text(" ", strip=True)

            if text:
                parts.append(f"\n{text}\n")

        else:
            text = element.get_text(" ", strip=True)

            if text and text not in parts:
                parts.append(text)

    return "\n\n".join(parts)


def save_document(text: str, output_path: Path) -> None:
    """Save extracted document text to a file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    text = load_webpage(URL)
    save_document(text, OUTPUT_PATH)

    print(f"Downloaded characters: {len(text)}")
    print(f"Saved document: {OUTPUT_PATH}")