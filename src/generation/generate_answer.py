import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def generate_answer(query: str, context: str) -> str:
    prompt = f"""You are a helpful assistant answering questions about scikit-learn.

Answer the question using ONLY the provided context.
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
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()["response"].strip()


if __name__ == "__main__":
    query = "What is RandomForestClassifier?"

    context = """
Scikit-learn provides dozens of built-in machine learning algorithms
and models, called estimators. Each estimator can be fitted to some data
using its fit method.

Here is a simple example where we fit a RandomForestClassifier:

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=0)
clf.fit(X, y)
"""

    answer = generate_answer(query, context)

    print("\n--- Answer ---")
    print(answer)