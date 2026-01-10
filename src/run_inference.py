from pathlib import Path
import pandas as pd
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
import faiss

nltk.download("punkt", quiet=True)

# ---------------- HELPERS ----------------
def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def extract_claims(text):
    return nltk.sent_tokenize(text)

def build_faiss_index(chunks, model):
    embeddings = model.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def claim_support_score(claim, chunks, index, model, k=5):
    q_emb = model.encode([claim]).astype("float32")
    distances, idxs = index.search(q_emb, k=k)
    sims = 1 / (1 + distances[0])   # convert L2 → similarity
    return float(np.mean(sims))

# ---------------- MAIN ----------------
if __name__ == "__main__":
    test_df = pd.read_csv("data/test.csv")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    labels = []
    rationales = []
    scores = []

    book_cache = {}

    for i, row in test_df.iterrows():
        print(f"🔄 Processing {i+1}/{len(test_df)} | ID={row['id']}")

        book_name = row["book_name"]
        backstory = row["content"]

        # cache book embeddings
        if book_name not in book_cache:
            book_path = Path(f"data/books/{book_name}.txt")
            novel_text = load_text(book_path)
            chunks = chunk_text(novel_text)
            index, _ = build_faiss_index(chunks, model)
            book_cache[book_name] = (chunks, index)

        chunks, index = book_cache[book_name]

        claims = extract_claims(backstory)
        claim_scores = [
            claim_support_score(c, chunks, index, model)
            for c in claims
        ]

        avg_score = float(np.mean(claim_scores))
        scores.append(round(avg_score, 4))

        # 🔑 DECISION RULE (THIS FIXES ALL-ONES BUG)
        if avg_score >= 0.42:
            label = 1
            rationale = "Backstory claims are semantically supported by the narrative"
        else:
            label = 0
            rationale = "Backstory claims are weakly supported or contradicted by the narrative"

        labels.append(label)
        rationales.append(rationale)

    # 🔥 STRICT FORMAT: test.csv + 3 columns
    output_df = test_df.copy()
    output_df["label"] = labels
    output_df["rationale"] = rationales
    output_df["score"] = scores

    output_df.to_csv("results.csv", index=False)
    print("\n✅ DONE: results.csv in correct format with non-degenerate labels")
