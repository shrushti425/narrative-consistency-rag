from pathlib import Path
import pandas as pd
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
import faiss

nltk.download("punkt")
nltk.download("punkt_tab")

# ---------- Helpers ----------
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

def extract_claims(backstory):
    return nltk.sent_tokenize(backstory)

def classify_claim(claim, evidence_chunks):
    claim_lower = claim.lower()

    support_signals = ["trusted", "leader", "led", "fearless", "brave"]
    contradiction_signals = ["afraid", "feared", "avoided", "hesitant", "distrusted"]

    score = 0

    for chunk in evidence_chunks:
        text = chunk.lower()

        for w in support_signals:
            if w in claim_lower and w in text:
                score += 1

        for w in contradiction_signals:
            if w in claim_lower and w in text:
                score -= 1

    if score > 0:
        return "SUPPORTED"
    elif score < 0:
        return "CONTRADICTED"
    else:
        return "UNCLEAR"

def aggregate_verdicts(verdicts):
    if "CONTRADICTED" in verdicts:
        return 0
    return 1

# ---------- MAIN ----------
if __name__ == "__main__":
    df = pd.read_csv("data/train.csv")
    row = df.iloc[0]

    book_name = row["book_name"]
    backstory = row["content"]

    print("\n📘 Book:", book_name)
    print("🧾 Backstory:", backstory)

    book_path = Path(f"data/books/{book_name}.txt")
    novel_text = load_text(book_path)

    chunks = chunk_text(novel_text)
    print(f"\n📚 Total novel chunks: {len(chunks)}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_embeddings = model.encode(chunks)
    chunk_embeddings = np.array(chunk_embeddings).astype("float32")

    index = faiss.IndexFlatL2(chunk_embeddings.shape[1])
    index.add(chunk_embeddings)

    claims = extract_claims(backstory)

    print("\n📜 Extracted Claims:")
    for i, c in enumerate(claims, 1):
        print(f"{i}. {c}")

    print("\n🔍 Evidence & Verdicts:")
    claim_verdicts = []

    for claim in claims:
        print(f"\nCLAIM → {claim}")

        query_emb = model.encode([claim]).astype("float32")
        _, idxs = index.search(query_emb, k=5)

        evidence = [chunks[i] for i in idxs[0]]
        verdict = classify_claim(claim, evidence)
        claim_verdicts.append(verdict)

        print("VERDICT:", verdict)
        print("EVIDENCE:")
        for ev in evidence:
            print(" •", ev[:200].replace("\n", " "))

    final_label = aggregate_verdicts(claim_verdicts)

    print("\n📊 Claim Verdicts:", claim_verdicts)
    print("🏁 FINAL PREDICTION:", final_label)

