from pathlib import Path
import pandas as pd
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

def classify_claim(claim, evidence_chunks):
    claim_lower = claim.lower()

    support_words = ["leader", "led", "trusted", "fearless", "brave"]
    contradict_words = ["afraid", "feared", "avoided", "hesitant", "distrusted"]

    score = 0
    for chunk in evidence_chunks:
        text = chunk.lower()
        for w in support_words:
            if w in claim_lower and w in text:
                score += 1
        for w in contradict_words:
            if w in claim_lower and w in text:
                score -= 1

    if score > 0:
        return "SUPPORTED"
    elif score < 0:
        return "CONTRADICTED"
    else:
        return "UNCLEAR"

def aggregate(verdicts):
    if "CONTRADICTED" in verdicts:
        return 0, "At least one backstory claim contradicts the narrative"
    else:
        return 1, "All backstory claims align with the narrative"

# ---------------- MAIN ----------------
if __name__ == "__main__":
    test_df = pd.read_csv("data/test.csv")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    book_cache = []

    labels = []
    rationales = []

    for i, row in test_df.iterrows():
        print(f"🔄 Processing {i+1}/{len(test_df)} | ID={row['id']}")

        book_name = row["book_name"]
        backstory = row["content"]

        # load & index book once
        book_path = Path(f"data/books/{book_name}.txt")
        novel_text = load_text(book_path)
        chunks = chunk_text(novel_text)

        embeddings = model.encode(chunks).astype("float32")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        claims = extract_claims(backstory)
        verdicts = []

        for claim in claims:
            q_emb = model.encode([claim]).astype("float32")
            _, idxs = index.search(q_emb, k=5)
            evidence = [chunks[j] for j in idxs[0]]
            verdicts.append(classify_claim(claim, evidence))

        label, rationale = aggregate(verdicts)
        labels.append(label)
        rationales.append(rationale)

    # 🔥 APPEND — DO NOT REPLACE
    test_df["label"] = labels
    test_df["rationale"] = rationales

    test_df.to_csv("results.csv", index=False)
    print("\n✅ DONE: results.csv matches test.csv + predictions")
