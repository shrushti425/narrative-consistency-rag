import pathway as pw
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

class NovelSchema(pw.Schema):
    book_name: str
    chunk: str
    embedding: list[float]

def build_index(novel_chunks):
    rows = []
    for book, chunks in novel_chunks.items():
        for ch in chunks:
            emb = model.encode(ch).tolist()
            rows.append((book, ch, emb))

    table = pw.Table.from_rows(
        rows,
        schema=NovelSchema
    )

    index = pw.ml.index.KNNIndex(
        table.embedding,
        table,
        k=5
    )
    return index
