import os
import pickle
from typing import Dict

import faiss
from sentence_transformers import SentenceTransformer

# Load embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# Paths to store FAISS index + mapping
INDEX_PATH = "data/faiss_index.bin"
MAPPING_PATH = "data/doc_id_to_index.pkl"

# In-memory objects
faiss_index = None
doc_id_to_index = {}


def load_faiss_index():
    """Loads an existing FAISS index and document mapping."""
    global faiss_index, doc_id_to_index
    if os.path.exists(INDEX_PATH) and os.path.exists(MAPPING_PATH):
        try:
            faiss_index = faiss.read_index(INDEX_PATH)
            with open(MAPPING_PATH, "rb") as f:
                doc_id_to_index.update(pickle.load(f))
            print(f"[embedder] Loaded FAISS index with {faiss_index.ntotal} vectors.")
        except Exception as e:
            print(f"[embedder] Error loading FAISS index: {e}. Creating new index.")
            dim = 384
            faiss_index = faiss.IndexFlatIP(dim)
    else:
        dim = 384  # Dimension for all-MiniLM-L6-v2
        faiss_index = faiss.IndexFlatIP(dim)  # Cosine similarity
        print("[embedder] Created empty FAISS index.")


def save_faiss_index():
    """Saves the current FAISS index and document mapping."""
    if faiss_index and faiss_index.ntotal > 0:
        faiss.write_index(faiss_index, INDEX_PATH)
        with open(MAPPING_PATH, "wb") as f:
            pickle.dump(doc_id_to_index, f)
        print("[embedder] Saved FAISS index and mapping.")
    else:
        print("[embedder] No data to save in FAISS index.")


def build_faiss_index(documents):
    """
    Builds a new FAISS index from a collection of documents.
    Args:
        documents (dict): A dictionary of documents from the InvertedIndexer.
    """
    global faiss_index, doc_id_to_index

    if faiss_index and faiss_index.ntotal > 0:
        print("[embedder] FAISS index already exists. Clearing old index.")
        faiss_index.reset()
        doc_id_to_index.clear()

    if not documents:
        print("[embedder] No documents to build FAISS index from.")
        return

    print("[embedder] Building FAISS index...")
    doc_texts = []
    doc_ids = []
    for doc_id, doc in documents.items():
        doc_text = f"{doc.get('title', '')} {doc.get('text', '')}"
        doc_texts.append(doc_text)
        doc_ids.append(doc_id)

    # Generate embeddings and add to index
    if doc_texts:
        vectors = model.encode(
            doc_texts, normalize_embeddings=True, show_progress_bar=True
        )
        faiss_index.add(vectors)
        for i, doc_id in enumerate(doc_ids):
            doc_id_to_index[doc_id] = i

    print(f"[embedder] Built FAISS index with {faiss_index.ntotal} vectors.")


def semantic_search(query, top_k=5):
    """Return (doc_id, score) for top_k most similar documents."""
    global faiss_index, doc_id_to_index
    if faiss_index is None or faiss_index.ntotal == 0:
        return []
    query_vec = model.encode([query], normalize_embeddings=True)
    scores, indices = faiss_index.search(query_vec, k=top_k)

    results = []
    index_to_doc_id = {v: k for k, v in doc_id_to_index.items()}
    for i in range(len(indices[0])):
        idx = indices[0][i]
        score = scores[0][i]
        if idx != -1:
            doc_id = index_to_doc_id.get(idx)
            if doc_id:
                results.append((doc_id, float(score)))

    return results


# Load at startup
load_faiss_index()
