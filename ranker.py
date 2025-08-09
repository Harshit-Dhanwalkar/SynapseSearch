import math
import os
import re
from collections import defaultdict
from functools import lru_cache

# Set the NLTK_DATA environment variable *before* importing NLTK
local_nltk_data_path = os.path.join(os.path.dirname(__file__), "data", "nltk_data")
os.environ['NLTK_DATA'] = local_nltk_data_path
os.makedirs(local_nltk_data_path, exist_ok=True)

import joblib
import nltk
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print(f"Intended NLTK download directory: {local_nltk_data_path}")
print(f"NLTK data search paths: {nltk.data.path}")

nltk.download("wordnet", download_dir=local_nltk_data_path, quiet=True)
nltk.download("omw-1.4", download_dir=local_nltk_data_path, quiet=True)


def clean_text(text):
    """
    Cleans the extracted text.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    """Simple tokenizer."""
    return text.split()


class TFIDFRanker:
    def __init__(self, documents, rebuild=False):
        """
        Initializes the TFIDFRanker, building or loading the TF-IDF matrix.

        Args:
            documents (dict): {doc_id: {"url": url, "title": title, "text": text, "images": [...]}}
            rebuild (bool): If True, forces a rebuild of the TF-IDF matrix.
        """
        self.documents = documents
        self.doc_ids = list(documents.keys())
        self.corpus_size = len(documents)
        self.vectorizer_path = "data/tfidf_vectorizer.joblib"
        self.matrix_path = "data/tfidf_matrix.npz"
        self.stemmer = PorterStemmer()

        # Check for persistence files to decide whether to build or load
        if rebuild or not (
            os.path.exists(self.matrix_path) and os.path.exists(self.vectorizer_path)
        ):
            self._build_sparse_matrix()
        else:
            self._load_sparse_matrix()

    def _build_sparse_matrix(self):
        """
        Builds the TF-IDF sparse matrix and persists it to disk.
        """
        print("Building TF-IDF matrix...")
        corpus = [
            clean_text(doc.get("title", "") + " " + doc.get("text", ""))
            for doc in self.documents.values()
        ]

        self.vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english", tokenizer=tokenize, max_features=50000
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

        # Save the vectorizer and the sparse matrix for later use
        joblib.dump(self.vectorizer, self.vectorizer_path)
        save_npz(self.matrix_path, self.tfidf_matrix)
        print("TF-IDF matrix built and saved.")

    def _load_sparse_matrix(self):
        """
        Loads a pre-built TF-IDF sparse matrix from disk.
        """
        print("Loading pre-built TF-IDF matrix...")
        self.vectorizer = joblib.load(self.vectorizer_path)
        self.tfidf_matrix = load_npz(self.matrix_path)
        print("TF-IDF matrix loaded.")

    def rank(self, query):
        """
        Ranks documents using the Vector Space Model (cosine similarity)
        against the pre-built TF-IDF sparse matrix.

        Args:
            query (str): The user's search query.

        Returns:
            list: A sorted list of tuples: (doc_id, score, url, title, snippet, images).
        """
        cleaned_query = clean_text(query)
        query_vector = self.vectorizer.transform([cleaned_query])

        # Calculate cosine similarity with the pre-built matrix
        cosine_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        ranked_results = []
        for i, score in enumerate(cosine_scores):
            if score > 0:
                doc_id = self.doc_ids[i]
                doc_data = self.documents[doc_id]
                query_terms = tokenize(cleaned_query)
                snippet = self._generate_snippet(doc_data["text"], query_terms)
                ranked_results.append(
                    (
                        doc_id,
                        score,
                        doc_data["url"],
                        doc_data["title"],
                        snippet,
                        doc_data["images"],
                    )
                )

        ranked_results.sort(key=lambda x: x[1], reverse=True)
        return ranked_results

    def _generate_snippet(self, text, query_terms):
        """
        Generates a snippet of text with query terms highlighted.
        """
        words = text.split()
        stemmed_query_terms = {self.stemmer.stem(t) for t in query_terms}

        try:
            first_match_index = next(
                i
                for i, word in enumerate(words)
                if self.stemmer.stem(word.lower()) in stemmed_query_terms
            )
        except StopIteration:
            return " ".join(words[:50]) + "..." if len(words) > 50 else text

        start_index = max(0, first_match_index - 10)
        end_index = min(len(words), first_match_index + 30)

        snippet_words = words[start_index:end_index]
        highlighted_snippet = []
        for word in snippet_words:
            stemmed_word = self.stemmer.stem(word.lower())
            if stemmed_word in stemmed_query_terms:
                highlighted_snippet.append(f'<span class="highlight">{word}</span>')
            else:
                highlighted_snippet.append(word)

        final_snippet = (
            ("... " if start_index > 0 else "")
            + " ".join(highlighted_snippet)
            + (" ..." if end_index < len(words) else "")
        )
        return final_snippet


if __name__ == "__main__":
    # --- Example Usage ---
    sample_documents = {
        "0": {
            "url": "http://example.com/doc0",
            "title": "A Lazy Cat",
            "text": "The cat sleeps. It is a big cat and it sleeps all day.",
            "images": [
                {"src": "http://example.com/cat.jpg", "alt": "A picture of a lazy cat"},
                {"src": "http://example.com/cat2.jpg", "alt": "Another image of a cat"},
            ],
        },
        "1": {
            "url": "http://example.com/doc1",
            "title": "A Quick Brown Fox",
            "text": "The quick brown fox jumps over the lazy dog. The fox is very fast.",
            "images": [
                {"src": "http://example.com/fox.jpg", "alt": "A picture of a fox"},
                {
                    "src": "http://example.com/fox_running.jpg",
                    "alt": "The fox running fast",
                },
            ],
        },
        "2": {
            "url": "http://example.com/doc2",
            "title": "Warm Sun",
            "text": "The sun is shining. The weather is warm and bright.",
            "images": [],
        },
        "3": {
            "url": "http://example.com/doc3",
            "title": "Energetic Dog",
            "text": "A very quick dog runs and is playing. The sun is out.",
            "images": [
                {
                    "src": "http://example.com/dog.png",
                    "alt": "An energetic dog playing",
                },
            ],
        },
    }

    # Keeping it here for example completeness, but it's not used in ranker
    sample_inverted_index = {}

    # Initialize the ranker
    ranker = TFIDFRanker(sample_documents)

    print("\n--- VSM Ranking with TF-IDF Sparse Matrix ---")
    query = "quick dog running"
    ranked_docs = ranker.rank(query)

    print(f"Query: '{query}'")
    for doc_id, score, url, title, snippet, images in ranked_docs:
        print(f"  Doc ID: {doc_id}")
        print(f"    URL: {url}")
        print(f"    Title: {title}")
        print(f"    Score: {score:.4f}")
        print(f"    Snippet: {snippet}")
        if images:
            print(f"    Images found: {len(images)}")
        print("-" * 20)
