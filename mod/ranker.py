# ranker.py

import math
import os
import re
from bisect import bisect_left
from collections import defaultdict
from functools import lru_cache

import joblib

# Initialize local path for NLTK data to download
local_nltk_data_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "nltk_data"
)
os.environ["NLTK_DATA"] = local_nltk_data_path
os.makedirs(local_nltk_data_path, exist_ok=True)

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print(f"[ranker         ] Intended NLTK download directory: {local_nltk_data_path}")
print(f"[ranker         ] NLTK data search paths: {nltk.data.path}")

# Download NLTK data
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("stopwords", quiet=True)


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
    def __init__(self, documents, stopwords_path=None, rebuild=False):
        """
        Initializes the TFIDFRanker, building or loading the TF-IDF matrix.

        Args:
            documents (dict): {doc_id: {"url": url, "title": title, "text": text, "images": [...]}}
            stopwords_path (str, optional): Path to custom stopwords file. Defaults to None.
            rebuild (bool): If True, forces a rebuild of the TF-IDF matrix.
        """
        print("[ranker          ] TF-IDF ranker initializing...")
        self.documents = documents
        self.doc_ids = list(documents.keys())
        self.vectorizer = None
        self.tfidf_matrix = None
        self.stemmer = PorterStemmer()
        self.stopwords_set = self._load_stopwords(stopwords_path)
        # self.stopwords_set = set(stopwords.words("english"))

        # Data for autocomplete functionality
        self.original_terms_map = {}
        all_terms = set()
        for doc in documents.values():
            text = f"{doc.get('title', '')} {doc.get('text', '')}"
            for token in tokenize(text):
                stemmed_token = self.stemmer.stem(token)
                self.original_terms_map[stemmed_token] = token
                all_terms.add(stemmed_token)

        self.sorted_terms = sorted(list(all_terms))

        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        self.tfidf_path = os.path.join(self.model_dir, "tfidf_matrix.npz")
        self.vectorizer_path = os.path.join(self.model_dir, "tfidf_vectorizer.joblib")

        if os.path.exists(self.tfidf_path) and not rebuild:
            self.load_tfidf()
        else:
            print(
                "[ranker          ] TF-IDF files not found or rebuild is forced. Building new matrix..."
            )
            self.build_tfidf_matrix()
            self.save_tfidf()
        print("[ranker          ] TF-IDF ranker initialized.")

    def build_tfidf_matrix(self):
        """
        Builds the TF-IDF sparse matrix and persists it to disk.
        """
        print("[ranker          ] Building TF-IDF matrix...")
        corpus = [
            f"{doc.get('title', '')} {doc.get('text', '')}"
            for doc in self.documents.values()
        ]

        print(f"[ranker          ] Processing a corpus of {len(corpus)} documents...")
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.custom_tokenizer,
            stop_words="english",  # built-in English stopwords
        )
        # self.vectorizer = TfidfVectorizer(
        #     tokenizer=self.custom_tokenizer,
        #     stop_words=list(self.stopwords_set)  # Convert set to list
        # )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        print("[ranker          ] TF-IDF matrix built and saved.")

    def save_tfidf(self):
        if self.tfidf_matrix is not None:
            save_npz(self.tfidf_path, self.tfidf_matrix)
        if self.vectorizer is not None:
            joblib.dump(self.vectorizer, self.vectorizer_path)
        print("[ranker          ] TF-IDF matrix and vectorizer saved to disk.")

    def load_tfidf(self):
        print("[ranker          ] Loading existing TF-IDF matrix...")
        try:
            self.tfidf_matrix = load_npz(self.tfidf_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            if self.vectorizer.tokenizer is None:
                self.vectorizer.tokenizer = self.custom_tokenizer
            print(
                "[ranker          ] TF-IDF matrix and vectorizer loaded successfully."
            )
        except FileNotFoundError:
            print("[ranker          ] TF-IDF files not found. Rebuilding...")
            self.build_tfidf_matrix()
            self.save_tfidf()
        except Exception as e:
            print(f"[ranker          ] Error loading TF-IDF files: {e}. Rebuilding...")
            self.build_tfidf_matrix()
            self.save_tfidf()

    def custom_tokenizer(self, text):
        return [self.stemmer.stem(t) for t in tokenize(text)]

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

    def _load_stopwords(self, stopwords_path=None):
        """Load stopwords from file or use default NLTK stopwords."""
        stopwords_set = set(stopwords.words("english"))

        if stopwords_path and os.path.exists(stopwords_path):
            try:
                with open(stopwords_path, "r") as f:
                    custom_stopwords = set(f.read().split())
                stopwords_set.update(custom_stopwords)
                print(
                    f"[ranker          ] Loaded {len(custom_stopwords)} additional stopwords from {stopwords_path}"
                )
            except Exception as e:
                print(f"[ranker          ] Error loading custom stopwords: {e}")

        return stopwords_set

    def build_tfidf_matrix(self):
        """
        Builds the TF-IDF sparse matrix and persists it to disk.
        """
        print("[ranker          ] Building TF-IDF matrix...")
        corpus = [
            f"{doc.get('title', '')} {doc.get('text', '')}"
            for doc in self.documents.values()
        ]

        print(f"[ranker          ] Processing a corpus of {len(corpus)} documents...")
        self.vectorizer = TfidfVectorizer(
            tokenizer=self.custom_tokenizer,
            stop_words=list(self.stopwords_set),  # Use our stopwords set
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        print("[ranker          ] TF-IDF matrix built and saved.")

    def get_autocomplete_suggestions(self, query):
        if not query or len(query) < 2:
            return []

        stemmed_query = self.stemmer.stem(query.lower())
        start_index = bisect_left(self.sorted_terms, stemmed_query)

        suggestions = []
        for i in range(start_index, len(self.sorted_terms)):
            term = self.sorted_terms[i]
            if term.startswith(stemmed_query):
                if term in self.original_terms_map:
                    original_term = self.original_terms_map[term]
                    if original_term not in suggestions:
                        suggestions.append(original_term)
                if len(suggestions) >= 5:
                    break
            else:
                break
        return suggestions


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

    # Example for autocomplete
    print("\n--- Autocomplete Suggestions ---")
    autocomplete_query = "qui"
    suggestions = ranker.get_autocomplete_suggestions(autocomplete_query)
    print(f"[ranker          ] Autocomplete for '{autocomplete_query}': {suggestions}")
