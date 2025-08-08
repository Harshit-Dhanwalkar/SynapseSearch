import math
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache

import nltk
from nltk.stem import PorterStemmer

# TODO: Faster ranking (vectorize TF-IDF) Switch from Python counters loops → sklearn TfidfVectorizer + sparse matrix
# from sklearn.feature_extraction.text import TfidfVectorizer
# corpus = [doc['title'] + " " + doc['text'] for doc in documents.values()]
# vectorizer = TfidfVectorizer(max_features=50000)
# X = vectorizer.fit_transform(corpus)  # sparse (n_docs x n_terms)
# persist X (sparse) using scipy.sparse.save_npz

local_nltk_data_path = os.path.join(os.path.dirname(__file__), "data", "nltk_data")
if local_nltk_data_path not in nltk.data.path:
    nltk.data.path.append(local_nltk_data_path)

os.makedirs(local_nltk_data_path, exist_ok=True)

from nltk.corpus import wordnet
from nltk.stem import PorterStemmer

nltk.download("wordnet", download_dir=local_nltk_data_path)
nltk.download("omw-1.4", download_dir=local_nltk_data_path)


def clean_text(text):
    """
    Cleans the extracted text for TF-IDF calculation.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    """Simple tokenizer."""
    return text.split()


class TFIDFRanker:
    def __init__(self, documents, inverted_index):
        """
        Initializes the TFIDFRanker with loaded documents and inverted index.

        Args:
            documents (dict): {doc_id: {"url": url, "title": title, "text": text, "images": [...]}}
            inverted_index (dict): {word: {doc_id: [pos1, pos2, ...], ...}}
        """
        self.documents = documents
        self.inverted_index = inverted_index
        self.corpus_size = len(documents)
        self.term_frequencies = {}  # {doc_id: {term: count}}
        self.document_frequencies = defaultdict(
            int
        )  # {term: count_of_docs_containing_term}

        # Title weighting factor: terms in title get this much more weight
        self.TITLE_WEIGHT = 3.0
        self.HEADING_WEIGHT = 1.5
        self.ALT_TEXT_WEIGHT = 1.0
        self.stemmer = PorterStemmer()
        self._build_frequencies()
        self.document_vectors = {}
        self._build_document_vectors()

    def _build_frequencies(self):
        """
        Build term and document frequencies from documents.
        """
        for doc_id, doc in self.documents.items():
            full_text = []
            full_text.extend(tokenize(clean_text(doc["text"])))
            full_text.extend(
                tokenize(clean_text(doc["title"])) * int(self.TITLE_WEIGHT)
            )

            # Alt text from images with a specific weight
            for image in doc.get("images", []):
                if image.get("alt"):
                    full_text.extend(
                        tokenize(clean_text(image["alt"])) * int(self.ALT_TEXT_WEIGHT)
                    )

            term_counts = Counter(full_text)
            self.term_frequencies[doc_id] = term_counts

            # Update document frequencies
            for term in term_counts:
                self.document_frequencies[term] += 1

    @lru_cache(maxsize=128)
    def _calculate_idf(self, term):
        """Calculates the Inverse Document Frequency for a term."""
        doc_freq = self.document_frequencies.get(term, 0)
        # Avoid division by zero if term is not found
        if doc_freq == 0:
            return 0
        return math.log(self.corpus_size / doc_freq)

    def _calculate_tfidf(self, term, doc_id):
        """Calculates the TF-IDF score for a term in a document."""
        tf = self.term_frequencies.get(doc_id, {}).get(term, 0)
        idf = self._calculate_idf(term)
        return tf * idf

    def _build_document_vectors(self):
        """
        Build TF-IDF vectors for all documents.
        """
        for doc_id, term_counts in self.term_frequencies.items():
            doc_vector = {}
            for term, tf in term_counts.items():
                idf = self._calculate_idf(term)
                doc_vector[term] = tf * idf
            self.document_vectors[doc_id] = doc_vector

    def rank_documents(self, query):
        """
        Ranks documents based on a search query using the TF-IDF model.

        Args:
            query (str): The user's search query.

        Returns:
            list: A sorted list of (doc_id, score, url, title, snippet, images).
        """
        # query_terms = tokenize(clean_text(query))
        query_terms = [self.stemmer.stem(t) for t in tokenize(clean_text(query))]
        document_scores = defaultdict(float)

        for term in query_terms:
            if term in self.inverted_index:
                idf = self._calculate_idf(term)
                # The inverted_index provides the documents containing the term
                for doc_id in self.inverted_index[term]:
                    # The score is simply the sum of TF-IDF scores for all query terms
                    tf = self.term_frequencies.get(doc_id, {}).get(term, 0)
                    document_scores[doc_id] += tf * idf

        #  List of tuples for sorting, including URL and title
        ranked_results = [
            (
                doc_id,
                score,
                self.documents[doc_id]["url"],
                self.documents[doc_id]["title"],
                self.documents[doc_id]["text"],
                self.documents[doc_id]["images"],
            )
            for doc_id, score in document_scores.items()
        ]

        # Sort the results in descending order of score
        ranked_results.sort(key=lambda x: x[1], reverse=True)

        # Truncate snippets for display and highlight query terms
        final_results = []
        for doc_id, score, url, title, text, images in ranked_results:
            snippet = self._generate_snippet(text, query_terms)
            final_results.append((doc_id, score, url, title, snippet, images))

        return final_results

    def get_query_vector(self, query_terms):
        """
        Build a TF-IDF vector for the query.
        """
        query_vector = defaultdict(float)
        for term in query_terms:
            tf = query_terms.count(term)
            idf = self._calculate_idf(term)
            query_vector[term] = tf * idf
        return query_vector

    def rank(self, query):
        """
        Rank documents using Vector Space Model (cosine similarity).
        """
        query_terms = [self.stemmer.stem(t) for t in tokenize(clean_text(query))]
        query_vector = self.get_query_vector(query_terms)

        ranked_results = []
        for doc_id, doc_vector in self.document_vectors.items():
            score = self.calculate_cosine_similarity(doc_vector, query_vector)
            if score > 0:
                ranked_results.append(
                    (
                        doc_id,
                        score,
                        self.documents[doc_id]["url"],
                        self.documents[doc_id]["title"],
                        self._generate_snippet(
                            self.documents[doc_id]["text"], query_terms
                        ),
                        self.documents[doc_id]["images"],
                    )
                )

        ranked_results.sort(key=lambda x: x[1], reverse=True)
        return ranked_results

    def _generate_snippet(self, text, query_terms, snippet_length=200):
        """
        Generates a snippet of text with query terms highlighted.
        """
        words = text.split()

        # Find the first occurrence of any query term to center the snippet
        try:
            first_match_index = min(
                i for i, word in enumerate(words) if word.lower() in query_terms
            )
        except ValueError:
            # No query terms found in text, return a simple truncated snippet
            snippet_text = " ".join(words[:50]) + "..." if len(words) > 50 else text
            return snippet_text

        # Determine the start and end of the snippet
        start_index = max(0, first_match_index - 10)  # Start 10 words before match
        end_index = min(len(words), first_match_index + 30)  # End 30 words after match

        snippet_words = words[start_index:end_index]

        # Highlight the query terms within the snippet
        highlighted_snippet = []
        for word in snippet_words:
            # Use regex to find and replace the term, ignoring case
            if re.search(
                r"\b" + re.escape(word.lower()) + r"\b",
                " ".join(query_terms),
                re.IGNORECASE,
            ):
                highlighted_snippet.append(f'<span class="highlight">{word}</span>')
            else:
                highlighted_snippet.append(word)

        # Ellipsis if the snippet doesn't start or end with the document boundaries
        final_snippet = (
            ("... " if start_index > 0 else "")
            + " ".join(highlighted_snippet)
            + (" ..." if end_index < len(words) else "")
        )
        return final_snippet

    # TODO: Instead of simple window, use scoring per sentence
    # def best_snippet(text, query_terms, k=1):
    #     sentences = re.split(r"(?<=[.!?])\s+", text)
    #     scores = []
    #     for s in sentences:
    #         words = set(re.findall(r"\w+", s.lower()))
    #         overlap = sum(1 for t in query_terms if t in words)
    #         scores.append((overlap, s))
    #     best = sorted(scores, reverse=True)[:k]
    #     return " ... ".join(s for _, s in best)

    def calculate_cosine_similarity(self, doc_vector, query_vector):
        # This is the core of the VSM
        dot_product = sum(
            doc_vector[term] * query_vector[term] for term in query_vector
        )

        # Calculate the magnitude of each vector
        doc_magnitude = math.sqrt(sum(v**2 for v in doc_vector.values()))
        query_magnitude = math.sqrt(sum(v**2 for v in query_vector.values()))

        if doc_magnitude == 0 or query_magnitude == 0:
            return 0

        return dot_product / (doc_magnitude * query_magnitude)


if __name__ == "__main__":
    # --- Example Usage ---
    # Sample documents with titles and image alt text
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

    sample_inverted_index = {
        "cat": {"0": [2, 7, 10]},
        "sleeps": {"0": [3]},
        "quick": {"1": [1], "3": [1]},
        "brown": {"1": [2]},
        "fox": {"1": [3]},
        "jumps": {"1": [4]},
        "over": {"1": [5]},
        "lazy": {"1": [6], "0": [1, 8]},
        "dog": {"1": [8], "3": [2, 5]},
        "the": {"0": [0, 6, 9], "1": [0, 7], "2": [0, 4]},
        "is": {"0": [4], "1": [3, 7], "2": [2, 6]},
        "a": {"1": [0], "0": [5]},
        "fast": {"1": [9], "3": [4]},
        "sun": {"2": [1], "3": [7]},
        "shining": {"2": [3]},
        "weather": {"2": [5]},
        "warm": {"2": [7]},
        "playing": {"3": [6]},
        "picture": {"0": [11], "1": [12]},
        "of": {"0": [12], "1": [13]},
        "running": {"1": [14]},
    }

    ranker = TFIDFRanker(sample_documents, sample_inverted_index)

    print("--- TF-IDF Ranking Example (with Title Weighting & Image Alt Text) ---")
    query = "quick dog running"
    ranked_docs = ranker.rank_documents(query)

    print("\n--- VSM Ranking ---")
    for res in ranker.rank(query):
        print(res)

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
