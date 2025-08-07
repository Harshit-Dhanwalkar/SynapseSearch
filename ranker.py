import math
import re
from collections import Counter, defaultdict
from functools import lru_cache

import nltk
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer

nltk.download("wordnet")
nltk.download("omw-1.4")


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
        self.HEADING_WEIGHT = 2.0
        self.BODY_WEIGHT = 1.0
        self.ALT_TEXT_WEIGHT = 1.5

        self._build_frequencies()
        self.stemmer = PorterStemmer()

    def _build_frequencies(self):
        """
        Processes documents to build term frequencies and document frequencies.
        This is separate from building the inverted index itself, as the ranker
        needs term counts per document and global document counts.
        """
        for doc_id, doc_data in self.documents.items():
            # Combine title and text for overall term frequency, applying title weight
            title_tokens = tokenize(clean_text(doc_data["title"]))
            body_tokens = tokenize(clean_text(doc_data["text"]))

            # combined list of tokens with title words repeated for weighting
            weighted_tokens = []
            for token in title_tokens:
                weighted_tokens.extend([token] * int(self.TITLE_WEIGHT))
            weighted_tokens.extend(body_tokens)

            term_counts = Counter(weighted_tokens)
            self.term_frequencies[doc_id] = term_counts

            all_unique_doc_terms = set(
                tokenize(clean_text(doc_data["title"] + " " + doc_data["text"]))
            )
            for term in all_unique_doc_terms:
                self.document_frequencies[term] += 1

    def _calculate_tf(self, term, doc_id, field):
        """Calculate TF for specific field with field weighting"""
        doc = self.documents[doc_id]
        text = ""

        if field == "title":
            text = doc["title"]
            weight = self.TITLE_WEIGHT
        elif field == "body":
            text = doc["text"]
            weight = self.BODY_WEIGHT
        elif field == "alt":
            text = " ".join(img.get("alt", "") for img in doc["images"])
            weight = self.ALT_TEXT_WEIGHT

        tokens = tokenize(clean_text(text))
        term_count = tokens.count(term)
        return (term_count / len(tokens)) * weight if tokens else 0.0

    def _calculate_idf(self, term):
        """Calculates Inverse Document Frequency (IDF) for a term."""
        num_docs_with_term = self.document_frequencies.get(term, 0)
        if num_docs_with_term == 0:
            return 0.0
        return math.log(self.corpus_size / (1 + num_docs_with_term)) + 1

    def calculate_tfidf(self, term, doc_id):
        """Combine TF-IDF scores from all fields"""
        tf_title = self._calculate_tf(term, doc_id, "title")
        tf_body = self._calculate_tf(term, doc_id, "body")
        tf_alt = self._calculate_tf(term, doc_id, "alt")

        idf = self._calculate_idf(term)
        return (tf_title + tf_body + tf_alt) * idf

    def _generate_highlighted_snippet(self, text, query_terms, max_length=150):
        """
        Generates a snippet of the text with query terms highlighted.
        Prioritizes showing query terms if they appear early.
        """
        cleaned_text = clean_text(text)
        text_words = cleaned_text.split()

        highlighted_words = []
        # set of query terms for faster lookup
        query_terms_set = set(tokenize(query_terms))

        # first occurrence of any query term to center the snippet
        first_match_index = -1
        for i, word in enumerate(text_words):
            if word in query_terms_set:
                first_match_index = i
                break

        start_index = 0
        if first_match_index != -1:
            start_index = max(0, first_match_index - (max_length // 20))

        end_index = min(len(text_words), start_index + (max_length // 5))

        # If the snippet is too short, expand from the beginning
        if (end_index - start_index) * 5 < max_length:
            end_index = min(len(text_words), start_index + max_length // 5)

        snippet_words = text_words[start_index:end_index]

        # Highlight words in the snippet
        for word in snippet_words:
            if word in query_terms_set:
                highlighted_words.append(f'<span class="highlight">{word}</span>')
            else:
                highlighted_words.append(word)

        highlighted_snippet = " ".join(highlighted_words)

        # Add ellipsis if the original text is longer than the snippet
        if len(cleaned_text) > (end_index - start_index) * 5:
            highlighted_snippet += "..."

        return highlighted_snippet

    @lru_cache(maxsize=1000)
    def _expand_query(self, query):
        """Expand query with synonyms and stem terms"""
        expanded_terms = set()
        try:
            for term in tokenize(clean_text(query)):
                # Add original term
                expanded_terms.add(term)
                # Add stemmed version
                expanded_terms.add(self.stemmer.stem(term))
                # Add synonyms if WordNet is available
                try:
                    for syn in wordnet.synsets(term):
                        for lemma in syn.lemmas():
                            expanded_terms.add(lemma.name().replace("_", " "))
                except:
                    pass  # Skip if WordNet fails
        except Exception as e:
            print(f"Error expanding query: {e}")
            return tokenize(clean_text(query))  # Fallback to basic tokenization

        return list(expanded_terms)

    def rank_documents(self, query):
        """
        Ranks documents based on the sum of TF-IDF scores for query terms.

        Args:
            query (str): The search query.

        Returns:
            list: A list of tuples (doc_id, score, url, title, highlighted_snippet, images)
                  sorted by score in descending order.
        """
        # query_tokens = tokenize(clean_text(query))
        query_tokens = self._expand_query(query)

        if not query_tokens:
            return []

        document_scores = defaultdict(float)

        # candidate documents that contain at least one query term
        candidate_doc_ids = set()
        for term in query_tokens:
            if term in self.inverted_index:
                candidate_doc_ids.update(self.inverted_index[term].keys())

        # Calculate score for each candidate document
        for doc_id in candidate_doc_ids:
            for term in query_tokens:
                document_scores[doc_id] += self.calculate_tfidf(term, doc_id)

        # Prepare results with snippet and sort
        ranked_results = []
        for doc_id, score in document_scores.items():
            doc_data = self.documents.get(doc_id)
            if doc_data:
                # Generate highlighted snippet
                highlighted_snippet = self._generate_highlighted_snippet(
                    doc_data["text"], query
                )

                ranked_results.append(
                    (
                        doc_id,
                        score,
                        doc_data["url"],
                        doc_data["title"],
                        highlighted_snippet,
                        doc_data["images"],
                    )
                )

        ranked_results.sort(key=lambda x: x[1], reverse=True)
        return ranked_results


if __name__ == "__main__":
    # Dummy data for testing
    sample_documents = {
        "0": {
            "url": "http://example.com/page1",
            "title": "The Quick Brown Fox",
            "text": "the quick brown fox jumps over the lazy dog. dog is lazy and sleeps all day.",
            "images": [
                {"src": "http://example.com/fox.jpg", "alt": "a picture of a fox"}
            ],
        },
        "1": {
            "url": "http://example.com/page2",
            "title": "Fast Dog Running",
            "text": "a brown fox is fast and the dog is quick. the dog runs very fast.",
            "images": [
                {"src": "http://example.com/dog.png", "alt": "a fast dog running"}
            ],
        },
        "2": {
            "url": "http://example.com/page3",
            "title": "Warm Sun Shining",
            "text": "the sun is shining and the weather is warm. it's a beautiful day.",
            "images": [],
        },
        "3": {
            "url": "http://example.com/page4",
            "title": "Quick Dog in Sun",
            "text": "the quick dog runs fast in the sun. a very energetic dog.",
            "images": [
                {"src": "http://example.com/dog_sun.gif", "alt": "dog playing in sun"}
            ],
        },
    }

    # Simplified inverted index for testing (would come from Indexer)
    sample_inverted_index = {
        "the": {"0": [0, 6, 10], "1": [5], "2": [0, 6], "3": [0, 5]},
        "quick": {"0": [1], "1": [8], "3": [1]},
        "brown": {"0": [2], "1": [2]},
        "fox": {"0": [3], "1": [3]},
        "jumps": {"0": [4]},
        "over": {"0": [5]},
        "lazy": {"0": [7, 11]},
        "dog": {"0": [8, 9], "1": [6, 12], "3": [2, 6]},
        "is": {"0": [10], "1": [3, 7], "2": [2, 6]},
        "a": {"1": [0], "0": [16]},  # 'a' from alt text
        "fast": {"1": [4, 11], "3": [4]},
        "and": {"1": [4], "2": [5]},
        "sleeps": {"0": [13]},
        "all": {"0": [14]},
        "day": {"0": [15]},
        "runs": {"1": [10], "3": [3]},
        "sun": {"2": [1], "3": [7]},
        "shining": {"2": [3]},
        "weather": {"2": [5]},
        "warm": {"2": [7]},
        "it's": {"2": [8]},
        "beautiful": {"2": [10]},
        "very": {"1": [9], "3": [9]},
        "energetic": {"3": [10]},
        "picture": {"0": [17]},
        "of": {"0": [18], "3": [11]},
        "running": {"1": [13]},
        "playing": {"3": [12]},
    }

    ranker = TFIDFRanker(sample_documents, sample_inverted_index)

    print("--- TF-IDF Ranking Example (with Title Weighting & Image Alt Text) ---")
    query = "quick dog running"
    ranked_docs = ranker.rank_documents(query)

    print(f"Query: '{query}'")
    if ranked_docs:
        for doc_id, score, url, title, snippet, images in ranked_docs:
            print(
                f"  Score: {score:.4f}, URL: {url} (Title: {title}, Snippet: {snippet})"
            )
            if images:
                print(f"    Images found: {len(images)}")
                for img in images[:1]:  # Print first image src
                    print(f"      - Img Src: {img['src']}")
    else:
        print("  No relevant documents found.")
