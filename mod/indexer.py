# embedder.py

import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Set


class InvertedIndexer:
    def __init__(self, data_dir="data"):
        """
        Initializes the indexer with paths and data structures.
        """
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # self.data_dir = data_dir
        # self.data_dir_exists = os.path.exists(self.data_dir)
        self.data_dir = os.path.join(self.project_root, "data")
        # self.documents_path = f"{data_dir}/documents.json"
        self.documents_path = os.path.join(self.data_dir, "documents.json")
        # self.inverted_index_path = f"{data_dir}/inverted_index.json"
        self.inverted_index_path = os.path.join(self.data_dir, "inverted_index.json")

        # self.documents = {}
        self.documents: Dict[str, Dict[str, Any]] = {}
        # inverted_index: {word: {doc_id: [pos1, pos2, ...], ...}}
        # self.inverted_index = defaultdict(lambda: defaultdict(list))
        self.inverted_index: defaultdict = defaultdict(lambda: defaultdict(list))
        # documents: {doc_id: {"url": url, "title": title, "text": text, "images": [...]}}
        # self.term_to_doc_map = defaultdict(set)
        self.term_to_doc_map: defaultdict[str, Set[str]] = defaultdict(set)
        self.next_doc_id = 0
        self.doc_count = 0

        os.makedirs(self.data_dir, exist_ok=True)

    def tokenize_and_normalize(self, text: str) -> List[str]:
        """
        Tokenizes text into words and normalizes them (lowercase, remove punctuation).
        """
        if not text:
            return []
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    def add_document(
        self, url: str, title: str, text: str, images: List[Dict[str, str]]
    ):
        """
        Adds a document to the index.

        Args:
            url (str): The URL of the document.
            title (str): The cleaned title of the document.
            text (str): The cleaned body text content of the document.
            images (list): A list of dictionaries, each with 'src' and 'alt' for images.
        """
        doc_id = str(self.next_doc_id)
        full_text = f"{title} {text} {' '.join(img['alt'] for img in images)}"
        tokens = self.tokenize_and_normalize(full_text)

        # Store the document
        self.documents[doc_id] = {
            "url": url,
            "title": title,
            "text": text,
            "images": images,
        }

        # Build inverted index
        for i, token in enumerate(tokens):
            self.inverted_index[token][doc_id].append(i)

        self.next_doc_id += 1
        self.doc_count = len(self.documents)

    def save_index(self):
        """
        Saves the inverted index and documents to files.
        """
        print("[indexer        ] Saving index...")
        # if not self.data_dir_exists:
        #     os.makedirs(self.data_dir, exist_ok=True)
        # with open(self.inverted_index_path, "w") as f:
        #     # Convert defaultdicts to dicts for JSON serialization
        #     serializable_index = {
        #         word: dict(postings) for word, postings in self.inverted_index.items()
        #     }
        #     json.dump(serializable_index, f, indent=2)
        # with open(self.documents_path, "w") as f:
        #     json.dump(self.documents, f, indent=2)
        try:
            serializable_index = {
                word: dict(postings) for word, postings in self.inverted_index.items()
            }
            with open(self.inverted_index_path, "w") as f:
                json.dump(serializable_index, f, indent=4)

            with open(self.documents_path, "w") as f:
                json.dump(self.documents, f, indent=4)
            print("[indexer        ] Index saved successfully.")
        except Exception as e:
            print(f"[indexer        ] Failed to save index: {e}")
        print(f"[indexer        ] Saved index with {len(self.documents)} documents.")

    def load_index(self) -> bool:
        """
        Loads an existing inverted index and documents from files.
        """
        # if not os.path.exists(self.documents_path) or not os.path.exists(
        #     self.inverted_index_path
        # ):
        #     print(
        #         "[indexer        ] Documents or inverted index file not found. Starting with empty index."
        #     )
        #     return False
        if not os.path.exists(self.inverted_index_path) or not os.path.exists(
            self.documents_path
        ):
            print("[indexer        ] Index files not found. Starting with empty index.")
            return False

        print("[indexer        ] Loading existing index...")
        try:
            with open(self.inverted_index_path, "r") as f:
                loaded_index = json.load(f)
                for word, postings in loaded_index.items():
                    self.inverted_index[word] = defaultdict(list, postings)

            with open(self.documents_path, "r") as f:
                self.documents = json.load(f)

            self.doc_count = len(self.documents)
            self.next_doc_id = self.doc_count
            print(
                f"[indexer        ] Loaded existing index with {self.doc_count} documents."
            )
            return True
        except Exception as e:
            print(
                f"[indexer        ] Error loading index: {e}. Starting with empty index."
            )
            self.documents = {}
            self.inverted_index = defaultdict(lambda: defaultdict(list))
            self.doc_count = 0
            return False

    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Retrieves a document by its ID.
        """
        return self.documents.get(doc_id, {})

    def get_doc_ids_for_term(self, term: str) -> Set[str]:
        """
        Retrieves the set of document IDs containing a specific term.
        """
        return self.term_to_doc_map.get(term, set())

    def index_documents(self, crawled_documents):
        """
        Processes and indexes a list of crawled documents.
        """
        for doc in crawled_documents:
            doc_id = str(len(self.documents))
            self.add_document(
                doc["url"], doc["title"], doc["text"], doc.get("images", [])
            )

            # Simple tokenization and indexing logic
            words = doc["text"].lower().split()
            for position, word in enumerate(words):
                if word not in self.inverted_index:
                    self.inverted_index[word] = defaultdict(list)
                self.inverted_index[word][doc_id].append(position)
                self.term_to_doc_map[word].add(doc_id)

        self.doc_count = len(self.documents)

    def search(self, query):
        """Searches the inverted index for the given query."""
        # Simple search - returns document IDs with matching terms
        tokens = self.tokenize_and_normalize(query)
        if not tokens:
            return []

        matching_docs = set()
        for token in tokens:
            if token in self.inverted_index:
                matching_docs.update(self.inverted_index[token].keys())

        return list(matching_docs)


if __name__ == "__main__":
    # Example usage
    indexer = InvertedIndexer(data_dir="temp_data")
    if not indexer.load_index():
        doc1_url = "http://example.com/doc1"
        doc1_title = "The quick brown fox"
        doc1_text = "A quick brown fox jumps over the lazy dog."
        doc1_images = [
            {"src": "http://example.com/fox.jpg", "alt": "a picture of a fox"}
        ]
        indexer.add_document(doc1_url, doc1_title, doc1_text, doc1_images)

        doc2_url = "http://example.com/doc2"
        doc2_title = "Fast Dog"
        doc2_text = "A brown fox is fast. The dog is quick."
        doc2_images = [
            {"src": "http://example.com/dog.png", "alt": "a fast dog running"}
        ]
        indexer.add_document(doc2_url, doc2_title, doc2_text, doc2_images)

        indexer.save_index()

    print("\nInverted Index (first 5 words):")
    count = 0
    for word, postings in indexer.inverted_index.items():
        if count < 5:
            print(f"  '{word}': {dict(postings)}")
            count += 1
        else:
            break

    print("\n--- Search Results ---")
    query = "quick dog"
    results = indexer.search(query)
    print(f"[indexer        ] Query: '{query}'")
    print(f"[indexer        ] Matching Document IDs: {results}")
