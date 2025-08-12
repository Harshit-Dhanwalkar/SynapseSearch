import json
import os
import re
from collections import defaultdict
from typing import Dict, List


class InvertedIndexer:
    def __init__(self, data_dir="data"):
        # inverted_index: {word: {doc_id: [pos1, pos2, ...], ...}}
        self.inverted_index = defaultdict(lambda: defaultdict(list))
        # documents: {doc_id: {"url": url, "title": title, "text": text, "images": [...]}}
        self.documents = {}
        self.next_doc_id = 0
        self.data_dir = data_dir
        self.inverted_index_path = f"{data_dir}/inverted_index.json"
        self.documents_path = f"{data_dir}/documents.json"
        self.data_dir_exists = os.path.exists(self.data_dir)

    def tokenize_and_normalize(self, text: str) -> List[str]:
        """
        Tokenizes text into words and normalizes them (lowercase, remove punctuation).
        """
        if not text:
            return []
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    def add_document(self, url: str, title: str, text: str, images: List[Dict]):
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

    def save_index(self):
        """Saves the inverted index and documents to files."""
        if not self.data_dir_exists:
            os.makedirs(self.data_dir, exist_ok=True)
        with open(self.inverted_index_path, "w") as f:
            # Convert defaultdicts to dicts for JSON serialization
            serializable_index = {
                word: dict(postings) for word, postings in self.inverted_index.items()
            }
            json.dump(serializable_index, f, indent=2)
        with open(self.documents_path, "w") as f:
            json.dump(self.documents, f, indent=2)
        print(f"[indexer] Saved index with {len(self.documents)} documents.")

    def load_index(self) -> bool:
        """Loads an existing inverted index and documents from files."""
        if not self.data_dir_exists:
            print("[indexer] Data directory not found. Starting with empty index.")
            return False

        if os.path.exists(self.inverted_index_path) and os.path.exists(
            self.documents_path
        ):
            try:
                with open(self.inverted_index_path, "r") as f:
                    # Load the JSON and convert back to defaultdict
                    loaded_index = json.load(f)
                    for word, postings in loaded_index.items():
                        self.inverted_index[word] = defaultdict(list, postings)
                with open(self.documents_path, "r") as f:
                    self.documents = json.load(f)
                self.next_doc_id = len(self.documents)
                print(
                    f"[indexer] Loaded existing index with {len(self.documents)} documents."
                )
                return True
            except Exception as e:
                print(f"[indexer] Error loading index: {e}. Starting with empty index.")
                return False
        return False

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
    print(f"Query: '{query}'")
    print(f"Matching Document IDs: {results}")
