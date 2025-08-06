import json
import re
from collections import defaultdict


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

    def tokenize_and_normalize(self, text):
        """
        Tokenizes text into words and normalizes them (lowercase, remove punctuation).
        """
        if not text:
            return []
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    def add_document(self, url, title, text, images):
        """
        Adds a document to the index.

        Args:
            url (str): The URL of the document.
            title (str): The cleaned title of the document.
            text (str): The cleaned body text content of the document.
            images (list): A list of dictionaries, each with 'src' and 'alt' for images.
        """
        doc_id = str(self.next_doc_id)  # Store doc_id as string for JSON keys
        self.documents[doc_id] = {
            "url": url,
            "title": title,
            "text": text,
            "images": images,
        }

        # Combine title, body text, and image alt text for indexing
        full_content = title + " " + text
        for img in images:
            full_content += " " + img.get(
                "alt", ""
            )  # Add alt text to content for indexing

        tokens = self.tokenize_and_normalize(full_content)

        for position, token in enumerate(tokens):
            self.inverted_index[token][doc_id].append(position)

        self.next_doc_id += 1
        return doc_id

    def get_document(self, doc_id):
        """Retrieves a document by its ID."""
        return self.documents.get(str(doc_id))  # Ensure doc_id is string for lookup

    def search(self, query):
        """
        Performs a basic search on the inverted index.
        Returns documents that contain all query terms (AND logic).
        """
        query_tokens = self.tokenize_and_normalize(query)

        if not query_tokens:
            return []

        # Start with the document IDs for the first query token
        if not self.inverted_index.get(query_tokens[0]):
            return []  # No documents for the first token

        matching_doc_ids = set(self.inverted_index[query_tokens[0]].keys())

        # Intersect with document IDs for subsequent tokens
        for i in range(1, len(query_tokens)):
            token = query_tokens[i]
            if token in self.inverted_index:
                matching_doc_ids.intersection_update(self.inverted_index[token].keys())
            else:
                # If any token is not found, no documents match all tokens
                return []

        # Return the actual document data for the matching IDs
        results = []
        for doc_id in matching_doc_ids:
            results.append(self.get_document(doc_id))
        return results

    def save_index(self):
        """Saves the inverted index and documents to JSON files."""
        # Convert defaultdicts to regular dicts for JSON serialization
        serializable_inverted_index = {
            word: {doc_id: positions for doc_id, positions in postings.items()}
            for word, postings in self.inverted_index.items()
        }

        import os

        os.makedirs(self.data_dir, exist_ok=True)  # Ensure data directory exists

        with open(self.inverted_index_path, "w", encoding="utf-8") as f:
            json.dump(serializable_inverted_index, f, ensure_ascii=False, indent=4)

        with open(self.documents_path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=4)

        print(f"Index and documents saved to {self.data_dir}/")

    def load_index(self):
        """Loads the inverted index and documents from JSON files."""
        try:
            with open(self.inverted_index_path, "r", encoding="utf-8") as f:
                loaded_inverted_index = json.load(f)
                # Reconstruct defaultdicts
                for word, postings in loaded_inverted_index.items():
                    for doc_id, positions in postings.items():
                        self.inverted_index[word][doc_id] = positions

            with open(self.documents_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)

            # Determine next_doc_id based on loaded documents
            if self.documents:
                self.next_doc_id = (
                    max(int(doc_id) for doc_id in self.documents.keys()) + 1
                )
            else:
                self.next_doc_id = 0

            print(f"Index and documents loaded from {self.data_dir}/")
            return True
        except FileNotFoundError:
            print("No existing index or documents found. Starting fresh.")
            return False
        except json.JSONDecodeError as e:
            print(f"Error loading index files: {e}. Starting fresh.")
            # Optionally, delete corrupted files here
            return False


if __name__ == "__main__":
    # Ensure 'data' directory exists for testing persistence
    import os

    os.makedirs("data", exist_ok=True)

    indexer = InvertedIndexer()
    indexer.load_index()  # Try to load existing data

    if not indexer.documents:  # If no data loaded, add some dummy data
        print("\nAdding sample documents...")
        doc1_url = "http://example.com/doc1"
        doc1_title = "The Quick Brown Fox"
        doc1_text = "The quick brown fox jumps over the lazy dog. Dog is lazy."
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

        doc3_url = "http://example.com/doc3"
        doc3_title = "Warm Sun"
        doc3_text = "The sun is shining. The weather is warm and bright."
        doc3_images = []
        indexer.add_document(doc3_url, doc3_title, doc3_text, doc3_images)

        indexer.save_index()  # Save the new data

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
    if results:
        for doc in results:
            print(
                f"  Found in: {doc['url']} (Title: {doc['title'][:20]}..., Text: {doc['text'][:50]}...)"
            )
            if doc["images"]:
                print(f"    Images: {doc['images']}")
    else:
        print("  No documents found for this query.")
