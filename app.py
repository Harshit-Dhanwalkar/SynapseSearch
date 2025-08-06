import os

from flask import Flask, render_template, request

from config import START_URLS
from indexer import InvertedIndexer
from ranker import TFIDFRanker

app = Flask(__name__)

# Initialize indexer globally
indexer = InvertedIndexer()
ranker = None  # Will be initialized after indexer loads documents

# --- Data Loading and Ranker Initialization ---
# This code will execute once when the Flask application starts
print("Attempting to load search engine data...")
if indexer.load_index():
    # Only initialize ranker if documents were successfully loaded
    ranker = TFIDFRanker(indexer.documents, indexer.inverted_index)
    print("Search engine data loaded and ranker initialized.")
else:
    print(
        "No search engine data found. Please run `python3 main.py` first to build the index."
    )
# --- End Data Loading ---


@app.route("/")
def index():
    """Renders the main search page."""
    # Pass START_URLS to the template
    return render_template("index.html", start_urls=START_URLS)


@app.route("/search", methods=["GET"])
def search():
    """Handles search queries and displays results."""
    query = request.args.get("query", "").strip()
    results = []

    if not ranker:
        # Handle case where data wasn't loaded (e.g., first run and main.py wasn't executed)
        message = (
            "Search engine data not initialized. Please run `python3 main.py` first."
        )
        return render_template("results.html", query=query, results=[], message=message)

    if query:
        print(f"Received query: '{query}'")
        # Use the ranker to get sorted results
        # ranked_docs now contains (doc_id, score, url, title, snippet, images)
        ranked_docs = ranker.rank_documents(query)

        for doc_id, score, url, title, snippet, images in ranked_docs:
            results.append(
                {
                    "url": url,
                    "title": title if title else url,  # Use URL if title is empty
                    "snippet": snippet,
                    "score": f"{score:.4f}",
                    "images": images,  # Pass image data to the template
                }
            )

        if not results:
            message = "No results found for your query."
        else:
            message = ""
    else:
        message = "Please enter a search query."

    return render_template(
        "results.html", query=query, results=results, message=message
    )


if __name__ == "__main__":
    # Ensure the 'data' directory exists for persistence
    os.makedirs("data", exist_ok=True)

    # Run Flask app
    app.run(debug=True, port=5000)
