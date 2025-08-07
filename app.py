import logging
import os
from collections import defaultdict
from logging.handlers import RotatingFileHandler
from urllib.parse import urlparse

import bleach
from flask import Flask, jsonify, render_template, request
from flask_caching import Cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from config import START_URLS
from indexer import InvertedIndexer
from ranker import TFIDFRanker

app = Flask(__name__)
limiter = Limiter(
    app=app, key_func=get_remote_address, default_limits=["200 per day", "50 per hour"]
)

# Initialize indexer globally
indexer = InvertedIndexer()
ranker = None  # Will be initialized after indexer loads documents

# Configure cache
cache = Cache(
    config={
        "CACHE_TYPE": "SimpleCache",  # For development
        "CACHE_DEFAULT_TIMEOUT": 300,  # 5 minutes
    }
)
cache.init_app(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File handler which logs even debug messages
file_handler = RotatingFileHandler(
    "search_engine.log", maxBytes=1024 * 1024, backupCount=5
)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

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


def sanitize_query(query):
    """Sanitize user input to prevent XSS attacks."""
    return bleach.clean(query.strip(), tags=[], strip=True)


@app.template_filter("url_domain")
def url_domain_filter(url):
    """Custom filter to extract the domain from a URL."""
    return urlparse(url).netloc


@app.route("/")
def index():
    """Renders the main search page."""
    return render_template("index.html", start_urls=START_URLS)


@app.route("/search")
@cache.cached(query_string=True)
@limiter.limit("10/minute")
# @app.route("/search", methods=["GET"])
def search():
    """Handles search queries and displays results with content type tabs."""
    query = sanitize_query(request.args.get("query", ""))
    # query = request.args.get("query", "").strip()
    # Get the content type from the URL, defaulting to 'all'
    content_type = request.args.get("type", "all").strip()
    results = []

    if not ranker:
        message = (
            "Search engine data not initialized. Please run `python3 main.py` first."
        )
        return render_template(
            "results.html",
            query=query,
            results=[],
            message=message,
            content_type=content_type,
        )

    if query:
        logger.info(f"Received query: '{query}' with type '{content_type}'")
        try:
            ranked_docs = ranker.rank_documents(query)

            for doc_id, score, url, title, snippet, images in ranked_docs:
                results.append(
                    {
                        "url": url,
                        "title": title if title else url,
                        "snippet": snippet,
                        "score": f"{score:.4f}",
                        "images": images,
                    }
                )

            # Filter results based on the content_type parameter
            if content_type == "images":
                # Only keep results that have images
                results = [r for r in results if r["images"]]
            elif content_type == "videos":
                # Placeholder for video results. In a real app, you would filter for videos.
                # Here, we just return an empty list and a message.
                results = []
                message = "Video search is not yet implemented. Please select 'All' or 'Images'."
            elif content_type == "others":
                # Keep results that do not have images
                results = [r for r in results if not r["images"]]

            # Check if there are results after filtering
            if not results and not "message" in locals():
                message = "No results found for your query with the selected filter."
            elif "message" in locals():
                pass  # Message is already set for videos
            else:
                message = ""

        except Exception as e:
            logger.error(f"Search error: {e}")
            results = []
            message = "An error occurred during search. Please try again."
    else:
        message = "Please enter a search query."

    return render_template(
        "results.html",
        query=query,
        results=results,
        message=message,
        content_type=content_type,
    )


@app.route("/autocomplete")
def autocomplete():
    query = request.args.get("query", "").lower().strip()
    suggestions = []

    if not query:
        return jsonify(suggestions)

    if not ranker or not hasattr(ranker, "inverted_index"):
        return jsonify(suggestions)

    try:
        matching_terms = (
            term for term in ranker.inverted_index if term.startswith(query)
        )

        suggestions = sorted(
            matching_terms, key=lambda t: -len(ranker.inverted_index[t])
        )[:5]

    except Exception as e:
        logger.error(f"Autocomplete error: {e}")
        return jsonify(suggestions)

    return jsonify(suggestions)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    app.run(debug=True)
