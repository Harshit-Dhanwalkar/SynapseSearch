import logging
import os
from bisect import bisect_left
from collections import defaultdict
from logging.handlers import RotatingFileHandler
from urllib.parse import urlparse

import bleach
import faiss
from flask import Flask, jsonify, render_template, request
from flask_caching import Cache
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from config import START_URLS
from embedder import doc_id_to_index, faiss_index, model
from indexer import InvertedIndexer
from query_processor import QueryProcessor
from ranker import TFIDFRanker

app = Flask(__name__)

# Configure cache
cache = Cache(
    config={
        "CACHE_TYPE": "SimpleCache",
        "CACHE_DEFAULT_TIMEOUT": 300,
    }
)
cache.init_app(app)

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri="memory://",
    default_limits=["200 per day", "50 per hour"],
)

# Initialize indexer globally
indexer = InvertedIndexer()
ranker = None
query_processor = QueryProcessor()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
file_handler = RotatingFileHandler(
    "search_engine.log", maxBytes=1024 * 1024, backupCount=5
)
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# This code will execute once when the Flask application starts
print("Attempting to load search engine data...")
if indexer.load_index():
    ranker = TFIDFRanker(indexer.documents, indexer.inverted_index)
    print("Search engine data loaded and ranker initialized.")
else:
    print(
        "No search engine data found. Please run `python3 main.py` first to build the index."
    )


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
def search():
    """Handles search queries and displays results with content type tabs."""
    raw_query = request.args.get("query", "")
    query = query_processor.process(sanitize_query(raw_query))
    content_type = request.args.get("type", "all").strip()
    mode = request.args.get("mode", "keyword")

    try:
        if not ranker:
            return render_template(
                "results.html",
                query=query,
                results=[],
                message="Search engine not initialized",
            )

        if mode == "semantic":
            # Semantic search logic (unchanged)
            q_vec = model.encode([raw_query], convert_to_numpy=True)
            faiss.normalize_L2(q_vec)
            D, I = faiss_index.search(q_vec, k=50)
            doc_ids = [doc_id_to_index[i] for i in I[0]]
            results = [build_result_from_docid(did) for did in doc_ids]
        else:
            ranked_docs = ranker.rank(query)
            results = []

            for doc in ranked_docs:
                result = {
                    "url": doc[2],
                    "title": doc[3] or doc[2],
                    "snippet": doc[4],
                    "score": f"{doc[1]:.2f}",
                    "images": doc[5] if len(doc) > 5 else [],
                }
                results.append(result)

        if content_type == "images":
            results = [r for r in results if r.get("images")]
        elif content_type == "others":
            results = [r for r in results if not r.get("images")]

        return render_template(
            "results.html",
            query=query,
            results=results,
            content_type=content_type,
            message="" if results else "No results found",
        )

    except Exception as e:
        logger.error(f"Search error: {e}")
        return render_template(
            "results.html",
            query=sanitize_query(raw_query),
            results=[],
            content_type=content_type,
            message="An error occurred during search",
        )


@app.route("/autocomplete")
def autocomplete():
    query = request.args.get("query", "").lower().strip()
    suggestions = []

    if not query or not ranker or not hasattr(ranker, "sorted_terms"):
        return jsonify(suggestions)

    try:
        start_index = bisect_left(ranker.sorted_terms, query)
        for i in range(start_index, len(ranker.sorted_terms)):
            term = ranker.sorted_terms[i]
            if term.startswith(query):
                suggestions.append(term)
                if len(suggestions) >= 5:
                    break
            else:
                # Stop if we no longer have a prefix match
                break
    except Exception as e:
        logger.error(f"Autocomplete error: {e}")

    return jsonify(suggestions)


# HACK: for debug
@app.route("/ping")
def ping():
    return "pong"


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    app.run(debug=True, use_reloader=False)
