# appy.py

import logging
import os
import sys
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

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from config import MAX_CRAWL_DEPTH, MAX_CRAWLED_PAGES, START_URLS
from mod.embedder import doc_id_to_index, faiss_index, model
from mod.indexer import InvertedIndexer
from mod.query_processor import QueryProcessor
from mod.ranker import TFIDFRanker

# --- Path Configurations ---
search_engine_data_path = os.path.join(project_root, "data", "search_engine_data.json")
index_data_path = os.path.join(project_root, "data", "inverted_index.json")
faiss_index_path = os.path.join(project_root, "data", "faiss_index.bin")
doc_id_map_path = os.path.join(project_root, "data", "doc_id_map.json")
models_dir = os.path.join(project_root, "models")
stopwords_path = os.path.join(project_root, "data", "stopwords.txt")
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# --- Logging Setup ---
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler = RotatingFileHandler(
    os.path.join(project_root, "search_engine.log"), maxBytes=10000, backupCount=1
)
handler.setFormatter(formatter)
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)

# --- Flask App Initialization ---
app = Flask(__name__, template_folder=os.path.join(project_root, "templates"))
app.config["JSON_SORT_KEYS"] = False
app.config["CACHE_TYPE"] = "simple"
cache = Cache(app)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri="memory://",
    default_limits=["200 per day", "50 per hour"],
)

# --- Global Variables for Data ---
# indexer = InvertedIndexer()
# ranker = None
# query_processor = QueryProcessor()
indexer = None
ranker = None
query_processor = None


def build_result_from_docid(doc_id):
    """
    Builds a result dictionary for semantic search from a document ID.
    """
    global indexer, ranker
    if not indexer or not ranker or doc_id not in indexer.documents:
        return None

    doc_data = indexer.documents[doc_id]
    snippet = ranker._generate_snippet(doc_data["text"], [])

    return {
        "url": doc_data.get("url", ""),
        "title": doc_data.get("title", doc_data.get("url", "")),
        "snippet": snippet,
        "score": "N/A",
        "images": doc_data.get("images", []),
    }


def load_search_engine_data():
    """
    Load search engine components from disk.
    """
    global indexer, ranker, query_processor
    try:
        logger.info("Attempting to load search engine data...")
        # Initialize and load indexer
        logger.info("Initializing InvertedIndexer...")
        indexer = InvertedIndexer()
        if not indexer.load_index():
            logger.warning(
                "No index found. Please run `python3 main.py` first to build the index."
            )
            return False

        # Initialize and load ranker
        logger.info("Initializing ranker...")
        ranker = TFIDFRanker(documents=indexer.documents)

        # Initialize query processor
        logger.info("Initializing QueryProcessor...")
        query_processor = QueryProcessor(
            # indexer.documents,
            # indexer.term_to_doc_map,
            # ranker.vectorizer,
            # ranker.tfidf_matrix,
        )
        logger.info("Search engine data loaded and ranker initialized.")
        return True
    except Exception as e:
        logger.error(f"Failed to load search engine data: {e}")
        return False


def sanitize_query(query):
    """
    Sanitize user input to prevent XSS attacks.
    """
    return bleach.clean(query.strip(), tags=[], strip=True)


# --- Routes ---
@app.template_filter("url_domain")
def url_domain_filter(url):
    """
    Custom filter to extract the domain from a URL.
    """
    return urlparse(url).netloc


@app.route("/")
def index():
    """
    Renders the main search page.
    """
    if indexer and ranker:
        # Pass the document count to the template
        return render_template(
            "index.html", start_urls=START_URLS, indexer_doc_count=indexer.doc_count
        )
    else:
        # If data is not loaded, show a loading page or error
        return (
            "Search engine data is not loaded yet. Please run the crawler first.",
            503,
        )


@app.route("/search")
@cache.cached(query_string=True)
def search():
    """
    Handles search queries and displays results with content type tabs.
    """
    raw_query = request.args.get("query", "")

    if not ranker or not query_processor:
        return render_template(
            "results.html",
            query=raw_query,
            results=[],
            message="Search engine not initialized",
        )

    query = query_processor.process(sanitize_query(raw_query))
    content_type = request.args.get("type", "all").strip()
    mode = request.args.get("mode", "keyword")

    try:
        if mode == "semantic":
            # Semantic search logic
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
            results = [r for r in results if r is not None and r.get("images")]
        elif content_type == "others":
            results = [r for r in results if r is not None and not r.get("images")]

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
@limiter.limit("100 per minute")
def autocomplete():
    global ranker
    query = request.args.get("query", "").lower().strip()

    if not ranker or not query or len(query) < 2:
        return jsonify([])

    try:
        suggestions = ranker.get_autocomplete_suggestions(query)
        # suggestions = [ranker.original_terms_map.get(s, s) for s in stemmed_suggestions]
        return jsonify(suggestions)
    except Exception as e:
        logger.error(f"Autocomplete error: {e}")
        return jsonify([])


# HACK: for debug
@app.route("/ping")
def ping():
    return "pong"


if __name__ == "__main__":
    os.makedirs(os.path.join(project_root, "data"), exist_ok=True)
    load_search_engine_data()
    app.run(debug=True, use_reloader=False)
