import logging
import os
import time
from multiprocessing import Pool

project_root = os.path.dirname(os.path.abspath(__file__))
nltk_data_dir = os.path.join(project_root, "data", "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
os.environ["NLTK_DATA"] = nltk_data_dir

from parser import HTMLParser, ParsedDocument

from config import MAX_CRAWL_DEPTH, MAX_CRAWLED_PAGES, START_URLS
from crawler import get_robots_parser, web_crawler  # Corrected import statement
from embedder import build_faiss_index, save_faiss_index
from indexer import InvertedIndexer
from ranker import TFIDFRanker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_data_pipeline(start_urls, max_depth, max_pages):
    """
    Orchestrates the crawling and indexing process.
    Loads existing data if available, otherwise crawls and builds new data.
    """
    indexer = InvertedIndexer()
    data_loaded = indexer.load_index()

    if not data_loaded or os.getenv("FORCE_RECRAWL") == "true":
        print("\n" + "-" * 25 + " Starting Web Crawling " + "-" * 25)
        start_time = time.time()

        args = [
            (url, get_robots_parser(url), max_depth, max_pages // len(start_urls))
            for url in start_urls
        ]

        # Multiprocessing Pool to run the crawler in parallel
        all_crawled_data = {}
        try:
            with Pool(processes=min(4, len(start_urls))) as pool:
                results = pool.starmap(web_crawler, args)
            for result in results:
                all_crawled_data.update(result)
        except Exception as e:
            logger.error(f"Error during multiprocessing crawl: {e}")
            return

        print(f"\nTotal pages crawled across all seeds: {len(all_crawled_data)}")
        end_time = time.time()
        print(f"Crawling finished in {end_time - start_time:.2f} seconds.")
        print("-" * 25 + " Finished Web Crawling " + "-" * 25 + "\n")

        print("-" * 25 + " Starting Indexing " + "-" * 25)
        start_time = time.time()

        if os.getenv("FORCE_RECRAWL") == "true" or not data_loaded:
            indexer = InvertedIndexer()  # Re-initialize to clear old data

        for url, content in all_crawled_data.items():
            indexer.add_document(
                url, content["title"], content["text"], content["images"]
            )

        indexer.save_index()
        end_time = time.time()
        print(f"Indexing finished in {end_time - start_time:.2f} seconds.")
        print("-" * 25 + " Finished Indexing " + "-" * 25 + "\n")

        print("-" * 25 + " Starting FAISS Indexing " + "-" * 25)
        start_time = time.time()
        build_faiss_index(indexer.documents)
        save_faiss_index()
        end_time = time.time()
        print(f"FAISS Indexing finished in {end_time - start_time:.2f} seconds.")
        print("-" * 25 + " Finished FAISS Indexing " + "-" * 25 + "\n")

    else:
        print("\n--- Using existing crawled data and index ---")

    print("\nData pipeline complete. Index ready for searching.")


if __name__ == "__main__":
    run_data_pipeline(START_URLS, MAX_CRAWL_DEPTH, MAX_CRAWLED_PAGES)
