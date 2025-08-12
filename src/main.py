import asyncio
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from config import MAX_CRAWL_DEPTH, MAX_CRAWLED_PAGES, START_URLS

# project_root = os.path.dirname(os.path.abspath(__file__))
nltk_data_dir = os.path.join(project_root, "data", "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
os.environ["NLTK_DATA"] = nltk_data_dir

from mod.crawler import get_robots_parser, web_crawler
from mod.embedder import build_faiss_index, save_faiss_index
from mod.indexer import InvertedIndexer
from mod.parser import HTMLParser, ParsedDocument
from mod.ranker import TFIDFRanker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_crawlers_async(start_urls, max_depth, max_pages_per_url):
    """
    Runs multiple async web_crawler tasks concurrently.
    """
    tasks = []
    for url in start_urls:
        tasks.append(web_crawler(url, max_depth, max_pages_per_url))

    results = await asyncio.gather(*tasks)

    all_crawled_data = {}
    for result in results:
        all_crawled_data.update(result)

    return all_crawled_data


def run_data_pipeline(start_urls, max_depth, max_pages):
    """
    Orchestrates the crawling and indexing process.
    """
    indexer = InvertedIndexer()
    data_loaded = indexer.load_index()

    if not data_loaded or os.getenv("FORCE_RECRAWL") == "true":
        print("\n" + "-" * 25 + " Starting Web Crawling " + "-" * 25)
        start_time = time.time()

        pages_per_url = max_pages // len(start_urls)
        all_crawled_data = asyncio.run(
            run_crawlers_async(start_urls, max_depth, pages_per_url)
        )

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
                url,
                content.get("title", ""),
                content.get("text", ""),
                content.get("images", []),
            )

        indexer.save_index()
        end_time = time.time()
        print(f"Indexing finished in {end_time - start_time:.2f} seconds.")
        print("-" * 25 + " Finished Indexing " + "-" * 25 + "\n")

        print("-" * 25 + " Starting FAISS Indexing " + "-" * 25)

        start_time = time.time()
        # documents_list = [
        #     {"url": url, "title": doc["title"], "text": doc["text"]}
        #     for url, doc in indexer.documents.items()
        # ]
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
