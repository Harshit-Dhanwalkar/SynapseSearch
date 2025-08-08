import json
import os
import time
from multiprocessing import Pool

from config import MAX_CRAWL_DEPTH, MAX_CRAWLED_PAGES, START_URLS
from crawler import get_robots_parser, web_crawler
from indexer import InvertedIndexer


def run_data_pipeline(start_urls, max_depth, max_pages):
    """
    Orchestrates the crawling and indexing process.
    Loads existing data if available, otherwise crawls and builds new data.
    """
    indexer = InvertedIndexer()

    data_loaded = indexer.load_index()

    if not data_loaded or os.getenv("FORCE_RECRAWL") == "true":
        print("\n--- Starting Web Crawling ---")

        args = [
            (url, get_robots_parser(url), max_depth, max_pages // len(start_urls))
            for url in start_urls
        ]

        with Pool(processes=min(4, len(start_urls))) as pool:
            results = pool.starmap(web_crawler, args)

        all_crawled_data = {}
        for result in results:
            all_crawled_data.update(result)

        print(f"\nTotal pages crawled across all seeds: {len(all_crawled_data)}")

        print("\n--- Building/Updating Index ---")
        if os.getenv("FORCE_RECRAWL") == "true" or not data_loaded:
            indexer = InvertedIndexer()  # Re-initialize to clear old data

        for url, content in all_crawled_data.items():
            indexer.add_document(
                url, content["title"], content["text"], content["images"]
            )

        indexer.save_index()
    else:
        print("\n--- Using existing crawled data and index ---")

    print("\nData pipeline complete. Index ready for searching.")


if __name__ == "__main__":
    run_data_pipeline(START_URLS, MAX_CRAWL_DEPTH, MAX_CRAWLED_PAGES)
