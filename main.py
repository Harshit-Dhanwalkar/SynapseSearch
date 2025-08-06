import os
import time

from config import MAX_CRAWL_DEPTH, MAX_CRAWLED_PAGES, START_URLS  # Import from config
from crawler import web_crawler
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

        all_crawled_data = {}
        for url in start_urls:
            print(f"\nInitiating crawl for: {url}")
            crawled_data_from_seed = web_crawler(
                url, max_depth, max_pages // len(start_urls) + 1
            )  # Distribute pages
            all_crawled_data.update(crawled_data_from_seed)
            if len(all_crawled_data) >= max_pages:
                print(
                    f"Reached total max pages ({max_pages}). Stopping further crawls."
                )
                break

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
