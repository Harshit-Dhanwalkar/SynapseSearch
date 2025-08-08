# SynapseSearch

SynapseSearch is an intelligent web search engine designed to connect information and deliver highly relevant results by mimicking neural network-like connections. It features a custom web crawler, an inverted indexer, a TF-IDF ranker, and a Flask-based web interface.

## Installation

```git
git clone https://github.com/Harshit-Dhanwalkar/SynapseSearch.git
```

```python
python3.11 -m pip venv .venv
pip install -r requirement.txt
```

```bash
sudo apt update && sudo apt install -y tesseract-ocr
```

## TODO: Future Enhancements (Advanced Search Capabilities) 🚀

- [ ] Image Indexer Enhancement:

  - Goal: Improve the relevance of image search results.
  - Action: Implement more advanced image filtering heuristics (e.g., analyzing image dimensions, file types, and URL patterns more deeply). Explore integrating basic computer vision for content analysis (e.g., using libraries like OpenCV or pre-trained models to identify objects in images).
  - Current Status: Basic image extraction and alt text indexing are in place.

- [ ] Backlinks Processor:

  - Goal: Analyze the link structure of the web to determine page authority.
  - Action: Modify the crawler to record incoming links to each page. Develop a separate module to process this data, identifying which pages link to others.
  - Benefit: Provides data for link-based ranking algorithms.

- [ ] PageRank Algorithm Integration:

  - Goal: Implement a link-based ranking signal similar to Google's original PageRank.
  - Action: Develop an algorithm that calculates a "PageRank" score for each document based on the backlinks data. Integrate this score as a weighting factor in the TF-IDF ranking (or a new ranking model).
  - Benefit: Improves result relevance by favoring authoritative and well-linked pages.

- [ ] Scalable Data Storage (Elasticsearch/MongoDB):

  - Goal: Move beyond JSON file storage for index and documents to handle large datasets.
  - Action: Fully integrate a robust database like Elasticsearch (recommended for search) or MongoDB. This involves replacing file I/O in indexer.py with database client operations.
  - Benefit: Enables indexing millions of documents, faster queries, and better data management.

- [ ] Monitoring System:

  - Goal: Track the health, performance, and usage of the search engine components.
  - Action: Implement logging for crawler progress, indexing speed, and query latency. Consider using tools like Prometheus/Grafana for real-time metrics visualization.
  - Benefit: Essential for debugging, performance optimization, and understanding user behavior in a larger system.

- [ ] Dynamic Content (JavaScript) Crawling:
  - Goal: Enable the crawler to render and extract content from JavaScript-heavy websites.
  - Action: Integrate a headless browser (e.g., Selenium, Playwright) into the crawler.py to execute JavaScript before parsing HTML.
  - Benefit: Significantly expands the range of crawlable web content.
