# SynapseSearch

SynapseSearch is an intelligent web search engine designed to connect information and deliver highly relevant results by mimicking neural network-like connections. It features a custom web crawler, an inverted indexer, a TF-IDF ranker, and a Flask-based web interface.

## Installation

To set up the project, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Harshit-Dhanwalkar/SynapseSearch.git
    cd SynapseSearch
    ```

2.  **Run the setup script:**
    The `setup.sh` script will create a Python virtual environment, install the necessary dependencies, and set up the `data` directory.

    ```bash
    ./setup.sh setup
    ```

3.  **Activate the virtual environment:**

    ```bash
    source .venv/bin/activate
    ```

4.  **Install system dependencies (for OCR):**
    This step is required for the image OCR feature.

    ```bash
    sudo apt update && sudo apt install -y tesseract-ocr
    ```

---

## Project Status

- **Web Crawler:** A functional, asynchronous crawler that respects `robots.txt` and downloads web pages.
- **Indexer:** An inverted index that tokenizes documents and stores them in JSON files.
- **Ranker:** A TF-IDF based ranker that uses cosine similarity to score documents.
- **Semantic Search:** An experimental feature using `sentence-transformers` and FAISS for vector-based search.
- **Web Interface:** A Flask application with a basic search interface.

---

## TODO: Future Enhancements 🚀

- **Improved Ranking:** Implement a more advanced ranking algorithm, such as a custom PageRank, that utilizes the web's link structure to determine page authority.
- **Scalable Storage:** Transition from local JSON files to a robust database like **Elasticsearch** or **MongoDB** to handle larger datasets efficiently.
- **Monitoring:** Develop a logging and monitoring system to track the health, performance, and usage of the search engine components.
- **Dynamic Content:** Integrate a headless browser to enable the crawler to render and parse content from JavaScript-heavy websites.
- **Image Indexing:** Enhance the image indexing process with more advanced computer vision techniques for better content analysis.

---

[MIT License](https://github.com/Harshit-Dhanwalkar/SynapseSearch/blob/main/LICENSE)
