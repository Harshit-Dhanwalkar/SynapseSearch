import re
import time
import urllib.robotparser
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


def clean_text(text):
    """
    Cleans the extracted text by removing extra whitespace,
    converting to lowercase, and removing non-alphanumeric characters.
    """
    if not text:
        return ""
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r"[^a-z0-9\s]", "", text)
    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text).strip()
    return text


def web_crawler(start_url, robots_parser, max_depth=2, max_pages=50):
    """
    A simple web crawler that fetches pages, extracts text, titles, and images.
    It stores crawled content and URLs.

    Args:
        start_url (str): The URL to start crawling from.
        robots_parser (urllib.robotparser.RobotFileParser): An object to check robots.txt rules.
        max_depth (int): The maximum depth to crawl (number of link clicks).
        max_pages (int): The maximum number of pages to crawl.

    Returns:
        dict: A dictionary where keys are URLs and values are a dict
              containing 'title', 'text', and 'images' (list of {'src', 'alt'}).
              Example: {url: {'title': 'Page Title', 'text': 'cleaned body text', 'images': [{'src': 'url', 'alt': 'desc'}]}}
    """
    queue = [(start_url, 0)]
    visited_urls = set()
    crawled_data = {}

    print(f"Starting crawl from: {start_url}\n")

    while queue and len(crawled_data) < max_pages:
        current_url, depth = queue.pop(0)

        # Skip if already visited or depth limit reached
        if current_url in visited_urls or depth > max_depth:
            continue

        # Check robots.txt rules for the current URL before crawling
        if robots_parser and not robots_parser.can_fetch("*", current_url):
            print(f"Skipping {current_url} as per robots.txt rules.")
            visited_urls.add(current_url)  # Mark as visited to avoid re-adding to queue
            continue

        print(
            f"Crawling (Depth: {depth}, Pages: {len(crawled_data)+1}/{max_pages}): {current_url}"
        )
        visited_urls.add(current_url)

        try:
            # Fetch the page content
            response = requests.get(current_url, timeout=5)
            response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)

            # Parse the HTML content
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract title
            page_title = soup.title.string if soup.title else ""
            cleaned_page_title = clean_text(page_title)

            # Extract visible body text content
            page_body_text = " ".join(
                p.get_text()
                for p in soup.find_all(["p", "h1", "h2", "h3", "li", "div", "span"])
            )
            cleaned_page_body_text = clean_text(page_body_text)

            # Extract images with more aggressive filtering
            images = []
            for img in soup.find_all("img", src=True):
                img_src = urljoin(current_url, img["src"])
                img_alt = clean_text(
                    img.get("alt", "")
                )  # Get alt text, default to empty string

                # Check file extension
                is_design_extension = any(
                    img_src.lower().endswith(ext)
                    for ext in [".ico", ".gif", ".svg", ".webp", ".png"]
                )

                width = img.get("width")
                height = img.get("height")
                is_small_image = False
                if width and height:
                    try:
                        width = int(width)
                        height = int(height)
                        if width > 0 and height > 0:
                            is_small_image = width < 50 and height < 50
                    except ValueError:
                        pass  # Ignore if width/height are not valid numbers

                # Check generic alt text patterns
                generic_alt_patterns = [
                    "logo",
                    "icon",
                    "advertisement",
                    "ad",
                    "spacer",
                    "banner",
                    "menu",
                    "arrow",
                    "button",
                    "share",
                    "facebook",
                    "twitter",
                    "instagram",
                    "linkedin",
                    "youtube",
                    "close",
                    "expand",
                    "collapse",
                    "search",
                    "print",
                    "email",
                    "rss",
                    "background",
                    "decorative",
                    "placeholder",
                    "graph",
                    "chart",
                    "diagram",
                    "figure",
                ]
                is_generic_alt = any(
                    pattern in img_alt for pattern in generic_alt_patterns
                )

                # Check common URL path patterns for non-content images
                url_path = urlparse(img_src).path.lower()
                is_design_url_path = any(
                    pattern in url_path
                    for pattern in [
                        "/static/",
                        "/assets/",
                        "/icons/",
                        "/images/icons/",
                        "/img/icons/",
                        "/css/",
                        "/js/",
                        "/favicons/",
                        "/sprites/",
                        "/thumbs/",
                        "/thumbnails/",
                    ]
                )
                is_common_decorative_name = any(
                    name in url_path
                    for name in [
                        "logo",
                        "icon",
                        "spinner",
                        "loading",
                        "pixel",
                        "blank",
                        "clear",
                    ]
                )

                # Combine all filtering logic
                if (
                    img_src.startswith("http")
                    and not is_design_extension
                    and not is_small_image
                    and not is_generic_alt
                    and not is_design_url_path
                    and not is_common_decorative_name
                ):
                    images.append({"src": img_src, "alt": img_alt})

            if cleaned_page_body_text or cleaned_page_title or images:
                crawled_data[current_url] = {
                    "title": cleaned_page_title,
                    "text": cleaned_page_body_text,
                    "images": images,
                }

            # Find all links on the page
            for link in soup.find_all("a", href=True):
                href = link["href"]
                absolute_url = urljoin(current_url, href)
                parsed_absolute_url = urlparse(absolute_url)

                if (
                    parsed_absolute_url.netloc == urlparse(start_url).netloc
                    and absolute_url not in visited_urls
                    and absolute_url.startswith("http")
                ):
                    queue.append((absolute_url, depth + 1))

            time.sleep(0.1)

        except requests.exceptions.RequestException as e:
            print(f"Error crawling {current_url}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred with {current_url}: {e}")

    print(f"\nFinished crawling {start_url}. Total pages crawled: {len(crawled_data)}")
    return crawled_data


def get_robots_parser(url):
    """Fetches and returns a robotparser object for a given URL with a timeout."""
    robots_url = urljoin(url, "/robots.txt")
    try:
        # Fetch the robots.txt content with a timeout
        response = requests.get(robots_url, timeout=5)
        response.raise_for_status()

        rp = urllib.robotparser.RobotFileParser()
        rp.parse(response.text.splitlines())
        return rp
    except requests.exceptions.RequestException as e:
        print(f"Could not fetch robots.txt for {url}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while parsing robots.txt for {url}: {e}")
        return None


if __name__ == "__main__":
    start_url = "http://quotes.toscrape.com/"
    robots_parser = get_robots_parser(start_url)
    crawled_data = web_crawler(start_url, robots_parser, max_depth=1, max_pages=10)

    print("\n--- Crawled Content Summary ---")
    for url, data in crawled_data.items():
        print(f"URL: {url[:70]}...")
        print(f"Title: {data['title'][:50]}...")
        print(f"Text: {data['text'][:100]}...\n")
        if data["images"]:
            print(f"  Images found: {len(data['images'])}")
            for img in data["images"][:2]:
                print(f"    - Src: {img['src'][:50]}..., Alt: {img['alt'][:50]}...")
            if len(data["images"]) > 2:
                print("    ...")
