import asyncio
import io
import re
import time
import urllib.robotparser
from typing import Any, Dict, List, Tuple
from urllib.parse import urljoin, urlparse

import aiohttp
import pytesseract
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_DOWNLOAD_TIMEOUT = 10
IMAGE_OCR_MAX_PIXELS = 1600 * 1200
MAX_IMAGE_BYTES = 5 * 1024 * 1024
ENABLE_IMAGE_OCR = True  # False to skip OCR


def clean_text(text: str) -> str:
    """
    Cleans text by converting to lowercase, removing non-alphanumeric characters,
    and normalizing whitespace.
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_robots_parser(url):
    """
    Fetches and parses robots.txt for a given URL.
    """
    robots_url = urljoin(url, "/robots.txt")
    try:
        response = requests.get(robots_url, timeout=5)
        response.raise_for_status()
        rp = urllib.robotparser.RobotFileParser()
        rp.parse(response.text.splitlines())
        return rp
    except Exception as e:
        print(f"Error fetching robots.txt for {url}: {e}")
        return None


def fetch_image_bytes(url: str) -> bytes:
    """
    Fetches image bytes synchronously using `requests`.
    """
    try:
        response = requests.get(url, timeout=IMAGE_DOWNLOAD_TIMEOUT)
        response.raise_for_status()
        if (
            "Content-Length" in response.headers
            and int(response.headers["Content-Length"]) > MAX_IMAGE_BYTES
        ):
            return None
        data = response.content
        if len(data) > MAX_IMAGE_BYTES:
            return None
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching image {url}: {e}")
        return None


def ocr_from_pil_image(pil_img: Image.Image) -> str:
    """
    Performs OCR on a PIL Image object.
    """
    try:
        img = pil_img.convert("L")
        w, h = img.size
        if w * h > IMAGE_OCR_MAX_PIXELS:
            ratio = (IMAGE_OCR_MAX_PIXELS / (w * h)) ** 0.5
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        return pytesseract.image_to_string(img).strip()
    except Exception as e:
        print(f"Error during OCR: {e}")
        return ""


def fetch_image_text_sync(img_url: str) -> str:
    """
    Synchronously fetches an image and extracts text using OCR.
    """
    try:
        data = fetch_image_bytes(img_url)
        if not data:
            return ""
        pil_img = Image.open(io.BytesIO(data))
        return ocr_from_pil_image(pil_img)
    except Exception as e:
        print(f"Error processing image {img_url}: {e}")
        return ""


def is_content_image_extension(url: str) -> bool:
    """
    Checks if a URL has a common image file extension.
    """
    url = url.lower()
    return any(
        url.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]
    )


def sanitize_img_src(src: str, base_url: str) -> str:
    """
    Joins a relative image source URL with the base URL.
    """
    try:
        return urljoin(base_url, src)
    except Exception:
        return src


def extract_page_text_and_images(
    soup: BeautifulSoup, base_url: str
) -> Tuple[str, str, List[Dict[str, Any]]]:
    """
    Extracts title, body text, and a list of images from a BeautifulSoup object.
    """
    page_title = clean_text(soup.title.string if soup.title else "")
    page_body_text = clean_text(
        " ".join(
            el.get_text(separator=" ")
            for el in soup.find_all(["p", "h1", "h2", "h3", "li", "article", "section"])
        )
    )

    images = []
    for img in soup.find_all("img", src=True):
        img_src = sanitize_img_src(img["src"], base_url)
        img_alt = clean_text(img.get("alt", "") or "")
        url_path = urlparse(img_src).path.lower()

        # filter decorative images
        if img_src.startswith("data:"):
            continue
        if any(
            p in url_path for p in ["/icon", "/logo", "sprite", "/thumb", "/favicon"]
        ):
            continue
        if not is_content_image_extension(img_src) and not img_alt:
            continue

        images.append({"src": img_src, "alt": img_alt, "ocr_text": ""})
    return page_title, page_body_text, images


def find_internal_links(
    soup: BeautifulSoup, current_url: str, seed_netloc: str
) -> List[str]:
    """
    Finds and returns internal links from a BeautifulSoup object.
    """
    links = []
    for link in soup.find_all("a", href=True):
        absolute_url = urljoin(current_url, link["href"])
        if urlparse(absolute_url).netloc == seed_netloc and absolute_url.startswith(
            "http"
        ):
            links.append(absolute_url)
    return links


def process_images_for_page(images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Synchronously processes images for a page to extract OCR text.
    """
    if not images:
        return images
    for img in images:
        img_url = img["src"]
        img["ocr_text"] = clean_text(fetch_image_text_sync(img_url))
    return images


def web_crawler(
    start_url: str,
    robots_parser: urllib.robotparser.RobotFileParser,
    max_depth: int,
    max_pages: int,
) -> Dict[str, Dict[str, Any]]:
    """
    Crawls a website synchronously starting from a given URL.
    This function is designed to be run in a multiprocessing pool.
    """
    queue = [(start_url, 0)]
    visited_urls = set()
    crawled_data = {}
    seed_netloc = urlparse(start_url).netloc

    print(f"Starting crawl from: {start_url}")

    while queue and len(crawled_data) < max_pages:
        current_url, depth = queue.pop(0)

        if current_url in visited_urls or depth > max_depth:
            continue
        if robots_parser and not robots_parser.can_fetch("*", current_url):
            print(f"Skipping {current_url} (robots.txt)")
            visited_urls.add(current_url)
            continue

        print(f"[{len(crawled_data)+1}/{max_pages}] Depth {depth}: {current_url}")
        visited_urls.add(current_url)

        try:
            r = requests.get(current_url, timeout=5)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")

            title, body_text, images = extract_page_text_and_images(soup, current_url)

            if ENABLE_IMAGE_OCR and images:
                images = process_images_for_page(images)

            crawled_data[current_url] = {
                "title": title,
                "text": body_text,
                "images": images,
            }

            for link in find_internal_links(soup, current_url, seed_netloc):
                if link not in visited_urls and link not in [u for u, _ in queue]:
                    queue.append((link, depth + 1))

            time.sleep(0.1)
        except Exception as e:
            print(f"Error fetching {current_url}: {e}")

    print(f"Finished crawling {len(crawled_data)} pages.")
    return crawled_data


if __name__ == "__main__":
    start_url = "http://quotes.toscrape.com/"
    robots_parser = get_robots_parser(start_url)
    data = web_crawler(start_url, robots_parser, max_depth=1, max_pages=5)
    for url, d in data.items():
        print(url, d["title"], len(d["images"]), "images")
