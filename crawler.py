import asyncio
import io
import re
import time
from collections import deque
from typing import Any, Dict, List, Tuple
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import aiohttp
import pytesseract
from bs4 import BeautifulSoup
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_DOWNLOAD_TIMEOUT = 10
IMAGE_OCR_MAX_PIXELS = 1600 * 1200
MAX_IMAGE_BYTES = 5 * 1024 * 1024
ENABLE_IMAGE_OCR = True
ROBOTS_PARSERS = {}


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


async def get_robots_parser(session, domain):
    """Asynchronously fetches and returns a RobotFileParser for a domain."""
    if domain in ROBOTS_PARSERS:
        return ROBOTS_PARSERS[domain]

    robots_url = urljoin(f"https://{domain}", "/robots.txt")
    parser = RobotFileParser()
    parser.set_url(robots_url)
    try:
        async with session.get(robots_url, timeout=5) as response:
            if response.status == 200:
                robots_content = await response.text()
                parser.parse(robots_content.splitlines())
    except Exception as e:
        print(f"Could not fetch or parse robots.txt for {domain}: {e}")

    ROBOTS_PARSERS[domain] = parser
    return parser


def can_fetch(parser, url):
    """Checks if a URL can be fetched according to robots.txt rules."""
    if parser:
        return parser.can_fetch("*", url)
    return True


async def fetch_image_bytes_async(session, url: str) -> bytes:
    """
    Asynchronously fetches image bytes using `aiohttp`.
    """
    try:
        async with session.get(url, timeout=IMAGE_DOWNLOAD_TIMEOUT) as response:
            response.raise_for_status()
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > MAX_IMAGE_BYTES:
                return None
            data = await response.read()
            if len(data) > MAX_IMAGE_BYTES:
                return None
            return data
    except aiohttp.ClientError as e:
        print(f"Error fetching image {url}: {e}")
        return None


def ocr_from_pil_image(pil_img: Image.Image) -> str:
    """
    Performs OCR on a PIL Image object (synchronous).
    This function will be run in a separate thread to avoid blocking.
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


async def fetch_image_text_async(session, img_url: str) -> str:
    """
    Asynchronously fetches an image and extracts text using OCR.
    """
    try:
        data = await fetch_image_bytes_async(session, img_url)
        if not data:
            return ""

        pil_img = Image.open(io.BytesIO(data))
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, ocr_from_pil_image, pil_img)

    except Exception as e:
        print(f"Error processing image {img_url}: {e}")
        return ""


def is_content_image_extension(url: str) -> bool:
    """Checks if a URL has a common image file extension."""
    url = url.lower()
    return any(
        url.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]
    )


def sanitize_img_src(src: str, base_url: str) -> str:
    """Joins a relative image source URL with the base URL."""
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


async def process_images_for_page_async(
    session, images: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Asynchronously processes images for a page to extract OCR text.
    """
    if not ENABLE_IMAGE_OCR or not images:
        return images

    tasks = []
    for img in images:
        tasks.append(fetch_image_text_async(session, img["src"]))

    ocr_texts = await asyncio.gather(*tasks)

    for img, text in zip(images, ocr_texts):
        img["ocr_text"] = clean_text(text)

    return images


async def web_crawler(start_url: str, max_depth: int, max_pages: int):
    """
    An asynchronous web crawler that respects robots.txt.
    """
    visited_urls = set()
    pages_crawled = 0
    start_time = time.time()
    crawled_data = {}
    queue = deque([(start_url, 0)])

    parsed_start_url = urlparse(start_url)
    seed_netloc = parsed_start_url.netloc

    async with aiohttp.ClientSession() as session:
        while queue and len(crawled_data) < max_pages:
            current_url, depth = queue.popleft()

            if current_url in visited_urls or depth > max_depth:
                continue

            parsed_url = urlparse(current_url)
            domain = parsed_url.netloc

            robots_parser = await get_robots_parser(session, domain)
            if not can_fetch(robots_parser, current_url):
                print(f"Skipping {current_url} (robots.txt)")
                visited_urls.add(current_url)
                continue

            visited_urls.add(current_url)
            pages_crawled += 1
            print(f"Crawling: {current_url} at depth {depth}")

            try:
                async with session.get(current_url, timeout=10) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        soup = BeautifulSoup(html_content, "html.parser")
                        page_title, page_body_text, images = (
                            extract_page_text_and_images(soup, current_url)
                        )

                        # Asynchronously process images
                        processed_images = await process_images_for_page_async(
                            session, images
                        )

                        crawled_data[current_url] = {
                            "title": page_title,
                            "text": page_body_text,
                            "images": processed_images,
                        }

                        print(f"Successfully crawled and parsed {current_url}")

                        # Find and add new links to the queue
                        links = find_internal_links(soup, current_url, seed_netloc)
                        for link in links:
                            if (
                                link not in visited_urls
                                and urlparse(link).netloc == seed_netloc
                            ):
                                queue.append((link, depth + 1))
                    else:
                        print(
                            f"Failed to fetch {current_url} with status: {response.status}"
                        )
            except Exception as e:
                print(f"Error fetching {current_url}: {e}")

    end_time = time.time()
    print(f"\nCrawl finished in {end_time - start_time:.2f} seconds.")
    print(f"Crawled {len(crawled_data)} pages.")

    return crawled_data


if __name__ == "__main__":

    async def run_example():
        start_url = "http://quotes.toscrape.com/"
        data = await web_crawler(start_url, max_depth=1, max_pages=5)
        for url, d in data.items():
            print(url, d["title"], len(d["images"]), "images")

    asyncio.run(run_example())
