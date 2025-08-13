# parser.py

from dataclasses import dataclass, field
from typing import Dict, List

import bleach
from bs4 import BeautifulSoup


@dataclass
class ParsedDocument:
    doc_id: int
    url: str
    title: str
    text: str
    images: List[Dict] = field(default_factory=list)


class HTMLParser:
    def __init__(self):
        # A parser for HTML content.
        pass

    def parse(self, html_content: str, url: str) -> ParsedDocument:
        """Parses HTML content to extract title, text, and images."""
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract title
        title = soup.title.string if soup.title else "No Title"

        # Extract and clean text from body
        body_text = " ".join(soup.body.get_text(separator=" ", strip=True).split())
        body_text = bleach.clean(body_text, tags=[], attributes={}, strip=True)

        # Placeholder for images
        images = []
        for img in soup.find_all("img"):
            images.append({"src": img.get("src"), "alt": img.get("alt", "")})

        return ParsedDocument(
            doc_id=0, url=url, title=title, text=body_text, images=images
        )
