from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


def extract_pages(pdf_path: str | Path) -> list[PageText]:
    reader = PdfReader(pdf_path)
    return [
        PageText(page_number=i + 1, text=page.extract_text() or "")
        for i, page in enumerate(reader.pages)
    ]


def extract_text(pdf_path: str | Path) -> str:
    return "\n".join(page.text for page in extract_pages(pdf_path))
