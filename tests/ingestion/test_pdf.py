from pypdf import PdfWriter

from src.ingestion.pdf import PageText, extract_pages, extract_text


def _make_pdf(path, num_pages: int) -> None:
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        writer.write(f)


def test_extract_pages_returns_one_entry_per_page(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    _make_pdf(pdf_path, num_pages=3)

    pages = extract_pages(pdf_path)

    assert [p.page_number for p in pages] == [1, 2, 3]
    assert all(p.text == "" for p in pages)


def test_extract_text_joins_pages_in_order(monkeypatch):
    fake_pages = [PageText(1, "hello"), PageText(2, "world")]
    monkeypatch.setattr("src.ingestion.pdf.extract_pages", lambda path: fake_pages)

    assert extract_text("ignored.pdf") == "hello\nworld"
