from dataclasses import dataclass

import fitz  # PyMuPDF

# A page whose text layer has fewer non-whitespace characters than this is
# treated as a scanned/image page and sent to OCR instead.
MIN_TEXT_LAYER_CHARS = 20


@dataclass
class PDFPageContent:
    """Content of one PDF page: embedded text blocks, or a rendered PNG for OCR."""
    text_blocks: list[tuple[tuple[float, float, float, float], str]] | None = None
    image: bytes | None = None


def count_pdf_pages(data: bytes) -> int:
    """Return the number of pages in a PDF document."""
    with fitz.open(stream=data, filetype="pdf") as doc:
        return doc.page_count


def extract_pdf_pages(data: bytes, dpi: int = 200) -> list[PDFPageContent]:
    """Extract per-page content from a PDF.

    Pages with an embedded text layer return their text blocks directly (no OCR
    needed); image-only pages are rendered to PNG at the given DPI for OCR.
    Exactly one of PDFPageContent's fields is set per page.
    """
    pages: list[PDFPageContent] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            blocks = [
                ((b[0], b[1], b[2], b[3]), b[4].strip())
                for b in page.get_text("blocks")
                if b[6] == 0 and b[4].strip()
            ]
            if sum(len(text) for _, text in blocks) >= MIN_TEXT_LAYER_CHARS:
                pages.append(PDFPageContent(text_blocks=blocks))
                continue
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            pages.append(PDFPageContent(image=pix.tobytes("png")))
    return pages
