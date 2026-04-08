import fitz  # PyMuPDF


def count_pdf_pages(data: bytes) -> int:
    """Return the number of pages in a PDF document."""
    with fitz.open(stream=data, filetype="pdf") as doc:
        return doc.page_count


def pdf_to_images(data: bytes, dpi: int = 200) -> list[bytes]:
    """Convert each PDF page to a PNG image (as bytes)."""
    images: list[bytes] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            images.append(pix.tobytes("png"))
    return images
