from dataclasses import dataclass, field

import fitz  # PyMuPDF

# A page whose text layer has fewer non-whitespace characters than this is
# treated as a scanned/image page and sent to OCR instead.
MIN_TEXT_LAYER_CHARS = 20
# Embedded images smaller than this (either side, in pixels) are treated as
# decorations (icons, rules, spacers) and skipped when OCR-ing page images.
MIN_IMAGE_SIDE_PX = 100


@dataclass
class EmbeddedImage:
    """An image embedded in a text-layer page, with its position on the page."""
    data: bytes  # PNG
    rect: tuple[float, float, float, float]  # display bbox in page points
    width: int  # pixel dimensions of the source image
    height: int


@dataclass
class PDFPageContent:
    """Content of one PDF page: embedded text blocks, or a rendered PNG for OCR."""
    text_blocks: list[tuple[tuple[float, float, float, float], str]] | None = None
    image: bytes | None = None
    embedded_images: list[EmbeddedImage] = field(default_factory=list)


def count_pdf_pages(data: bytes) -> int:
    """Return the number of pages in a PDF document."""
    with fitz.open(stream=data, filetype="pdf") as doc:
        return doc.page_count


def extract_pdf_pages(
    data: bytes, dpi: int = 200, include_images: bool = False
) -> list[PDFPageContent]:
    """Extract per-page content from a PDF.

    Pages with an embedded text layer return their text blocks directly (no OCR
    needed); image-only pages are rendered to PNG at the given DPI for OCR.
    When include_images is true, images embedded in text-layer pages are also
    extracted (with their page position) so callers can OCR them separately.
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
                content = PDFPageContent(text_blocks=blocks)
                if include_images:
                    content.embedded_images = _extract_embedded_images(doc, page)
                pages.append(content)
                continue
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            pages.append(PDFPageContent(image=pix.tobytes("png")))
    return pages


def _extract_embedded_images(doc: fitz.Document, page: fitz.Page) -> list[EmbeddedImage]:
    """Pull the embedded raster images of a page as PNGs with page positions."""
    images: list[EmbeddedImage] = []
    seen_xrefs: set[int] = set()
    for info in page.get_image_info(xrefs=True):
        xref = info["xref"]
        # xref 0 marks inline images, which PyMuPDF cannot re-encode directly.
        if xref == 0 or xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        if info["width"] < MIN_IMAGE_SIDE_PX or info["height"] < MIN_IMAGE_SIDE_PX:
            continue
        rect = fitz.Rect(info["bbox"])
        if rect.is_empty or rect.is_infinite:
            continue
        pix = fitz.Pixmap(doc, xref)
        if pix.colorspace and pix.colorspace.n > 3:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        if pix.alpha:
            pix = fitz.Pixmap(pix, 0)
        images.append(
            EmbeddedImage(
                data=pix.tobytes("png"),
                rect=(rect.x0, rect.y0, rect.x1, rect.y1),
                width=info["width"],
                height=info["height"],
            )
        )
    return images
