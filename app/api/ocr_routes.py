import os

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.models.schemas import OCRResponse, PageResult, TextBoxResult
from app.services.ocr_service import get_ocr_service
from app.services.pdf_service import EmbeddedImage, extract_pdf_pages
from app.config import settings

router = APIRouter(prefix="/api/v1/ocr", tags=["ocr"])


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
PDF_EXTENSION = ".pdf"


def _validate_size(file):
    """Check file size against configured limit."""
    file.file.seek(0, 2)
    size_bytes = file.file.tell()
    file.file.seek(0)
    limit = settings.max_file_size_mb * 1024 * 1024
    if size_bytes > limit:
        raise HTTPException(
            status_code=413, detail=f"File exceeds {settings.max_file_size_mb}MB limit"
        )


def _validate_filename(filename: str):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS and ext != PDF_EXTENSION:
        raise HTTPException(
            status_code=400, detail=f"Unsupported file type: {ext}"
        )
    return ext


@router.post("/single", response_model=OCRResponse)
async def ocr_single(file: UploadFile = File(...), ocr_images: bool = Form(False)):
    """Upload a single image or PDF for OCR.

    For PDFs the embedded text layer is used directly. Pass ocr_images=true to
    additionally OCR images embedded in text pages (slower, one OCR call per
    image); scanned pages are always OCR'd as a whole.
    """
    _validate_size(file)
    ext = _validate_filename(file.filename)

    content = await file.read()

    if ext == PDF_EXTENSION:
        return await run_in_threadpool(_ocr_pdf_bytes, content, ocr_images)

    page = await run_in_threadpool(_ocr_image_bytes, content)
    return _build_response([page])


@router.post("/batch", response_model=OCRResponse)
async def ocr_batch(
    files: list[UploadFile] = File(...), ocr_images: bool = Form(False)
):
    """Upload multiple images/PDFs for batch OCR."""
    all_pages: list[PageResult] = []

    for idx, file in enumerate(files):
        _validate_size(file)
        _validate_filename(file.filename)
        content = await file.read()
        ext = os.path.splitext(file.filename)[1].lower()

        if ext == PDF_EXTENSION:
            pdf_response = await run_in_threadpool(_ocr_pdf_bytes, content, ocr_images)
            all_pages.extend(pdf_response.pages)
        else:
            page = await run_in_threadpool(_ocr_image_bytes, content)
            page.page_index = idx  # re-index for batch
            all_pages.append(page)

    return _build_response(all_pages)


def _ocr_image_bytes(data: bytes) -> PageResult:
    return get_ocr_service().recognize_bytes(data)


def _ocr_pdf_bytes(data: bytes, ocr_images: bool = False) -> OCRResponse:
    """Process a PDF: use the embedded text layer when present, OCR the rest.

    Digitally-born PDFs (most ebooks, exports) already carry a text layer, so
    extraction takes under a second; only scanned/image pages go through OCR.
    The OCR models are loaded lazily — never for a fully text-based PDF.
    With ocr_images=True, images embedded in text pages are OCR'd individually
    and merged into the page result (each OCR call adds seconds).
    """
    ocr = None
    pages: list[PageResult] = []
    for idx, content in enumerate(
        extract_pdf_pages(data, dpi=settings.pdf_ocr_dpi, include_images=ocr_images)
    ):
        if content.text_blocks is not None:
            page = _page_from_text_layer(idx, content.text_blocks)
            if content.embedded_images:
                if ocr is None:
                    ocr = get_ocr_service()
                _ocr_embedded_images(ocr, page, content.embedded_images)
        else:
            if ocr is None:
                ocr = get_ocr_service()
            page = ocr.recognize_bytes(content.image)
            page.page_index = idx
        pages.append(page)
    return _build_response(pages)


def _page_from_text_layer(
    page_index: int, blocks: list[tuple[tuple[float, float, float, float], str]]
) -> PageResult:
    text_blocks = [
        TextBoxResult(
            text=text,
            confidence=1.0,
            bbox=[
                (int(x0), int(y0)),
                (int(x1), int(y0)),
                (int(x1), int(y1)),
                (int(x0), int(y1)),
            ],
        )
        for (x0, y0, x1, y1), text in blocks
    ]
    return PageResult(
        page_index=page_index,
        text_blocks=text_blocks,
        full_text="\n".join(b.text for b in text_blocks),
    )


def _ocr_embedded_images(ocr, page: PageResult, images: list[EmbeddedImage]) -> None:
    """OCR a page's embedded images and merge results into the page.

    OCR boxes come back in image pixel coordinates; they are remapped to page
    points via each image's display rect so all bboxes in the page share the
    text layer's coordinate system.
    """
    extra_text: list[str] = []
    for img in images:
        result = ocr.recognize_bytes(img.data)
        x0, y0, x1, y1 = img.rect
        sx = (x1 - x0) / img.width
        sy = (y1 - y0) / img.height
        for block in result.text_blocks:
            block.bbox = [(int(x0 + x * sx), int(y0 + y * sy)) for x, y in block.bbox]
            page.text_blocks.append(block)
        if result.full_text.strip():
            extra_text.append(result.full_text)
    if extra_text:
        sep = "\n" if page.full_text else ""
        page.full_text += sep + "\n".join(extra_text)


def _build_response(pages: list[PageResult]) -> OCRResponse:
    total = "\n\n".join(p.full_text for p in pages)
    return OCRResponse(
        pages=pages,
        total_text=total,
        page_count=len(pages),
    )
