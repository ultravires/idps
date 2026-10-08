import os

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.models.schemas import OCRResponse, PageResult
from app.services.ocr_service import get_ocr_service
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
async def ocr_single(file: UploadFile = File(...)):
    """Upload a single image or PDF for OCR."""
    _validate_size(file)
    ext = _validate_filename(file.filename)

    content = await file.read()

    if ext == PDF_EXTENSION:
        return await run_in_threadpool(_ocr_pdf_bytes, content)

    page = await run_in_threadpool(_ocr_image_bytes, content)
    return _build_response([page])


@router.post("/batch", response_model=OCRResponse)
async def ocr_batch(files: list[UploadFile] = File(...)):
    """Upload multiple images/PDFs for batch OCR."""
    all_pages: list[PageResult] = []

    for idx, file in enumerate(files):
        _validate_size(file)
        _validate_filename(file.filename)
        content = await file.read()
        ext = os.path.splitext(file.filename)[1].lower()

        if ext == PDF_EXTENSION:
            pdf_response = await run_in_threadpool(_ocr_pdf_bytes, content)
            all_pages.extend(pdf_response.pages)
        else:
            page = await run_in_threadpool(_ocr_image_bytes, content)
            page.page_index = idx  # re-index for batch
            all_pages.append(page)

    return _build_response(all_pages)


def _ocr_image_bytes(data: bytes) -> PageResult:
    return get_ocr_service().recognize_bytes(data)


def _ocr_pdf_bytes(data: bytes) -> OCRResponse:
    """Process a PDF through page extraction → OCR → response."""
    from app.services.pdf_service import pdf_to_images

    ocr = get_ocr_service()
    image_bytes_list = pdf_to_images(data)
    pages: list[PageResult] = []
    for idx, img_bytes in enumerate(image_bytes_list):
        page = ocr.recognize_bytes(img_bytes)
        page.page_index = idx
        pages.append(page)
    return _build_response(pages)


def _build_response(pages: list[PageResult]) -> OCRResponse:
    total = "\n\n".join(p.full_text for p in pages)
    return OCRResponse(
        pages=pages,
        total_text=total,
        page_count=len(pages),
    )
