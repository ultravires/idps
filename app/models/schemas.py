from pydantic import BaseModel


class TextBoxResult(BaseModel):
    """Single text box OCR result with bounding box and confidence."""
    text: str
    confidence: float
    bbox: list[tuple[int, int]]  # [[x0,y0], [x1,y1], [x2,y2], [x3,y3]]


class PageResult(BaseModel):
    """OCR result for a single page."""
    page_index: int
    text_blocks: list[TextBoxResult]
    full_text: str  # concatenated text


class OCRResponse(BaseModel):
    """Full OCR response, potentially with multiple pages."""
    pages: list[PageResult]
    total_text: str
    page_count: int
