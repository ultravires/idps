import os

from fastapi import FastAPI

from app.config import settings
from app.api.ocr_routes import router as ocr_router

app = FastAPI(title=settings.app_name)

os.makedirs(settings.upload_dir, exist_ok=True)

app.include_router(ocr_router)


@app.get("/health")
def health():
    return {"status": "ok"}
