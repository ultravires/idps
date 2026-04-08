from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "IDPS-OCR"
    upload_dir: str = "./uploads"
    ocr_lang: str = "ch"  # Chinese + English
    ocr_det_model_dir: str | None = None
    ocr_rec_model_dir: str | None = None
    max_file_size_mb: int = 50

    model_config = {"env_prefix": "IDPS_"}


settings = Settings()
