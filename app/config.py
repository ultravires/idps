from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "IDPS-OCR"
    upload_dir: str = "./uploads"
    ocr_lang: str = "ch"  # Chinese + English
    ocr_det_model_dir: str | None = None
    ocr_rec_model_dir: str | None = None
    # Model names. Defaults are the small (mobile-tier) models: this service
    # runs on CPU (use_gpu=False), where the medium models are much slower.
    # Set to "PP-OCRv6_medium_det" / "PP-OCRv6_medium_rec" for max accuracy.
    ocr_det_model_name: str = "PP-OCRv6_small_det"
    ocr_rec_model_name: str = "PP-OCRv6_small_rec"
    # Cap the long side of the text-detection input. The default pipeline
    # allows up to 4000px, which makes CPU detection take minutes.
    ocr_det_limit_side_len: int = 2000
    max_file_size_mb: int = 50

    model_config = {"env_prefix": "IDPS_"}


settings = Settings()
