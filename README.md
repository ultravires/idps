# IDPS-OCR 智能文档处理系统

基于 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 的智能文档 OCR 识别服务，支持图片/PDF 的文本提取、边界框定位和置信度评估。

## 技术架构

| 层级 | 技术选型 | 说明 |
|---|---|---|
| OCR 引擎 | PaddleOCR v3.7 + PaddlePaddle | PP-OCRv6 模型，单模型支持中英日韩等 50 种语言 |
| Web 框架 | FastAPI | 异步 REST API |
| PDF 处理 | PyMuPDF | PDF 逐页转图片 |
| 数据校验 | Pydantic v2 | 请求/响应模型 |
| 图像处理 | OpenCV + NumPy | 图片解码与预处理 |

### 项目结构

```
idps-ocr/
├── app/
│   ├── main.py              # FastAPI 入口
│   ├── config.py            # 配置管理（环境变量 IDPS_* 前缀）
│   ├── models/
│   │   └── schemas.py       # Pydantic 响应模型
│   ├── services/
│   │   ├── ocr_service.py   # PaddleOCR 封装（懒加载单例）
│   │   └── pdf_service.py   # PDF 逐页转图片
│   └── api/
│       └── ocr_routes.py    # OCR REST 路由
├── .venv/                   # Python 虚拟环境
├── requirements.txt
└── uploads/                 # 上传目录（自动创建）
```

## 如何运行

### 1. 创建虚拟环境

```bash
# Python 3.12 是必须版本（PaddlePaddle 不支持 3.13+）
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install paddleocr paddlepaddle fastapi uvicorn pydantic python-multipart PyMuPDF pydantic-settings
```

> **平台说明：**
> - Linux (CPU): `pip install paddlepaddle`
> - Linux (GPU/CUDA): `pip install paddlepaddle-gpu`
> - macOS Apple Silicon: paddlepaddle 已提供 wheel 支持，可直接安装

首次运行时会自动下载 OCR 模型文件（约 50MB），保存在 `~/.paddlex/official_models/`。

### 3. 启动服务

```bash
# 开发模式（代码变更自动重启）
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 生产模式
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

服务启动后访问：
- API 文档（Swagger UI）：http://localhost:8000/docs
- 健康检查：http://localhost:8000/health

### 4. 环境变量

通过 `IDPS_` 前缀的配置项可自定义行为：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `IDPS_OCR_LANG` | `ch` | 识别语言，`ch`=中英文，`en`=仅英文 |
| `IDPS_OCR_DET_MODEL_NAME` | `PP-OCRv6_small_det` | 检测模型。CPU 部署请保持 small；追求最高准确率可设为 `PP-OCRv6_medium_det`（大图单张耗时可达数分钟） |
| `IDPS_OCR_REC_MODEL_NAME` | `PP-OCRv6_small_rec` | 识别模型，同上，可设为 `PP-OCRv6_medium_rec` |
| `IDPS_OCR_DET_LIMIT_SIDE_LEN` | `2000` | 检测模型输入的长边上限（像素）。过大图会先缩放到该尺寸再检测，识别仍基于原图 |
| `IDPS_MAX_FILE_SIZE_MB` | `50` | 上传文件大小限制（MB） |
| `IDPS_UPLOAD_DIR` | `./uploads` | 上传目录路径 |

```bash
# 示例：使用英文模式，限制文件大小
IDPS_OCR_LANG=en IDPS_MAX_FILE_SIZE_MB=20 uvicorn app.main:app --port 8000
```

---

## API 文档

所有接口基础路径：`/api/v1/ocr`

### POST `/api/v1/ocr/single`

上传单个图片或 PDF 文件进行 OCR 识别。

**请求：** `multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `file` | File | 是 | 图片或 PDF 文件 |

**支持格式：** JPG, JPEG, PNG, BMP, TIFF, PDF

**响应示例：**

```json
{
  "pages": [
    {
      "page_index": 0,
      "text_blocks": [
        {
          "text": "Hello World",
          "confidence": 0.9629,
          "bbox": [[5, 52], [351, 52], [351, 83], [5, 83]]
        }
      ],
      "full_text": "Hello World"
    }
  ],
  "total_text": "Hello World",
  "page_count": 1
}
```

**curl 示例：**

```bash
curl -X POST -F "file=@photo.jpg" http://localhost:8000/api/v1/ocr/single
```

---

### POST `/api/v1/ocr/batch`

批量上传多个图片或 PDF 文件进行 OCR 识别。

**请求：** `multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `files` | File 数组 | 是 | 多个文件或 PDF 文件 |

**响应格式：** 同 `single` 接口，所有文件的页按顺序合并返回。

**curl 示例：**

```bash
curl -X POST -F "files=@page1.jpg" -F "files=@page2.pdf" http://localhost:8000/api/v1/ocr/batch
```

---

### GET `/health`

服务健康检查。

**响应：** `{"status": "ok"}`

---

## 响应字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `pages` | 数组 | 每页的 OCR 结果 |
| `pages[].page_index` | int | 页码（从 0 开始） |
| `pages[].text_blocks[]` | 数组 | 文本块列表 |
| `pages[].text_blocks[].text` | string | 识别文本 |
| `pages[].text_blocks[].confidence` | float | 置信度（0~1） |
| `pages[].text_blocks[].bbox` | array | 4 个角点坐标 `[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]` |
| `pages[].full_text` | string | 该页拼接后的全文本 |
| `total_text` | string | 所有页拼接后的全文本 |
| `page_count` | int | 总页数 |

---

## 常见问题

**Q: 上传大图时卡在 "exceeds max_side_limit of 4000" 不动？**

A: 这不是卡死，而是 PaddleOCR 默认的 medium 模型在 CPU 上对约 4000×3000 的图像做检测推理需要数分钟，期间无任何日志输出。本服务已默认改用 small 模型（`PP-OCRv6_small_det/rec`）并将检测输入长边限制在 2000px，大图单张约几秒即可完成。如需最高准确率可改回 medium 模型（见环境变量表），但请做好耗时预期。

**Q: 首次启动很慢？**

A: 模型文件会从国内节点自动下载（约 50MB），之后会缓存在本地，后续启动很快。

**Q: 支持 GPU 加速吗？**

A: 支持。安装 `paddlepaddle-gpu` 并设置 `IDPS_USE_GPU=true` 即可启用。

**Q: 如何提高识别准确率？**

A: 建议使用 200 DPI 以上的清晰图片。对于 PDF，默认会按 200 DPI 转换为高分辨率图片进行处理。