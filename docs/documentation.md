# AI-OCR Receipt Extraction Pipeline
## Technical Documentation

**Carbon Crunch Shortlisting Assignment**
**Version:** 1.0 | **Language:** Python 3.9+

---

## 1. Approach

### Overview

The system implements a modular, four-stage pipeline designed to extract structured data from real-world receipt images and produce confidence-aware JSON outputs.

```
[Receipt Image]
      │
      ▼
[1. Preprocessing]     ← Denoise, deskew, enhance, binarize
      │
      ▼
[2. OCR]               ← EasyOCR (primary) / Tesseract (fallback)
      │
      ▼
[3. Extraction]        ← Regex + keyword heuristics per field
      │
      ▼
[4. Structuring]       ← Typed JSON with confidence scores
      │
      ▼
[5. Summarization]     ← Expense aggregation across receipts
```

### Stage 1 — Image Preprocessing (`src/preprocessor.py`)

Each receipt image passes through a sequential preprocessing pipeline:

- **Grayscale conversion** — Reduces dimensionality and simplifies processing.
- **Gaussian blur + Non-local means denoising** — Removes sensor and compression noise while preserving edges.
- **Skew correction** — Uses Hough Line Transform to detect dominant text angles, then rotates the image to align text horizontally (only applied if skew > 0.5°).
- **CLAHE contrast enhancement** — Adaptive histogram equalization handles receipts with uneven lighting (e.g., shadows from folding or poor illumination).
- **Adaptive binarization** — Combines Otsu's global threshold with Gaussian adaptive threshold using bitwise AND to handle receipts with gradient backgrounds.

### Stage 2 — OCR (`src/ocr_engine.py`)

**Primary engine: EasyOCR** — Used for its strong performance on varied fonts, rotated text, and non-ideal conditions. Returns per-detection confidence scores natively.

**Fallback: Tesseract** — Used if EasyOCR is unavailable. Provides word-level confidence via `image_to_data()` API.

Results are sorted spatially (top-to-bottom, left-to-right) and grouped into lines using Y-coordinate proximity clustering (12px threshold).

### Stage 3 — Key Information Extraction (`src/extractor.py`)

Each field uses a layered confidence model:

| Field | Strategy | Confidence Basis |
|-------|----------|-----------------|
| `store_name` | First non-numeric, non-date line in top 6 | OCR conf × case boost (ALL_CAPS/Title) |
| `date` | Multi-pattern regex (5 formats) | Pattern match strength × OCR conf |
| `total_amount` | Keyword scan (bottom-up) + currency regex | Keyword strength × OCR conf |
| `items` | `<name>   <price>` line pattern matching | 0.75 × OCR conf |

Fields with confidence < 0.70 are flagged in `flagged_fields`.

### Stage 4 — Financial Summarization (`src/summarizer.py`)

Aggregates across all receipts to produce:
- Total spend, transaction count, min/max/average values
- Spend breakdown per store
- Flags for low-confidence entries needing review
- List of unresolved receipts (no parseable total)

---

## 2. Tools Used

| Tool | Version | Purpose |
|------|---------|---------|
| EasyOCR | ≥1.7.0 | Primary OCR with confidence scores |
| Tesseract (pytesseract) | ≥0.3.10 | Fallback OCR engine |
| OpenCV (cv2) | ≥4.8.0 | Preprocessing, deskew, denoising |
| Pillow | ≥10.0.0 | Image I/O and format handling |
| NumPy | ≥1.24.0 | Array operations |
| Python stdlib | 3.9+ | re, json, pathlib, argparse, logging |

---

## 3. Output Formats

### Per-Receipt JSON (with confidence)

```json
{
  "file": "receipt_001.jpg",
  "store_name": { "value": "WHOLE FOODS", "confidence": 0.93 },
  "date": { "value": "15/03/2024", "confidence": 0.87 },
  "items": [
    { "name": "Organic Milk", "price": "$3.99", "confidence": 0.87 }
  ],
  "total_amount": { "value": "$16.77", "confidence": 0.95 },
  "overall_confidence": 0.92,
  "flagged_fields": [],
  "avg_ocr_confidence": 0.91,
  "processing_time_sec": 2.34
}
```

### Expense Summary JSON

```json
{
  "summary": {
    "total_spend": 142.63,
    "number_of_transactions": 5,
    "average_transaction_value": 28.53
  },
  "spend_per_store": [...],
  "low_confidence_receipts": [...],
  "unresolved_receipts": [...]
}
```

---

## 4. Challenges Faced

### 4.1 Skew Detection Sensitivity
Receipts with dominant vertical lines (borders, price columns) caused false skew angles. **Solution:** filtered Hough lines to only those within ±45° of horizontal, and added a minimum skew threshold (0.5°) to avoid over-rotation.

### 4.2 Store Name Ambiguity
Receipt headers often include phone numbers, URLs, and promotional text alongside the store name. **Solution:** combined regex exclusion patterns (phone, URL, tax keywords) with a line-length filter and case-preference heuristic.

### 4.3 Currency Format Variation
Receipts use `$`, `£`, `€`, `Rs.`, `INR`, and comma-vs-dot formatting. **Solution:** a multi-regex parser in `summarizer._parse_amount()` strips all non-numeric characters and handles thousand separators.

### 4.4 OCR Line Grouping
EasyOCR returns word/phrase-level bounding boxes, not lines. **Solution:** grouped results by Y-coordinate proximity (12px tolerance), then sorted by X within each line to reconstruct reading order.

### 4.5 Tesseract vs EasyOCR Tradeoffs
Tesseract is faster and lighter but struggles with non-standard fonts common in receipts. EasyOCR handles these better but requires more memory. **Solution:** EasyOCR as primary with automatic fallback.

---

## 5. Potential Improvements

### 5.1 Fine-Tuning
Fine-tune a CRNN or TrOCR model on a domain-specific receipt dataset (e.g., CORD, SROIE) for improved field extraction accuracy.

### 5.2 Layout-Aware Parsing
Use LayoutLM or Donut (document understanding transformers) to jointly model text and spatial layout, eliminating fragile regex-based heuristics.

### 5.3 Currency Normalization
Add a currency detection step to normalize all amounts to a single currency for cross-receipt comparison.

### 5.4 Confidence Calibration
Train a logistic regression calibrator on ground-truth annotations to map raw OCR confidence to true accuracy probabilities.

### 5.5 Multi-Language Support
Extend EasyOCR language models for multilingual receipts (Hindi, Arabic, Chinese common in international datasets).

### 5.6 Feedback Loop
Add a human-review queue: low-confidence receipts are flagged and corrections fed back to improve future extractions.

---

## 6. Project Structure

```
ai_ocr_receipt/
├── pipeline.py              # Main entry point
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
│   ├── preprocessor.py      # Image preprocessing
│   ├── ocr_engine.py        # OCR wrapper (EasyOCR / Tesseract)
│   ├── extractor.py         # Field extraction + confidence scoring
│   └── summarizer.py        # Financial summary generation
├── tests/
│   └── test_pipeline.py     # Unit tests (35 tests)
├── sample_images/           # Place receipt images here
└── outputs/
    ├── json_outputs/        # Per-receipt JSON files
    ├── expense_summary.json
    ├── expense_summary.txt
    └── pipeline_report.json
```

---

## 7. Running the Pipeline

### Setup

```bash
git clone <repo_url>
cd ai_ocr_receipt
pip install -r requirements.txt
```

For Tesseract (if using as engine):
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

### Run

```bash
# Default (EasyOCR)
python pipeline.py --input sample_images --output outputs

# With Tesseract fallback
python pipeline.py --input sample_images --engine tesseract

# Save preprocessed images for inspection
python pipeline.py --input sample_images --save-preprocessed --verbose
```

### Run Tests

```bash
python -m pytest tests/ -v
```

---

## 8. Evaluation Alignment

| Criterion | Implementation |
|-----------|----------------|
| Extraction Accuracy (30%) | Multi-strategy extraction with pattern + keyword heuristics |
| Robustness (15%) | CLAHE + NL-means + adaptive binarization handles real-world noise |
| Data Structuring (10%) | Typed JSON with confidence metadata per field |
| Financial Summary (10%) | Per-store aggregation, min/max/avg, unresolved tracking |
| Confidence Scoring (20%) | OCR + pattern validation + keyword heuristics combined |
| Code Quality (10%) | Modular src/, typed dataclasses, 35 unit tests |
| Edge Case Handling (5%) | Missing fields return None, errors caught per-image |
