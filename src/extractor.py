"""
Information Extractor Module
Extracts structured fields from OCR text with confidence scoring.
"""

import re
import logging
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field
from src.ocr_engine import OCRResult

logger = logging.getLogger(__name__)


@dataclass
class FieldResult:
    """Holds a field value with its confidence score."""
    value: Optional[str]
    confidence: float
    low_confidence: bool = False

    def to_dict(self):
        result = {"value": self.value, "confidence": round(self.confidence, 4)}
        if self.low_confidence:
            result["warning"] = "low_confidence"
        return result


@dataclass
class ItemResult:
    name: str
    price: Optional[str]
    confidence: float

    def to_dict(self):
        return {"name": self.name, "price": self.price, "confidence": round(self.confidence, 4)}


@dataclass
class ReceiptData:
    store_name: FieldResult = None
    date: FieldResult = None
    items: List[ItemResult] = field(default_factory=list)
    total_amount: FieldResult = None
    overall_confidence: float = 0.0
    raw_text: str = ""
    flagged_fields: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "store_name": self.store_name.to_dict() if self.store_name else {"value": None, "confidence": 0.0},
            "date": self.date.to_dict() if self.date else {"value": None, "confidence": 0.0},
            "items": [item.to_dict() for item in self.items],
            "total_amount": self.total_amount.to_dict() if self.total_amount else {"value": None, "confidence": 0.0},
            "overall_confidence": round(self.overall_confidence, 4),
            "flagged_fields": self.flagged_fields,
        }


# ─── Patterns ────────────────────────────────────────────────────────────────

DATE_PATTERNS = [
    (r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b', 0.95),
    (r'\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b', 0.95),
    (r'\b(\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{2,4})\b', 0.90),
    (r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{1,2},?\s\d{4})\b', 0.90),
    (r'\b(\d{8})\b', 0.70),  # YYYYMMDD or DDMMYYYY
]

CURRENCY_PATTERNS = [
    r'[\$£€₹]\s*[\d,]+\.?\d{0,2}',
    r'\d{1,5}[.,]\d{2}\s*(?:USD|EUR|GBP|INR)?',
    r'(?:Rs\.?|INR)\s*[\d,]+\.?\d{0,2}',
]

TOTAL_KEYWORDS = [
    "total", "grand total", "amount due", "balance due", "total amount",
    "net amount", "subtotal", "net total", "total payable", "due",
    "amount payable", "to pay", "bill amount"
]

STORE_NAME_EXCLUSIONS = re.compile(
    r'(?i)(receipt|invoice|date|time|tel|phone|address|www\.|http|thank you|'
    r'page \d|vat|gst|tax|reg\.|no\.|#|\d{3,})'
)

ITEM_PRICE_PATTERN = re.compile(
    r'^(.+?)\s+([\$£€₹]?\s*\d{1,5}[.,]\d{2})\s*$'
)


class InformationExtractor:
    """Extracts structured receipt fields from OCR results."""

    CONFIDENCE_THRESHOLD = 0.70

    def extract(self, ocr_results: List[OCRResult], avg_ocr_confidence: float = 1.0) -> ReceiptData:
        """
        Main extraction pipeline.
        """
        lines = self._group_into_lines(ocr_results)
        full_text = "\n".join(lines)

        receipt = ReceiptData(raw_text=full_text)
        receipt.store_name = self._extract_store_name(lines, avg_ocr_confidence)
        receipt.date = self._extract_date(lines, full_text, avg_ocr_confidence)
        receipt.total_amount = self._extract_total(lines, avg_ocr_confidence)
        receipt.items = self._extract_items(lines, receipt.total_amount, avg_ocr_confidence)

        # Compute overall confidence
        fields = [receipt.store_name, receipt.date, receipt.total_amount]
        valid = [f for f in fields if f and f.value]
        if valid:
            receipt.overall_confidence = sum(f.confidence for f in valid) / len(valid)
        else:
            receipt.overall_confidence = avg_ocr_confidence * 0.5

        # Flag low-confidence fields
        for fname, fobj in [("store_name", receipt.store_name), ("date", receipt.date),
                             ("total_amount", receipt.total_amount)]:
            if fobj and fobj.confidence < self.CONFIDENCE_THRESHOLD:
                fobj.low_confidence = True
                receipt.flagged_fields.append(fname)

        return receipt

    # ─── Field Extractors ─────────────────────────────────────────────────

    def _extract_store_name(self, lines: List[str], ocr_conf: float) -> FieldResult:
        """
        Heuristic: store name is typically in the first few lines,
        all-caps or title-cased, not a date/number line.
        """
        candidates = []
        for line in lines[:6]:
            clean = line.strip()
            if len(clean) < 3 or len(clean) > 60:
                continue
            if STORE_NAME_EXCLUSIONS.search(clean):
                continue
            if re.match(r'^\d+$', clean):
                continue
            # Prefer ALL CAPS or Title Case (common for store names)
            case_boost = 0.1 if (clean.isupper() or clean.istitle()) else 0.0
            score = (0.7 + case_boost) * ocr_conf
            candidates.append((clean, score))

        if candidates:
            best = max(candidates, key=lambda x: x[1])
            return FieldResult(value=best[0], confidence=round(min(best[1], 0.98), 4))
        return FieldResult(value=None, confidence=0.0)

    def _extract_date(self, lines: List[str], full_text: str, ocr_conf: float) -> FieldResult:
        """Extract date using regex patterns with pattern-based confidence."""
        for pattern, base_conf in DATE_PATTERNS:
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                confidence = base_conf * ocr_conf
                return FieldResult(value=match.group(1), confidence=round(confidence, 4))
        return FieldResult(value=None, confidence=0.0)

    def _extract_total(self, lines: List[str], ocr_conf: float) -> FieldResult:
        """
        Look for total amount using keyword matching.
        Scans from bottom of receipt upward (totals usually appear near end).
        """
        for line in reversed(lines):
            lower = line.lower()
            for keyword in TOTAL_KEYWORDS:
                if keyword in lower:
                    amount = self._find_currency_in_line(line)
                    if amount:
                        keyword_boost = 0.15 if keyword in ("total", "grand total", "amount due") else 0.05
                        conf = min((0.75 + keyword_boost) * ocr_conf, 0.99)
                        return FieldResult(value=amount, confidence=round(conf, 4))

        # Fallback: last currency value in the receipt
        for line in reversed(lines):
            amount = self._find_currency_in_line(line)
            if amount:
                return FieldResult(value=amount, confidence=round(0.45 * ocr_conf, 4))

        return FieldResult(value=None, confidence=0.0)

    def _extract_items(self, lines: List[str], total_field: FieldResult,
                       ocr_conf: float) -> List[ItemResult]:
        """
        Extract line items: lines matching <description> <price> pattern.
        Excludes the total line.
        """
        items = []
        total_val = total_field.value if total_field else None

        for line in lines:
            # Skip lines that are just the total
            if total_val and total_val in line:
                continue
            lower = line.lower()
            if any(k in lower for k in TOTAL_KEYWORDS[:5]):
                continue

            match = ITEM_PRICE_PATTERN.match(line.strip())
            if match:
                name = match.group(1).strip()
                price = match.group(2).strip()
                if len(name) > 1:
                    conf = round(0.75 * ocr_conf, 4)
                    items.append(ItemResult(name=name, price=price, confidence=conf))

        return items

    # ─── Helpers ──────────────────────────────────────────────────────────

    def _find_currency_in_line(self, line: str) -> Optional[str]:
        for pattern in CURRENCY_PATTERNS:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(0).strip()
        return None

    def _group_into_lines(self, ocr_results: List[OCRResult]) -> List[str]:
        """
        Group OCR word-level results into lines based on Y-coordinate proximity.
        """
        if not ocr_results:
            return []

        # Sort by Y then X
        sorted_results = sorted(
            ocr_results,
            key=lambda r: (r.bbox[0][1] if r.bbox else 0, r.bbox[0][0] if r.bbox else 0)
        )

        lines = []
        current_line = []
        current_y = sorted_results[0].bbox[0][1] if sorted_results[0].bbox else 0
        LINE_THRESHOLD = 12  # pixels

        for result in sorted_results:
            y = result.bbox[0][1] if result.bbox else current_y
            if abs(y - current_y) > LINE_THRESHOLD:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [result.text]
                current_y = y
            else:
                current_line.append(result.text)

        if current_line:
            lines.append(" ".join(current_line))

        return [l for l in lines if l.strip()]
