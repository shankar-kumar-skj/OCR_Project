"""
Unit Tests for AI OCR Receipt Extraction Pipeline
"""

import sys
import os
import json
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extractor import InformationExtractor, FieldResult, ItemResult, ReceiptData
from src.ocr_engine import OCRResult
from src.summarizer import generate_summary, _parse_amount, format_summary_text
from src.preprocessor import ImagePreprocessor


# ─── Helper ───────────────────────────────────────────────────────────────────

def make_ocr_result(text: str, confidence: float = 0.95, y: int = 0, x: int = 0) -> OCRResult:
    bbox = [[x, y], [x + 100, y], [x + 100, y + 12], [x, y + 12]]
    return OCRResult(text=text, confidence=confidence, bbox=bbox)


# ─── Preprocessor Tests ───────────────────────────────────────────────────────

class TestPreprocessor(unittest.TestCase):

    def setUp(self):
        self.preprocessor = ImagePreprocessor()

    def test_grayscale_conversion_rgb(self):
        rgb = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = self.preprocessor._to_grayscale(rgb)
        self.assertEqual(len(result.shape), 2)

    def test_grayscale_passthrough(self):
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = self.preprocessor._to_grayscale(gray)
        self.assertEqual(result.shape, gray.shape)

    def test_denoise_output_shape(self):
        gray = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        result = self.preprocessor._denoise(gray)
        self.assertEqual(result.shape, gray.shape)

    def test_enhance_contrast_output_range(self):
        gray = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        result = self.preprocessor._enhance_contrast(gray)
        self.assertLessEqual(result.max(), 255)
        self.assertGreaterEqual(result.min(), 0)

    def test_binarize_output_binary(self):
        gray = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        result = self.preprocessor._binarize(gray)
        unique_vals = set(result.flatten().tolist())
        self.assertTrue(unique_vals.issubset({0, 255}))

    def test_full_pipeline_shape_preserved(self):
        image = np.random.randint(0, 255, (300, 200, 3), dtype=np.uint8)
        result = self.preprocessor.preprocess(image)
        self.assertEqual(result.shape[:2], (300, 200))

    def test_deskew_no_crash_uniform_image(self):
        uniform = np.ones((200, 200), dtype=np.uint8) * 128
        result = self.preprocessor._deskew(uniform)
        self.assertEqual(result.shape, uniform.shape)


# ─── OCR Engine Tests ─────────────────────────────────────────────────────────

class TestOCRResult(unittest.TestCase):

    def test_ocr_result_creation(self):
        r = OCRResult(text="  Hello  ", confidence=0.9)
        self.assertEqual(r.text, "Hello")
        self.assertEqual(r.confidence, 0.9)

    def test_ocr_result_repr(self):
        r = OCRResult("Test", 0.85)
        self.assertIn("Test", repr(r))
        self.assertIn("0.85", repr(r))


# ─── Extractor Tests ──────────────────────────────────────────────────────────

class TestInformationExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = InformationExtractor()

    def _make_receipt_ocr(self, lines_with_y):
        """Helper: list of (text, y_position)"""
        return [make_ocr_result(text, y=y) for text, y in lines_with_y]

    def test_extract_store_name(self):
        ocr = self._make_receipt_ocr([
            ("WHOLE FOODS MARKET", 0),
            ("123 Main Street", 15),
            ("Date: 12/01/2024", 30),
        ])
        receipt = self.extractor.extract(ocr, avg_ocr_confidence=0.92)
        self.assertIsNotNone(receipt.store_name.value)
        self.assertIn("WHOLE FOODS", receipt.store_name.value)

    def test_extract_date_slash_format(self):
        ocr = self._make_receipt_ocr([
            ("STORE NAME", 0),
            ("Date: 15/03/2024", 15),
        ])
        receipt = self.extractor.extract(ocr, avg_ocr_confidence=0.90)
        self.assertIsNotNone(receipt.date.value)
        self.assertIn("15", receipt.date.value)

    def test_extract_date_dash_format(self):
        ocr = self._make_receipt_ocr([
            ("STORE", 0),
            ("2024-03-15", 15),
        ])
        receipt = self.extractor.extract(ocr, avg_ocr_confidence=0.88)
        self.assertIsNotNone(receipt.date.value)

    def test_extract_total_keyword(self):
        ocr = self._make_receipt_ocr([
            ("STORE", 0),
            ("Apple  $2.50", 15),
            ("Milk   $1.99", 30),
            ("Total  $4.49", 45),
        ])
        receipt = self.extractor.extract(ocr, avg_ocr_confidence=0.95)
        self.assertIsNotNone(receipt.total_amount.value)
        self.assertIn("4.49", receipt.total_amount.value)

    def test_extract_items(self):
        ocr = self._make_receipt_ocr([
            ("STORE", 0),
            ("Apple     $2.50", 15),
            ("Bread     $3.00", 30),
            ("Total     $5.50", 45),
        ])
        receipt = self.extractor.extract(ocr, avg_ocr_confidence=0.90)
        self.assertGreater(len(receipt.items), 0)

    def test_missing_date_returns_none(self):
        ocr = self._make_receipt_ocr([("STORE NAME", 0), ("No date here", 15)])
        receipt = self.extractor.extract(ocr, avg_ocr_confidence=0.80)
        self.assertIsNone(receipt.date.value)

    def test_low_confidence_flagging(self):
        ocr = self._make_receipt_ocr([("X", 0)])
        receipt = self.extractor.extract(ocr, avg_ocr_confidence=0.50)
        # Low confidence fields should be flagged
        for fname in receipt.flagged_fields:
            field = getattr(receipt, fname, None)
            if field:
                self.assertTrue(field.low_confidence)

    def test_overall_confidence_range(self):
        ocr = self._make_receipt_ocr([
            ("BEST BUY", 0),
            ("Date: 01/01/2024", 15),
            ("Item A  $10.00", 30),
            ("Total   $10.00", 45),
        ])
        receipt = self.extractor.extract(ocr, avg_ocr_confidence=0.90)
        self.assertGreaterEqual(receipt.overall_confidence, 0.0)
        self.assertLessEqual(receipt.overall_confidence, 1.0)

    def test_to_dict_structure(self):
        ocr = self._make_receipt_ocr([
            ("SHOP", 0),
            ("Date 12/12/2023", 15),
            ("Apple  $1.00", 30),
            ("Total  $1.00", 45),
        ])
        receipt = self.extractor.extract(ocr, 0.9)
        d = receipt.to_dict()
        self.assertIn("store_name", d)
        self.assertIn("date", d)
        self.assertIn("items", d)
        self.assertIn("total_amount", d)
        self.assertIn("overall_confidence", d)
        self.assertIn("flagged_fields", d)
        # Each field must have value and confidence
        for key in ("store_name", "date", "total_amount"):
            self.assertIn("value", d[key])
            self.assertIn("confidence", d[key])


# ─── Summarizer Tests ─────────────────────────────────────────────────────────

class TestSummarizer(unittest.TestCase):

    def _make_receipt(self, store, date, total, conf=0.90, file="receipt.jpg"):
        return {
            "file": file,
            "store_name": {"value": store, "confidence": conf},
            "date": {"value": date, "confidence": conf},
            "total_amount": {"value": total, "confidence": conf},
            "overall_confidence": conf,
        }

    def test_parse_amount_dollar(self):
        self.assertAlmostEqual(_parse_amount("$12.50"), 12.50)

    def test_parse_amount_comma_sep(self):
        self.assertAlmostEqual(_parse_amount("1,234.56"), 1234.56)

    def test_parse_amount_inr(self):
        self.assertAlmostEqual(_parse_amount("Rs. 500"), 500.0)

    def test_parse_amount_none(self):
        self.assertIsNone(_parse_amount(None))
        self.assertIsNone(_parse_amount("N/A"))

    def test_summary_total_spend(self):
        receipts = [
            self._make_receipt("Store A", "2024-01-01", "$10.00", file="r1.jpg"),
            self._make_receipt("Store B", "2024-01-02", "$20.00", file="r2.jpg"),
            self._make_receipt("Store A", "2024-01-03", "$15.00", file="r3.jpg"),
        ]
        summary = generate_summary(receipts)
        self.assertAlmostEqual(summary["summary"]["total_spend"], 45.00)

    def test_summary_transaction_count(self):
        receipts = [
            self._make_receipt("X", "2024-01-01", "$5.00", file="a.jpg"),
            self._make_receipt("Y", "2024-01-02", "$5.00", file="b.jpg"),
        ]
        summary = generate_summary(receipts)
        self.assertEqual(summary["summary"]["number_of_transactions"], 2)

    def test_summary_spend_per_store(self):
        receipts = [
            self._make_receipt("TESCO", "2024-01-01", "$30.00", file="r1.jpg"),
            self._make_receipt("TESCO", "2024-01-02", "$20.00", file="r2.jpg"),
            self._make_receipt("ASDA", "2024-01-03", "$10.00", file="r3.jpg"),
        ]
        summary = generate_summary(receipts)
        stores = {s["store"]: s["total_spent"] for s in summary["spend_per_store"]}
        self.assertAlmostEqual(stores["TESCO"], 50.00)
        self.assertAlmostEqual(stores["ASDA"], 10.00)

    def test_summary_unresolved(self):
        receipts = [
            self._make_receipt("STORE", "2024-01-01", None, file="r1.jpg"),
        ]
        summary = generate_summary(receipts)
        self.assertEqual(summary["receipts_unresolved"], 1)
        self.assertEqual(summary["summary"]["number_of_transactions"], 0)

    def test_summary_low_confidence_flagged(self):
        receipts = [
            self._make_receipt("STORE", "2024-01-01", "$10.00", conf=0.50, file="low.jpg"),
        ]
        summary = generate_summary(receipts)
        self.assertEqual(len(summary["low_confidence_receipts"]), 1)

    def test_format_summary_text(self):
        receipts = [
            self._make_receipt("BestShop", "2024-01-01", "$99.99", file="r.jpg"),
        ]
        summary = generate_summary(receipts)
        text = format_summary_text(summary)
        self.assertIn("EXPENSE SUMMARY", text)
        self.assertIn("99.99", text)
        self.assertIn("BestShop", text)

    def test_empty_receipts(self):
        summary = generate_summary([])
        self.assertEqual(summary["summary"]["total_spend"], 0)
        self.assertEqual(summary["summary"]["number_of_transactions"], 0)


# ─── Field Result Tests ───────────────────────────────────────────────────────

class TestFieldResult(unittest.TestCase):

    def test_to_dict_normal(self):
        fr = FieldResult(value="WALMART", confidence=0.93)
        d = fr.to_dict()
        self.assertEqual(d["value"], "WALMART")
        self.assertAlmostEqual(d["confidence"], 0.93)
        self.assertNotIn("warning", d)

    def test_to_dict_low_confidence(self):
        fr = FieldResult(value="??", confidence=0.55, low_confidence=True)
        d = fr.to_dict()
        self.assertIn("warning", d)
        self.assertEqual(d["warning"], "low_confidence")


if __name__ == "__main__":
    unittest.main(verbosity=2)
