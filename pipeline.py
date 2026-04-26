"""
Main OCR Pipeline
Orchestrates preprocessing → OCR → extraction → structuring → summarization
"""

import os
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np

from src.preprocessor import ImagePreprocessor
from src.ocr_engine import OCREngine
from src.extractor import InformationExtractor
from src.summarizer import generate_summary, format_summary_text

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("pipeline")

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def process_single_image(
    image_path: str,
    preprocessor: ImagePreprocessor,
    ocr_engine: OCREngine,
    extractor: InformationExtractor,
    output_dir: str = "outputs/json_outputs",
    save_preprocessed: bool = False,
) -> Dict:
    """
    Full pipeline for one receipt image.
    Returns structured dict.
    """
    start_time = time.time()
    fname = Path(image_path).stem
    logger.info(f"Processing: {image_path}")

    result = {
        "file": Path(image_path).name,
        "status": "success",
        "processing_time_sec": 0,
    }

    try:
        # 1. Load & Preprocess
        raw_image = preprocessor.load_image(image_path)
        preprocessed = preprocessor.preprocess(raw_image)

        if save_preprocessed:
            pp_path = os.path.join(output_dir, f"{fname}_preprocessed.png")
            cv2.imwrite(pp_path, preprocessed)

        # 2. OCR
        ocr_results = ocr_engine.extract_text(preprocessed)
        avg_conf = ocr_engine.get_average_confidence(ocr_results)
        full_text = ocr_engine.get_full_text(ocr_results)
        logger.debug(f"  Detected {len(ocr_results)} text blocks, avg confidence: {avg_conf:.2f}")

        # 3. Extract structured fields
        receipt_data = extractor.extract(ocr_results, avg_ocr_confidence=avg_conf)
        receipt_data.raw_text = full_text

        # 4. Build output dict
        output = receipt_data.to_dict()
        output["file"] = Path(image_path).name
        output["ocr_engine"] = ocr_engine.engine_name
        output["avg_ocr_confidence"] = round(avg_conf, 4)
        output["raw_text_snippet"] = full_text[:300] + ("..." if len(full_text) > 300 else "")

        result.update(output)

        # 5. Save JSON
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f"{fname}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        logger.info(f"  → Saved: {json_path}")

    except Exception as e:
        logger.error(f"  Failed to process {image_path}: {e}", exc_info=True)
        result["status"] = "error"
        result["error"] = str(e)

    result["processing_time_sec"] = round(time.time() - start_time, 3)
    return result


def run_pipeline(
    input_dir: str,
    output_dir: str = "outputs",
    ocr_engine_name: str = "easyocr",
    save_preprocessed: bool = False,
):
    """
    Full batch pipeline: process all images in input_dir.
    """
    logger.info(f"Starting pipeline | Input: {input_dir} | Output: {output_dir}")

    # Init components
    preprocessor = ImagePreprocessor()
    ocr_engine = OCREngine(engine=ocr_engine_name)
    extractor = InformationExtractor()

    # Find image files
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if Path(fname).suffix.lower() in SUPPORTED_EXTENSIONS:
                image_paths.append(os.path.join(root, fname))

    if not image_paths:
        logger.warning(f"No images found in {input_dir}")
        return

    logger.info(f"Found {len(image_paths)} receipt image(s)")

    json_output_dir = os.path.join(output_dir, "json_outputs")
    all_results = []

    for path in sorted(image_paths):
        res = process_single_image(
            image_path=path,
            preprocessor=preprocessor,
            ocr_engine=ocr_engine,
            extractor=extractor,
            output_dir=json_output_dir,
            save_preprocessed=save_preprocessed,
        )
        all_results.append(res)

    # Generate financial summary
    successful = [r for r in all_results if r.get("status") == "success"]
    summary = generate_summary(successful)
    summary_text = format_summary_text(summary)

    print("\n" + summary_text)

    # Save summary JSON
    summary_path = os.path.join(output_dir, "expense_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Expense summary saved: {summary_path}")

    # Save summary text
    summary_txt_path = os.path.join(output_dir, "expense_summary.txt")
    with open(summary_txt_path, "w") as f:
        f.write(summary_text)

    # Save pipeline run report
    report = {
        "total_images": len(image_paths),
        "successful": len(successful),
        "failed": len(all_results) - len(successful),
        "results": all_results,
    }
    report_path = os.path.join(output_dir, "pipeline_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Pipeline complete. {len(successful)}/{len(image_paths)} processed successfully.")
    return summary


def main():
    parser = argparse.ArgumentParser(description="AI OCR Receipt Extractor Pipeline")
    parser.add_argument("--input", "-i", default="sample_images",
                        help="Directory of receipt images")
    parser.add_argument("--output", "-o", default="outputs",
                        help="Output directory for JSON and summaries")
    parser.add_argument("--engine", "-e", default="easyocr",
                        choices=["easyocr", "tesseract"],
                        help="OCR engine to use")
    parser.add_argument("--save-preprocessed", action="store_true",
                        help="Save preprocessed images alongside JSON outputs")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_pipeline(
        input_dir=args.input,
        output_dir=args.output,
        ocr_engine_name=args.engine,
        save_preprocessed=args.save_preprocessed,
    )


if __name__ == "__main__":
    main()
