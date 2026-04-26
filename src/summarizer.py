"""
Financial Summary Module
Aggregates structured receipt data into an expense summary report.
"""

import re
import json
import logging
from typing import List, Dict, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


def _parse_amount(value: str) -> Optional[float]:
    """
    Robustly parse a currency string to a float.
    Handles: $12.50, Rs. 1,234.56, 1234.56, etc.
    """
    if not value:
        return None
    # Remove currency symbols, letters, whitespace — keep digits, dot, comma
    cleaned = re.sub(r'[^\d.,]', '', value)
    # If multiple dots, keep only last (e.g. "Rs.500" → ".500" → "500")
    parts = cleaned.split('.')
    if len(parts) > 2:
        cleaned = ''.join(parts[:-1]) + '.' + parts[-1]
    # Remove lone leading dot
    cleaned = cleaned.lstrip('.')
    # Handle comma as thousand separator
    cleaned = cleaned.replace(',', '')
    try:
        return float(cleaned)
    except ValueError:
        return None


def generate_summary(receipts: List[Dict]) -> Dict:
    """
    Generate a financial summary from a list of parsed receipt dicts.

    Args:
        receipts: list of dicts with keys: file, store_name, date, items, total_amount, overall_confidence

    Returns:
        summary dict with totals, per-store breakdown, and stats
    """
    total_spend = 0.0
    num_transactions = 0
    spend_per_store = defaultdict(float)
    store_transaction_count = defaultdict(int)
    low_confidence_receipts = []
    unresolved_receipts = []
    all_amounts = []

    for receipt in receipts:
        fname = receipt.get("file", "unknown")
        total_field = receipt.get("total_amount", {})
        total_val = total_field.get("value") if isinstance(total_field, dict) else None
        total_conf = total_field.get("confidence", 0) if isinstance(total_field, dict) else 0

        store_field = receipt.get("store_name", {})
        store_name = store_field.get("value") if isinstance(store_field, dict) else None
        store_name = store_name or "Unknown Store"

        overall_conf = receipt.get("overall_confidence", 0)

        amount = _parse_amount(total_val)

        if amount is not None:
            total_spend += amount
            num_transactions += 1
            spend_per_store[store_name] += amount
            store_transaction_count[store_name] += 1
            all_amounts.append(amount)

            if total_conf < 0.7 or overall_conf < 0.6:
                low_confidence_receipts.append({
                    "file": fname,
                    "total_amount": total_val,
                    "confidence": total_conf,
                    "note": "Review recommended"
                })
        else:
            unresolved_receipts.append({
                "file": fname,
                "reason": "Could not parse total amount",
                "raw_value": total_val
            })

    # Build per-store summary
    store_summary = []
    for store, amount in sorted(spend_per_store.items(), key=lambda x: -x[1]):
        store_summary.append({
            "store": store,
            "total_spent": round(amount, 2),
            "transactions": store_transaction_count[store],
            "average_transaction": round(amount / store_transaction_count[store], 2)
        })

    summary = {
        "summary": {
            "total_spend": round(total_spend, 2),
            "number_of_transactions": num_transactions,
            "average_transaction_value": round(total_spend / num_transactions, 2) if num_transactions else 0,
            "max_transaction": round(max(all_amounts), 2) if all_amounts else 0,
            "min_transaction": round(min(all_amounts), 2) if all_amounts else 0,
        },
        "spend_per_store": store_summary,
        "low_confidence_receipts": low_confidence_receipts,
        "unresolved_receipts": unresolved_receipts,
        "receipts_processed": len(receipts),
        "receipts_with_totals": num_transactions,
        "receipts_unresolved": len(unresolved_receipts),
    }

    return summary


def format_summary_text(summary: Dict) -> str:
    """Render a human-readable text version of the summary."""
    s = summary["summary"]
    lines = [
        "=" * 50,
        "       EXPENSE SUMMARY REPORT",
        "=" * 50,
        f"  Receipts Processed  : {summary['receipts_processed']}",
        f"  With Total Amount   : {summary['receipts_with_totals']}",
        f"  Unresolved          : {summary['receipts_unresolved']}",
        "-" * 50,
        f"  Total Spend         : {s['total_spend']}",
        f"  Transactions        : {s['number_of_transactions']}",
        f"  Avg Transaction     : {s['average_transaction_value']}",
        f"  Max Transaction     : {s['max_transaction']}",
        f"  Min Transaction     : {s['min_transaction']}",
        "=" * 50,
        "  SPEND PER STORE",
        "-" * 50,
    ]
    for store_entry in summary["spend_per_store"]:
        lines.append(
            f"  {store_entry['store'][:28]:<28} "
            f"{store_entry['total_spent']:>8.2f}  "
            f"({store_entry['transactions']} txn)"
        )
    if summary["low_confidence_receipts"]:
        lines += ["-" * 50, "  ⚠  LOW CONFIDENCE FLAGS"]
        for item in summary["low_confidence_receipts"]:
            lines.append(f"     {item['file']} — conf: {item['confidence']:.2f} → {item['note']}")
    if summary["unresolved_receipts"]:
        lines += ["-" * 50, "  ✗  UNRESOLVED RECEIPTS"]
        for item in summary["unresolved_receipts"]:
            lines.append(f"     {item['file']} — {item['reason']}")
    lines.append("=" * 50)
    return "\n".join(lines)
