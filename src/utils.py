from collections import Counter

import fitz


DEFAULT_LAYOUT_SAMPLE_PAGES = 4


def extract_pdf_text_blocks(page):
    page_rect = page.rect
    blocks = []

    for block in page.get_text("blocks", sort=True):
        x0, y0, x1, y1, text, _block_no, block_type = block
        text = (text or "").strip()

        if block_type != 0:
            continue
        if len(text) < 40:
            continue

        # Ignore likely headers / footers.
        if y1 < page_rect.height * 0.08 or y0 > page_rect.height * 0.93:
            continue

        width = x1 - x0
        if width <= 0:
            continue

        blocks.append(
            {
                "x0": x0,
                "x1": x1,
                "y0": y0,
                "y1": y1,
                "width_frac": width / page_rect.width,
                "center_frac": ((x0 + x1) / 2.0) / page_rect.width,
                "text": text,
            }
        )

    return blocks


def classify_pdf_page_layout(page):
    blocks = extract_pdf_text_blocks(page)
    if not blocks:
        return {
            "label": "unknown",
            "reason": "no usable text blocks",
            "block_count": 0,
            "wide_center_blocks": 0,
            "left_column_blocks": 0,
            "right_column_blocks": 0,
        }

    wide_center_blocks = [
        block
        for block in blocks
        if block["width_frac"] >= 0.55 and 0.38 <= block["center_frac"] <= 0.62
    ]
    left_column_blocks = [
        block
        for block in blocks
        if 0.22 <= block["width_frac"] <= 0.48 and 0.18 <= block["center_frac"] <= 0.42
    ]
    right_column_blocks = [
        block
        for block in blocks
        if 0.22 <= block["width_frac"] <= 0.48 and 0.58 <= block["center_frac"] <= 0.82
    ]

    narrow_column_count = len(left_column_blocks) + len(right_column_blocks)
    double_pairs = min(len(left_column_blocks), len(right_column_blocks))

    if double_pairs >= 2 and narrow_column_count / len(blocks) >= 0.6:
        label = "double"
        reason = "stable left/right text-block clusters with a center gutter"
    elif len(wide_center_blocks) >= max(2, int(len(blocks) * 0.5)):
        label = "single"
        reason = "most text blocks span a single centered body region"
    else:
        label = "mixed"
        reason = "page contains both wide and column-like blocks"

    return {
        "label": label,
        "reason": reason,
        "block_count": len(blocks),
        "wide_center_blocks": len(wide_center_blocks),
        "left_column_blocks": len(left_column_blocks),
        "right_column_blocks": len(right_column_blocks),
    }


def classify_pdf_layout(pdf_path, sample_pages=DEFAULT_LAYOUT_SAMPLE_PAGES):
    doc = fitz.open(pdf_path)
    page_results = []

    for page_index in range(min(sample_pages, doc.page_count)):
        page = doc.load_page(page_index)
        result = classify_pdf_page_layout(page)
        result["page_number"] = page_index + 1
        page_results.append(result)

    vote_counter = Counter(
        result["label"] for result in page_results if result["label"] in {"single", "double"}
    )

    if vote_counter["double"] > vote_counter["single"]:
        document_label = "double"
        reason = "majority of sampled pages are double-column"
    elif vote_counter["single"] > vote_counter["double"]:
        document_label = "single"
        reason = "majority of sampled pages are single-column"
    else:
        aggregate_wide = sum(result["wide_center_blocks"] for result in page_results)
        aggregate_narrow = sum(
            result["left_column_blocks"] + result["right_column_blocks"]
            for result in page_results
        )
        if aggregate_narrow > aggregate_wide:
            document_label = "double"
            reason = "tie broken by stronger aggregate column-pattern evidence"
        elif aggregate_wide > 0:
            document_label = "single"
            reason = "tie broken by stronger aggregate full-width body evidence"
        else:
            document_label = "unknown"
            reason = "not enough usable body-text evidence"

    return {
        "pdf_path": str(pdf_path),
        "label": document_label,
        "reason": reason,
        "page_results": page_results,
    }
