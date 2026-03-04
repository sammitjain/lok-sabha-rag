"""Compare EasyOCR vs Tesseract OCR on the 4 image-only PDFs.

Runs each PDF through Docling with:
  1. Default (no OCR) — baseline
  2. EasyOCR (force_full_page_ocr=True)
  3. Tesseract via tesserocr (force_full_page_ocr=True)

Outputs a side-by-side comparison: text length, time taken, and first 300 chars.

Usage:
    uv run python scripts/compare_ocr.py
"""

import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
    TesseractOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

# The 4 image-only PDFs from the build report
PDFS = [
    "data/18/pdfs/session_2/AS39_0OzSlM.pdf",
    "data/18/pdfs/session_2/AU374_vm4HTC.pdf",
    "data/18/pdfs/session_6/AU1701_cJQXhx.pdf",
    "data/18/pdfs/session_6/AU173_hWJ6ha.pdf",
]


def make_converter(ocr_options=None, do_ocr: bool = False) -> DocumentConverter:
    """Build a DocumentConverter with the given OCR config."""
    pipeline_opts = PdfPipelineOptions(do_ocr=do_ocr)
    if ocr_options:
        pipeline_opts.ocr_options = ocr_options
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)}
    )


def extract(converter: DocumentConverter, pdf_path: str) -> tuple[str, float]:
    """Run extraction, return (markdown_text, elapsed_seconds)."""
    t0 = time.time()
    result = converter.convert(pdf_path)
    md = result.document.export_to_markdown()
    elapsed = time.time() - t0
    return md, elapsed


def main():
    # ── Build converters ─────────────────────────────────────────────────────
    print("Initializing converters...")

    t0 = time.time()
    conv_default = make_converter(do_ocr=False)
    print(f"  Default (no OCR): ready ({time.time() - t0:.1f}s)")

    t0 = time.time()
    conv_easyocr = make_converter(
        ocr_options=EasyOcrOptions(lang=["en"], force_full_page_ocr=True),
        do_ocr=True,
    )
    print(f"  EasyOCR: ready ({time.time() - t0:.1f}s)")

    t0 = time.time()
    conv_tesseract = make_converter(
        ocr_options=TesseractOcrOptions(lang=["eng"], force_full_page_ocr=True),
        do_ocr=True,
    )
    print(f"  Tesseract: ready ({time.time() - t0:.1f}s)")

    # ── Run comparison ───────────────────────────────────────────────────────
    engines = [
        ("Default (no OCR)", conv_default),
        ("EasyOCR", conv_easyocr),
        ("Tesseract", conv_tesseract),
    ]

    for pdf_path in PDFS:
        fname = Path(pdf_path).name
        print(f"\n{'='*80}")
        print(f"PDF: {fname}")
        print(f"{'='*80}")

        for engine_name, converter in engines:
            try:
                text, elapsed = extract(converter, pdf_path)
                chars = len(text)
                words = len(text.split()) if text else 0
                preview = text[:300].replace("\n", " ") if text else "(empty)"

                print(f"\n  [{engine_name}]")
                print(f"    Time:  {elapsed:.2f}s")
                print(f"    Chars: {chars}  |  Words: {words}")
                print(f"    Preview: {preview}...")
            except Exception as e:
                print(f"\n  [{engine_name}]")
                print(f"    ERROR: {e}")

    print(f"\n{'='*80}")
    print("Done.")


if __name__ == "__main__":
    main()
