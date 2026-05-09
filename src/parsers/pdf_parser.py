import io
import re
from typing import Optional


def parse_pdf_transcript(pdf_bytes: bytes) -> Optional[dict]:
    """Extract text from PDF earnings transcript bytes."""
    try:
        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.layout import LAParams

        output = io.StringIO()
        laparams = LAParams(line_margin=0.5, word_margin=0.1)
        extract_text_to_fp(
            io.BytesIO(pdf_bytes),
            output,
            laparams=laparams,
            output_type="text",
            codec="utf-8",
        )
        text = output.getvalue()
    except Exception:
        return None

    if not text or len(text.split()) < 200:
        return None

    # Post-process: fix hyphenated line breaks, normalize whitespace
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # Treat as unstructured — delegate to html_parser normalizer-compatible dict
    return {
        "raw_text": text,
        "utterances": [],
        "speakers": [],
        "sections": {"prepared_remarks": [text], "qa": []},
        "word_count": len(text.split()),
    }
