import re
from typing import Optional

from bs4 import BeautifulSoup

BOILERPLATE_PATTERNS = re.compile(
    r"(forward[- ]looking statement[s]?|safe harbor|certain statement[s]? in this|"
    r"please stand by|your lines have been placed|operator instructions|"
    r"\[operator\]|ladies and gentlemen|thank you for holding)",
    re.IGNORECASE,
)

SPEAKER_LINE_RE = re.compile(
    r"^([A-Z][A-Za-z\s\-\.\']{2,40}?)"   # Name
    r"\s*[-–:]\s*"                          # Separator
    r"([A-Za-z][\w\s\-\,\&\.]{0,80}?)?"   # Optional title
    r"\s*$",
)

SECTION_MARKERS = re.compile(
    r"(question[s]?[\s-]and[\s-]answer|Q[\s&]A session|operator question|"
    r"we will now begin|now open.{0,20}question|please limit|"
    r"we'll now take question)",
    re.IGNORECASE,
)


def parse_html_transcript(html: str) -> Optional[dict]:
    """Parse raw HTML transcript into structured dict."""
    soup = BeautifulSoup(html, "lxml")

    # Remove script, style, nav, header, footer noise
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()

    # Try to find the main content area
    main = (
        soup.find("div", class_=re.compile(r"transcript|content|article|document|body", re.I))
        or soup.find("article")
        or soup.find("main")
        or soup.body
        or soup
    )

    paragraphs = []
    current_speaker = {"name": "Unknown", "title": "", "role": "Unknown"}
    utterances = []  # [{speaker, text}]

    for elem in main.find_all(["p", "div", "span", "strong", "b"], recursive=True):
        text = elem.get_text(separator=" ", strip=True)
        if not text or len(text) < 10:
            continue

        # Check if this element is a speaker label
        m = SPEAKER_LINE_RE.match(text)
        if m and len(text) < 120:
            name = m.group(1).strip()
            title = (m.group(2) or "").strip()
            if _is_plausible_speaker_name(name):
                current_speaker = {"name": name, "title": title, "role": _classify_role(title)}
                continue

        # Skip boilerplate lines
        if BOILERPLATE_PATTERNS.search(text) and len(text) < 200:
            continue

        utterances.append({"speaker": dict(current_speaker), "text": text})

    if not utterances:
        # Fallback: just grab all text
        raw_text = main.get_text(separator="\n", strip=True)
        return {
            "raw_text": raw_text,
            "utterances": [],
            "speakers": [],
            "sections": {"prepared_remarks": [raw_text], "qa": []},
            "word_count": len(raw_text.split()),
        }

    # Split into sections
    qa_start = None
    for i, utt in enumerate(utterances):
        if SECTION_MARKERS.search(utt["text"]):
            qa_start = i
            break

    prepared = utterances[:qa_start] if qa_start else utterances
    qa = utterances[qa_start:] if qa_start else []

    raw_text = "\n".join(u["text"] for u in utterances)
    speakers = _deduplicate_speakers(utterances)

    return {
        "raw_text": raw_text,
        "utterances": utterances,
        "speakers": speakers,
        "sections": {
            "prepared_remarks": [u["text"] for u in prepared],
            "qa": [u["text"] for u in qa],
        },
        "word_count": len(raw_text.split()),
    }


def _is_plausible_speaker_name(name: str) -> bool:
    words = name.split()
    if len(words) < 1 or len(words) > 5:
        return False
    # Most words should be capitalized
    cap_count = sum(1 for w in words if w and w[0].isupper())
    return cap_count >= max(1, len(words) - 1)


ROLE_KEYWORDS = {
    "CEO": ["chief executive", "ceo", "president and chief"],
    "CFO": ["chief financial", "cfo"],
    "COO": ["chief operating", "coo"],
    "CTO": ["chief technology", "chief technical", "cto"],
    "President": ["president"],
    "EVP": ["executive vice president", "evp"],
    "SVP": ["senior vice president", "svp"],
    "VP": ["vice president", "vp of", "vp,"],
    "Analyst": ["analyst", "managing director", "equity research", "securities"],
    "Operator": ["operator"],
}


def _classify_role(title: str) -> str:
    t = title.lower()
    for role, keywords in ROLE_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return role
    return "Other"


def _deduplicate_speakers(utterances: list) -> list:
    seen = {}
    for utt in utterances:
        sp = utt["speaker"]
        name = sp["name"]
        if name not in seen:
            seen[name] = sp
    return list(seen.values())
