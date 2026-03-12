import re
from typing import List


def detect_checked_symptoms(text: str) -> List[str]:
    """
    Detect symptoms only when a checkbox appears explicitly filled.
    Supported checked markers:
    - ✓
    - ✔
    - ☑


    This intentionally does NOT treat patterns like '□Y□N Fever'
    as checked, because in blank forms they are just options.
    """
    checked_patterns = [
        r"[✓✔]\s*([A-Za-z][A-Za-z /-]*)",
        r"☑\s*([A-Za-z][A-Za-z /-]*)"
    ]

    symptoms: List[str] = []

    for pattern in checked_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            symptom = match.strip()
            if symptom and symptom not in symptoms:
                symptoms.append(symptom)

    return symptoms


def contains_checklist_section(text: str) -> bool:
    """
    Heuristic to identify checklist-heavy medical review sections.
    """
    checklist_markers = [
        "Review of Systems",
        "General Medical Questionnaire",
        "Please indicate ALL that you have experienced",
        "□Y□N",
        "□ Y □ N",
    ]
    return any(marker.lower() in text.lower() for marker in checklist_markers)


def checklist_interpretation_message() -> str:
    return (
        "The form contains a checklist of possible symptoms or conditions, "
        "but none appear to be explicitly marked as present in the provided form."
    )


_FILLED_FIELD_RE = re.compile(r"^\s*[^:\n]{2,80}:\s*(.+?)\s*$")


def looks_unfilled_template(text: str) -> bool:
    """
    Heuristic: treat as "unfilled" if we can't find any line that looks like
    `Label: <non-empty value>`.
    """
    if not text or not text.strip():
        return True

    filled = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m = _FILLED_FIELD_RE.match(line)
        if not m:
            continue

        value = m.group(1).strip()
        value = value.replace("_", "").replace("•", "").strip()

        if len(value) >= 2 and any(ch.isalnum() for ch in value):
            filled += 1
            break

    return filled == 0

