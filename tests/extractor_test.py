from src.extractor import extract_text_from_pdf


def test_extractor_returns_string():
    text = extract_text_from_pdf("data/forms/form1.pdf")
    assert isinstance(text, str)


def test_extractor_returns_non_empty_text():
    text = extract_text_from_pdf("data/forms/form1.pdf")
    assert len(text.strip()) > 0