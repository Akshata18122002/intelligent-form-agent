from pathlib import Path
import pdfplumber


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    """

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    text_pages = []

    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text_pages.append(page_text.strip())

    full_text = "\n\n".join(text_pages)

    return full_text


if __name__ == "__main__":
    # simple test
    sample_pdf = "data/forms/form3.pdf"

    text = extract_text_from_pdf(sample_pdf)

    if text:
        print("Extraction successful\n")
        print(text[:1500])  # preview first part of the text
    else:
        print("No text could be extracted from the PDF.")