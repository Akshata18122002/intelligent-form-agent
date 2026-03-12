from src.llm import LLMClient
from src.form_utils import (
    detect_checked_symptoms,
    contains_checklist_section,
    checklist_interpretation_message,
)


class Summarizer:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def summarize_form(self, text: str) -> str:
        checked = detect_checked_symptoms(text)

        prompt = f"""
You are an intelligent form assistant.

Summarize this form in 5 to 7 bullet points only.

Focus only on information explicitly present in the form:
- patient/person identity
- important dates
- diagnosis, symptoms, or issue
- insurance or claim details
- medications, treatment, or follow-up

Rules:
1. Keep the summary concise.
2. Do not invent missing fields.
3. Do not treat checklist options as confirmed symptoms unless they are visibly marked.
4. If the form includes checklist sections with no marked items, mention that possible symptoms/conditions are listed but none are explicitly marked as present.
5. Do not include explanations or disclaimers.

Detected explicitly checked symptoms: {checked if checked else "None"}

Form Text:
{text}

Summary:
"""
        summary = self.llm.generate(prompt).strip()

        if not checked and contains_checklist_section(text):
            checklist_note = "- The form includes checklist-style symptom/condition sections, but no items appear explicitly marked as present."
            if checklist_note not in summary:
                summary = summary + "\n" + checklist_note

        return summary


if __name__ == "__main__":
    client = LLMClient()
    summarizer = Summarizer(client)

    sample_text = """
    Review of Systems
    □Y□N Fever □Y□N Fatigue □Y□N Yellow Skin
    """
    print(summarizer.summarize_form(sample_text))

