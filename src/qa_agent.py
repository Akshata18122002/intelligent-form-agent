from typing import List
from src.llm import LLMClient
from src.form_utils import (
    detect_checked_symptoms,
    contains_checklist_section,
    checklist_interpretation_message,
)


class QAAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def answer_question(self, question: str, retrieved_chunks: List[str]) -> str:
        context = "\n\n".join(retrieved_chunks)

        question_lower = question.lower()
        is_symptom_question = any(
            phrase in question_lower
            for phrase in [
                "symptom",
                "symptoms",
                "condition",
                "conditions",
                "medical issue",
                "medical issues",
                "issue",
                "issues",
                "review of systems",
            ]
        )

        if is_symptom_question:
            checked = detect_checked_symptoms(context)

            if checked:
                return "The following symptoms are explicitly marked as present in the form: " + ", ".join(checked)

            if contains_checklist_section(context):
                return checklist_interpretation_message()

        prompt = f"""
You are an intelligent form assistant.

Answer the user's question using only the provided form context.

Rules:
1. Give a short direct answer.
2. Do not explain your reasoning.
3. Do not add extra commentary.
4. If the answer is not explicitly present in the context, reply exactly:
Not found in the form.
5. If the form contains checklist-style symptom options but no boxes are visibly marked,
   state that the form lists possible symptoms but none are confirmed as present.

Form Context:
{context}

Question:
{question}

Answer:
"""
        return self.llm.generate(prompt).strip()


if __name__ == "__main__":
    client = LLMClient()
    agent = QAAgent(client)

    sample_chunks = [
        "Review of Systems\n□Y□N Fever □Y□N Fatigue □Y□N Yellow Skin",
    ]

    question = "What symptoms or medical issues are mentioned in this form?"
    answer = agent.answer_question(question, sample_chunks)

    print("Question:", question)
    print("Answer:", answer)