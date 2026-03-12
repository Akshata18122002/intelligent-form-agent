from typing import List
from src.llm import LLMClient


class MultiFormAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def answer_across_forms(self, question: str, contexts: List[str]) -> str:
        combined_context = "\n\n".join(contexts)

        prompt = f"""
You are an intelligent form assistant analyzing multiple forms.

Answer the question using only the provided contexts.

Rules:
1. Be concise and factual.
2. Combine information across forms only when clearly supported.
3. Do not invent diagnoses, names, claims, or symptoms.
4. Do not treat checklist options as confirmed findings unless they are explicitly marked.
5. If the answer cannot be determined from the provided contexts, reply exactly:
Not found in the provided forms.

Contexts:
{combined_context}

Question:
{question}

Answer:
"""
        return self.llm.generate(prompt).strip()


if __name__ == "__main__":
    client = LLMClient()
    agent = MultiFormAgent(client)

    sample_contexts = [
        "Form: form1.pdf\nReview of Systems\n□Y□N Fever □Y□N Fatigue",
        "Form: form2.pdf\nDiagnosis: Hypertension",
    ]

    question = "Across all forms, what symptoms or conditions are confirmed?"
    answer = agent.answer_across_forms(question, sample_contexts)

    print("Question:", question)
    print("Answer:", answer)

