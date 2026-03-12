from src.llm import LLMClient


def test_llm_client_generates_text():
    client = LLMClient()
    response = client.generate("Explain what a medical form is in one sentence.")

    assert isinstance(response, str)
    assert len(response.strip()) > 0

