from src.form_utils import detect_checked_symptoms


def test_detect_checked_symptoms_respects_explicit_checks_only():
    text = """
    Review of Systems
    □Y□N Fever
    □Y□N Cough
    ✓ Shortness of breath
    [x] Chest pain
    ☑ Palpitations
    """

    symptoms = detect_checked_symptoms(text)

    # Only explicitly checked items should be returned
    assert "Shortness of breath" in symptoms
    
    assert "Palpitations" in symptoms
    # Unchecked checklist options should not appear
    assert "Fever" not in symptoms
    assert "Cough" not in symptoms
    assert "Chest pain" not in symptoms

