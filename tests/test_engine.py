import pytest
import engine

def fda_ok(name):
    return {"found": True, "evidence_status": "ok", "summary": "label found", "rxnorm": {"evidence_status": "ok"}}

def test_local_provider_requires_no_key(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "local")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert engine.get_llm() is None

def test_optional_provider_requires_credentials(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        engine.get_llm()

def test_invalid_provider_rejected(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "invalid")
    with pytest.raises(ValueError): engine.get_ai_provider()

def test_unknown_pair_is_not_safe(monkeypatch):
    monkeypatch.setattr(engine, "check_local_interactions", lambda drugs: {"interactions": [], "summary": "none"})
    monkeypatch.setattr(engine, "assess_patient_risk", lambda *args: {"allergy_conflicts": [], "beers_flags": [], "condition_warnings": [], "summary": "none"})
    monkeypatch.setattr(engine, "openfda_lookup", fda_ok)
    result = engine.run_analysis(30, [], [], ["alpha"], "beta")
    assert result["risk_level"] == "NO_CURATED_MATCH"
    assert "SAFE" not in result["final_answer"]

def test_missing_upstream_evidence(monkeypatch):
    monkeypatch.setattr(engine, "check_local_interactions", lambda drugs: {"interactions": [], "summary": "none"})
    monkeypatch.setattr(engine, "assess_patient_risk", lambda *args: {"allergy_conflicts": [], "beers_flags": [], "condition_warnings": [], "summary": "none"})
    monkeypatch.setattr(engine, "openfda_lookup", lambda name: {"found": False, "evidence_status": "timeout", "summary": "timeout", "rxnorm": {}})
    assert engine.run_analysis(30, [], [], [], "beta")["risk_level"] == "INSUFFICIENT_EVIDENCE"

def test_unresolved_current_or_proposed_medication_is_insufficient_evidence(monkeypatch):
    monkeypatch.setattr(engine, "check_local_interactions", lambda drugs: {
        "interactions": [], "summary": "none",
        "resolutions": [
            {"brand_name": "unknown current", "normalized": False, "evidence_status": "unresolved"},
            {"brand_name": "known proposed", "normalized": True, "evidence_status": "ok"},
        ],
    })
    monkeypatch.setattr(engine, "assess_patient_risk", lambda *args: {"allergy_conflicts": [], "beers_flags": [], "condition_warnings": [], "summary": "none"})
    monkeypatch.setattr(engine, "openfda_lookup", fda_ok)
    result = engine.run_analysis(30, [], [], ["unknown current"], "known proposed")
    assert result["risk_level"] == "INSUFFICIENT_EVIDENCE"
    assert "RxNorm could not confirm: unknown current." in result["final_answer"]

def test_high_risk_precedence_is_preserved_when_name_is_unresolved():
    report = engine.synthesize_locally(30, [], [], [], "x", {
        "interactions": [{"severity": "HIGH", "drug_a": "a", "drug_b": "b", "effect": "risk"}],
        "resolutions": [{"brand_name": "unresolved", "normalized": False, "evidence_status": "unresolved"}],
    }, {"allergy_conflicts": [], "beers_flags": [], "condition_warnings": []}, fda_ok("x"))
    assert "OVERALL RISK: HIGH_RISK" in report
    assert "RxNorm could not confirm: unresolved." in report

def test_fda_excerpts_are_escaped_and_length_limited():
    warning = "<script>alert('x')</script>" + "w" * 400
    interaction = "<b>interaction</b>" + "i" * 400
    report = engine.synthesize_locally(30, [], [], [], "x", {"interactions": [], "resolutions": []}, {"allergy_conflicts": [], "beers_flags": [], "condition_warnings": []}, {
        **fda_ok("x"), "warnings": [warning], "drug_interactions": [interaction],
    })
    assert "FDA warning excerpt: &lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;" in report
    assert "FDA interaction excerpt: &lt;b&gt;interaction&lt;/b&gt;" in report
    assert warning not in report and interaction not in report
    assert "…" in report

def test_high_rule_wins_over_fda_boxed_warning():
    report = engine.synthesize_locally(65, [], [], [], "x", {"interactions": [{"severity": "HIGH", "drug_a": "a", "drug_b": "b", "effect": "risk"}]}, {"allergy_conflicts": [], "beers_flags": [], "condition_warnings": []}, {"found": True, "evidence_status": "ok", "boxed_warning": ["warning"]})
    assert "OVERALL RISK: HIGH_RISK" in report

@pytest.mark.parametrize("age,drug", [(0, "x"), (121, "x"), (40, "")])
def test_input_validation(age, drug):
    with pytest.raises(engine.InputValidationError): engine.validate_inputs(age, [], [], [], drug)
