"""Safety-first orchestration for the Drug Interaction Advisor.

Deterministic checks always run first. Optional OpenAI or Groq summaries cannot
change the displayed risk level.
"""
import json
import os
from typing import Any

from dotenv import load_dotenv
from tools import assess_patient_risk, check_local_interactions, openfda_lookup, rxnav_normalize

load_dotenv()
VALID_PROVIDERS = {"local", "openai", "groq"}


class ProviderConfigurationError(ValueError):
    """Raised when an explicitly selected AI provider cannot be used."""


def selected_provider(provider: str | None = None) -> str:
    """Return an explicit provider; local rules are the keyless default."""
    value = (provider or os.getenv("LLM_PROVIDER", "local")).strip().lower()
    if value not in VALID_PROVIDERS:
        raise ProviderConfigurationError("LLM_PROVIDER must be local, openai, or groq.")
    return value


def get_llm(provider: str | None = None):
    """Create only the requested LangChain provider; never silently switch."""
    provider = selected_provider(provider)
    if provider == "local":
        return None
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key or key == "sk-your-openai-key-here":
            raise ProviderConfigurationError("OpenAI selected but OPENAI_API_KEY is not configured.")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ProviderConfigurationError("OpenAI support is not installed; run pip install -r requirements.txt.") from exc
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0, api_key=key)
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key or key == "your-groq-key-here":
        raise ProviderConfigurationError("Groq selected but GROQ_API_KEY is not configured.")
    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:
        raise ProviderConfigurationError("Groq support is not installed; run pip install -r requirements.txt.") from exc
    return ChatGroq(model=os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"), temperature=0, api_key=key)


def _risk_level(interactions: dict, patient: dict) -> str:
    """Known high-risk evidence wins; no local match is UNKNOWN, never SAFE."""
    severities = [item.get("severity", "") for item in interactions.get("interactions", [])]
    severities.extend(item.get("severity", "") for item in patient.get("allergy_conflicts", []))
    if "CONTRAINDICATED" in severities:
        return "CONTRAINDICATED"
    if "HIGH" in severities or any("[HIGH]" in item for item in patient.get("condition_warnings", [])):
        return "HIGH RISK"
    if "MODERATE" in severities or patient.get("beers_flags") or patient.get("condition_warnings"):
        return "USE WITH CAUTION"
    return "UNKNOWN"


def synthesize_locally(interactions: dict, patient: dict, fda: dict) -> str:
    """Generate an evidence report, not a prescribing recommendation."""
    risk = _risk_level(interactions, patient)
    lines = [f"OVERALL RISK: {risk}", "", "DRUG INTERACTIONS:"]
    if interactions["interactions"]:
        lines.extend(f"  • [{x['severity']}] {x['drug_a']} + {x['drug_b']}: {x['effect']} (local curated rule)" for x in interactions["interactions"])
    else:
        lines.append("  No matching curated rule was found. Rule coverage is incomplete; this is not evidence of safety.")
    lines += ["", "PATIENT-SPECIFIC RISKS:"]
    items = patient.get("risk_items", [])
    lines.extend(f"  • {item}" for item in items) if items else lines.append("  No matching local patient-risk rule was found. This does not establish suitability.")
    lines += ["", "FDA LABEL DATA:"]
    if fda.get("found"):
        lines.append(f"  OpenFDA label retrieved. Evidence URL: {fda.get('source_url') or 'not supplied'}")
        for category in ("boxed_warning", "drug_interactions", "warnings"):
            lines.extend(f"  • {text}" for text in fda.get(category, []))
    else:
        lines.append(f"  {fda['summary']}")
    lines += ["", "CLINICAL GUIDANCE:"]
    if risk in {"CONTRAINDICATED", "HIGH RISK"}:
        lines.append("  Known high-risk evidence requires licensed prescriber or pharmacist review before any treatment decision.")
    elif risk == "USE WITH CAUTION":
        lines.append("  Local findings warrant pharmacist/prescriber review; do not use this report as prescribing guidance.")
    else:
        lines.append("  Evidence is incomplete or unavailable. Seek authoritative label and pharmacist review; no safety conclusion is made.")
    lines += ["", "TRACEABILITY:",
              f"  • Local interaction rules: {len(interactions['interactions'])} matches (coverage: incomplete).",
              f"  • Patient rules/FAISS: {len(items)} findings.",
              f"  • RxNorm status: {fda['rxnorm'].get('status')}; RxCUI: {fda['rxnorm'].get('rxcui') or 'unavailable'}.",
              f"  • OpenFDA status: {fda.get('status')}; URL: {fda.get('source_url') or 'unavailable'}.",
              "", "[Educational research output only. It is not clinically validated, prescribing approval, or professional medical advice.]",
              "[Generated by local rules; no AI clinical assessment was used.]"
              ]
    return "\n".join(lines)


def _optional_ai_summary(provider: str, evidence: str) -> tuple[str | None, str | None]:
    if provider == "local":
        return None, None
    try:
        prompt = ("Summarize supplied drug-safety evidence in 3 concise educational bullets. Do not provide prescribing advice, "
                  "claim safety, contradict or change the stated risk. Do not expose chain-of-thought. Evidence:\n" + evidence)
        message = get_llm(provider).invoke(prompt)
        return getattr(message, "content", str(message)), None
    except Exception as exc:
        return None, f"{provider.title()} summary unavailable: {str(exc)[:180]}"


def run_analysis(age: int, allergies: list[str], conditions: list[str], current_drugs: list[str],
                 new_drug: str, provider: str | None = None, use_fallback: bool = False) -> dict[str, Any]:
    """Run real RxNorm/OpenFDA lookups plus local checks, with optional AI summary."""
    provider = "local" if use_fallback else selected_provider(provider)
    all_drugs = current_drugs + [new_drug]
    rxnorm_new = rxnav_normalize(new_drug)
    interactions = check_local_interactions(all_drugs)
    patient = assess_patient_risk(age, allergies, conditions, all_drugs)
    fda = openfda_lookup(new_drug)
    final_answer = synthesize_locally(interactions, patient, fda)
    ai_summary, provider_message = _optional_ai_summary(provider, final_answer)
    if ai_summary:
        final_answer += "\n\nAI SUMMARY (supplemental; cannot override the evidence report):\n  " + ai_summary
    if provider_message:
        final_answer += "\n\nAI PROVIDER STATUS:\n  " + provider_message
    steps = [
        {"action": "check_local_interactions", "action_input": json.dumps(all_drugs), "observation": interactions["summary"]},
        {"action": "assess_patient_risk", "action_input": json.dumps(all_drugs), "observation": patient["summary"]},
        {"action": "openfda_lookup", "action_input": new_drug, "observation": fda["summary"]},
    ]
    return {"final_answer": final_answer, "risk_level": _risk_level(interactions, patient), "react_steps": steps,
            "trace_steps": [], "rxnorm_new": rxnorm_new, "provider": provider, "provider_message": provider_message,
            "evidence": {"interactions": interactions, "patient_risk": patient, "openfda": fda}}
