"""Deterministic analysis orchestration for the educational prototype.

The deterministic assessment is authoritative. An optional LLM can be
configured, but never changes the computed risk level.
"""
from __future__ import annotations

import os
from html import escape
from typing import Any

from dotenv import load_dotenv
from tools import assess_patient_risk, check_local_interactions, openfda_lookup, rxnav_normalize

load_dotenv()
RISK_LEVELS = ("HIGH_RISK", "CAUTION", "NO_CURATED_MATCH", "INSUFFICIENT_EVIDENCE", "UNKNOWN")


class InputValidationError(ValueError):
    """Raised when an analysis request cannot be safely evaluated."""


def validate_inputs(age: int, allergies: list[str], conditions: list[str], current_drugs: list[str], new_drug: str) -> None:
    if not isinstance(age, int) or not 0 < age <= 120:
        raise InputValidationError("Age must be a whole number from 1 to 120.")
    if not isinstance(new_drug, str) or not new_drug.strip() or len(new_drug.strip()) > 120:
        raise InputValidationError("Enter a proposed medication name of at most 120 characters.")
    for label, values in (("allergies", allergies), ("conditions", conditions), ("current medications", current_drugs)):
        if not isinstance(values, list) or len(values) > 50 or any(not isinstance(v, str) or len(v.strip()) > 120 for v in values):
            raise InputValidationError(f"Invalid {label} input.")


def get_ai_provider() -> str:
    provider = os.getenv("AI_PROVIDER", "local").strip().lower()
    if provider not in {"local", "openai", "groq"}:
        raise ValueError("AI_PROVIDER must be local, openai, or groq.")
    return provider


def get_llm() -> Any:
    """Lazy optional client creation; local startup never needs an API key."""
    provider = get_ai_provider()
    if provider == "local":
        return None
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key or key.startswith("sk-your-"):
            raise ValueError("AI_PROVIDER=openai requires a valid OPENAI_API_KEY.")
        from openai import OpenAI
        return OpenAI(api_key=key)
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key or key.startswith("your-groq-"):
        raise ValueError("AI_PROVIDER=groq requires a valid GROQ_API_KEY.")
    from groq import Groq
    return Groq(api_key=key)


def generate_supplemental_summary(client: Any, provider: str, report: str) -> str:
    """Optional plain-language summary. It is appended, never parsed for risk."""
    prompt = ("Summarize this educational medication-evidence report in two neutral sentences. "
              "Do not give prescribing instructions, change its risk outcome, or claim safety.\n\n" + report)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini") if provider == "openai" else os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    response = client.chat.completions.create(model=model, temperature=0, messages=[{"role": "user", "content": prompt}])
    return str(response.choices[0].message.content or "").strip()


def _unresolved_names(resolutions: list[dict]) -> list[str]:
    """Return entered names whose RxNorm normalization was not confirmed."""
    return [
        str(item.get("brand_name") or item.get("generic_name") or "unknown medication")
        for item in resolutions
        if item.get("evidence_status") != "ok" or not item.get("normalized")
    ]


def _safe_excerpt(values: list[str] | str | None, limit: int = 280) -> str | None:
    """Bound a label excerpt and escape it before it reaches Markdown output."""
    if isinstance(values, list):
        text = " ".join(str(value) for value in values)
    else:
        text = str(values or "")
    text = " ".join(text.split())
    if not text:
        return None
    if len(text) > limit:
        text = text[:limit].rstrip() + "…"
    return escape(text, quote=True)


def _risk_from_evidence(interactions: list[dict], patient: dict, fda: dict, resolutions: list[dict] | None = None) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if any(item.get("severity") in {"CONTRAINDICATED", "HIGH"} for item in interactions):
        reasons.append("A curated high-severity drug interaction was found.")
    if any(item.get("severity") in {"CONTRAINDICATED", "HIGH"} for item in patient.get("allergy_conflicts", [])):
        reasons.append("A high-severity allergy-related rule was found.")
    if any(flag.get("risk") == "HIGH" for flag in patient.get("beers_flags", [])) or any("[HIGH]" in warning for warning in patient.get("condition_warnings", [])):
        reasons.append("A high-severity patient-specific rule was found.")
    if reasons:
        return "HIGH_RISK", reasons
    if interactions or patient.get("allergy_conflicts") or patient.get("beers_flags") or patient.get("condition_warnings"):
        return "CAUTION", ["A curated interaction or patient-specific caution was found."]
    unresolved = _unresolved_names(resolutions or [])
    if unresolved:
        return "INSUFFICIENT_EVIDENCE", ["RxNorm could not confirm: " + ", ".join(unresolved) + "."]
    if fda.get("evidence_status") in {"timeout", "network_error", "http_error", "malformed_response", "rate_limited"}:
        return "INSUFFICIENT_EVIDENCE", ["The FDA label source could not be evaluated reliably."]
    if not fda.get("found"):
        return "INSUFFICIENT_EVIDENCE", ["No FDA label evidence was available for the proposed medication."]
    return "NO_CURATED_MATCH", ["No matching rule was found in the limited curated database; this is not a safety determination."]


def synthesize_locally(age: int, allergies: list[str], conditions: list[str], current_drugs: list[str], new_drug: str, interaction_result: dict, patient_risk_result: dict, fda_result: dict) -> str:
    """Render an evidence-limited report without prescribing recommendations."""
    resolutions = interaction_result.get("resolutions", [])
    risk, reasons = _risk_from_evidence(interaction_result.get("interactions", []), patient_risk_result, fda_result, resolutions)
    unresolved = _unresolved_names(resolutions)
    if unresolved and not any(reason.startswith("RxNorm could not confirm:") for reason in reasons):
        reasons.append("RxNorm could not confirm: " + ", ".join(unresolved) + ".")
    lines = [f"OVERALL RISK: {risk}", "", "DRUG INTERACTIONS:"]
    lines += [f"  - [{i.get('severity', 'UNKNOWN')}] {i.get('drug_a')} + {i.get('drug_b')}: {i.get('effect', '')}" for i in interaction_result.get("interactions", [])] or ["  - No curated interaction match found."]
    lines += ["", "PATIENT-SPECIFIC RISKS:"]
    patient_items = [c.get("message", "") for c in patient_risk_result.get("allergy_conflicts", [])] + [b.get("title", "") for b in patient_risk_result.get("beers_flags", [])] + patient_risk_result.get("condition_warnings", [])
    lines += [f"  - {item}" for item in patient_items if item] or ["  - No curated patient-specific rule matched."]
    lines += ["", "FDA LABEL DATA:", f"  - {fda_result.get('summary', 'FDA label lookup was unavailable.')}"]
    warning_excerpt = _safe_excerpt(fda_result.get("warnings"))
    interaction_excerpt = _safe_excerpt(fda_result.get("drug_interactions"))
    if warning_excerpt:
        lines.append(f"  - FDA warning excerpt: {warning_excerpt}")
    if interaction_excerpt:
        lines.append(f"  - FDA interaction excerpt: {interaction_excerpt}")
    lines += ["", "EVIDENCE LIMITATION:"]
    lines += [f"  - {reason}" for reason in reasons]
    lines += ["  - This educational prototype is not clinically validated prescribing software. Verify decisions with an appropriate licensed clinician or pharmacist.", "", "EXECUTION TRACE:", f"  - RxNorm: {fda_result.get('rxnorm', {}).get('evidence_status', 'not recorded')}", f"  - Local interaction rules: {len(interaction_result.get('interactions', []))} match(es)", f"  - OpenFDA: {fda_result.get('evidence_status', 'not recorded')}"]
    return "\n".join(lines)


def run_analysis(age: int, allergies: list[str], conditions: list[str], current_drugs: list[str], new_drug: str, use_fallback: bool = False) -> dict:
    """Run real external lookups and the authoritative deterministic assessment."""
    validate_inputs(age, allergies, conditions, current_drugs, new_drug)
    all_drugs = [*current_drugs, new_drug.strip()]
    interaction = check_local_interactions(all_drugs)
    patient = assess_patient_risk(age, allergies, conditions, all_drugs)
    fda = openfda_lookup(new_drug)
    report = synthesize_locally(age, allergies, conditions, current_drugs, new_drug, interaction, patient, fda)
    risk, _ = _risk_from_evidence(interaction["interactions"], patient, fda, interaction.get("resolutions", []))
    steps = [
        {"action": "check_local_interactions", "input": all_drugs, "status": "completed", "observation": interaction["summary"]},
        {"action": "assess_patient_risk", "input": all_drugs, "status": "completed", "observation": patient["summary"]},
        {"action": "openfda_lookup", "input": new_drug, "status": fda.get("evidence_status", "unknown"), "observation": fda["summary"]},
    ]
    provider = get_ai_provider()
    if provider != "local":
        try:
            supplemental = generate_supplemental_summary(get_llm(), provider, report)
            if supplemental:
                report += "\n\nSUPPLEMENTAL SUMMARY (does not change the deterministic outcome):\n  - " + supplemental
            observation, status = "Supplemental summary generated; deterministic flags remain authoritative.", "completed"
        except Exception:
            # Configuration and provider errors are intentionally not exposed as stack traces.
            observation, status = "Optional provider unavailable; deterministic report retained.", "unavailable"
        steps.append({"action": "optional_llm", "input": provider, "status": status, "observation": observation})
    return {"final_answer": report, "risk_level": risk, "react_steps": steps, "trace_steps": steps, "rxnorm_new": fda.get("rxnorm", rxnav_normalize(new_drug)), "provider": provider}
