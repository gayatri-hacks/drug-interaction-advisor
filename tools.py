"""External evidence providers and deterministic, deliberately limited rules."""
from __future__ import annotations

import os
import re
import time
from functools import lru_cache
from itertools import combinations
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "8"))
MAX_RETRIES = int(os.getenv("HTTP_MAX_RETRIES", "2"))
RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"
OPENFDA_BASE = "https://api.fda.gov/drug/label.json"


def _request_json(url: str, *, params: dict[str, Any] | None = None) -> tuple[dict | None, str, int | None]:
    """Bounded retry helper; transport details never reach the UI."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            status = response.status_code
            if status == 404:
                return None, "not_found", status
            if status in {429, 500, 502, 503, 504}:
                if attempt < MAX_RETRIES:
                    time.sleep(0.05 * (2 ** attempt))
                    continue
                return None, "rate_limited" if status == 429 else "http_error", status
            if status >= 400:
                return None, "http_error", status
            try:
                data = response.json()
            except ValueError:
                return None, "malformed_response", status
            return data if isinstance(data, dict) else None, "ok" if isinstance(data, dict) else "malformed_response", status
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                time.sleep(0.05 * (2 ** attempt)); continue
            return None, "timeout", None
        except requests.exceptions.RequestException:
            if attempt < MAX_RETRIES:
                time.sleep(0.05 * (2 ** attempt)); continue
            return None, "network_error", None
    return None, "unknown", None


def _clean_drug_name(name: str) -> str:
    return re.sub(r"\s+\d+\s*(?:mg|mcg|ml|g|iu|units?|tablets?|capsules?)\b.*", "", name, flags=re.I).strip()


@lru_cache(maxsize=256)
def rxnav_normalize(drug_name: str) -> dict:
    clean = _clean_drug_name(drug_name)
    result = {"generic_name": clean.lower(), "brand_name": drug_name, "rxcui": None, "tty": None, "drug_classes": [], "normalized": False, "evidence_status": "unresolved"}
    if not clean:
        result.update(summary="RxNorm: no medication name supplied.", evidence_status="invalid_input"); return result
    data, status, _ = _request_json(f"{RXNAV_BASE}/approximateTerm.json", params={"term": clean, "maxEntries": 1, "option": 1})
    candidates = (data or {}).get("approximateGroup", {}).get("candidate", [])
    if status != "ok" or not candidates or not candidates[0].get("rxcui"):
        result.update(evidence_status=status if status != "ok" else "unresolved", summary=f"RxNorm: {status if status != 'ok' else 'no match'} for '{clean}'; using entered name."); return result
    rxcui = candidates[0]["rxcui"]
    props, prop_status, _ = _request_json(f"{RXNAV_BASE}/rxcui/{rxcui}/properties.json")
    if prop_status != "ok":
        result.update(evidence_status=prop_status, summary=f"RxNorm: matched '{clean}' but properties were unavailable; using entered name."); return result
    properties = props.get("properties", {})
    generic = str(properties.get("name", clean)).lower()
    result.update(generic_name=generic, rxcui=rxcui, tty=properties.get("tty"), normalized=True, evidence_status="ok", summary=f"RxNorm: '{drug_name}' → '{generic}' (RxCUI {rxcui}).")
    return result


def normalize_drug_list(drug_list: list[str]) -> tuple[list[str], list[dict]]:
    resolutions = [rxnav_normalize(drug) for drug in drug_list]
    return [entry["generic_name"] for entry in resolutions], resolutions


def openfda_lookup(drug_name: str) -> dict:
    rxnorm = rxnav_normalize(drug_name)
    base = {"drug": drug_name, "generic_name": rxnorm["generic_name"], "rxnorm": rxnorm, "found": False, "adverse_reactions": [], "drug_interactions": [], "warnings": [], "boxed_warning": []}
    params: dict[str, Any] = {"limit": 1, "search": f'openfda.generic_name:"{rxnorm["generic_name"]}"'}
    key = os.getenv("OPENFDA_API_KEY", "").strip()
    if key and key != "your-openfda-key-here": params["api_key"] = key
    data, status, code = _request_json(OPENFDA_BASE, params=params)
    if status != "ok":
        return {**base, "evidence_status": status, "http_status": code, "summary": f"OpenFDA: label lookup {status.replace('_', ' ')} for '{drug_name}'."}
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, list) or not results or not isinstance(results[0], dict):
        return {**base, "evidence_status": "not_found", "summary": f"OpenFDA: no label found for '{drug_name}'."}
    label = results[0]
    def extract(field: str, maximum: int = 500) -> list[str]:
        value = label.get(field, [])
        text = " ".join(value) if isinstance(value, list) else str(value or "")
        return [text[:maximum]] if text else []
    boxed, interactions, warnings = extract("boxed_warning", 300), extract("drug_interactions"), extract("warnings", 400) or extract("warnings_and_cautions", 400)
    return {**base, "found": True, "evidence_status": "ok", "boxed_warning": boxed, "drug_interactions": interactions, "warnings": warnings, "adverse_reactions": extract("adverse_reactions", 400), "summary": f"OpenFDA: label found for '{drug_name}'. Boxed warnings and label text are evidence to review, not automatic patient-specific contraindications."}


HIGH_RISK_PAIRS: dict[frozenset[str], dict[str, str]] = {
    frozenset(("warfarin", "aspirin")): {"severity": "HIGH", "effect": "Increased bleeding risk.", "mechanism": "Anticoagulant plus antiplatelet effect."},
    frozenset(("warfarin", "ibuprofen")): {"severity": "HIGH", "effect": "Increased bleeding risk.", "mechanism": "Anticoagulant plus NSAID effect."},
    frozenset(("sertraline", "tramadol")): {"severity": "HIGH", "effect": "Serotonin-toxicity risk.", "mechanism": "Additive serotonergic activity."},
    frozenset(("fluoxetine", "tramadol")): {"severity": "HIGH", "effect": "Serotonin-toxicity risk.", "mechanism": "Additive serotonergic activity."},
    frozenset(("sildenafil", "isosorbide")): {"severity": "CONTRAINDICATED", "effect": "Potentially dangerous hypotension.", "mechanism": "Additive vasodilation."},
    frozenset(("azithromycin", "hydroxychloroquine")): {"severity": "HIGH", "effect": "QT-prolongation risk.", "mechanism": "Additive electrophysiologic effect."},
    frozenset(("metformin", "ibuprofen")): {"severity": "MODERATE", "effect": "Renal-function-related interaction risk.", "mechanism": "NSAID renal effect."},
    frozenset(("lisinopril", "ibuprofen")): {"severity": "MODERATE", "effect": "Renal-function and blood-pressure risk.", "mechanism": "NSAID renal effect."},
}


def check_local_interactions(drug_list: list[str]) -> dict:
    generics, resolutions = normalize_drug_list(drug_list)
    found: list[dict] = []
    for (i, a), (j, b) in combinations(enumerate(generics), 2):
        for pair, evidence in HIGH_RISK_PAIRS.items():
            if (a in pair and b in pair) or (len(pair) == 2 and any(token in a for token in pair) and any(token in b for token in pair)):
                found.append({"drug_a": drug_list[i], "drug_b": drug_list[j], "generic_a": a, "generic_b": b, **evidence}); break
    summary = f"Local curated rules: {len(found)} match(es)." if found else "Local curated rules: no match. A no-match is not a safety finding."
    return {"interactions": found, "count": len(found), "resolutions": resolutions, "summary": summary}


ALLERGY_DIRECT = {"penicillin": {"amoxicillin", "ampicillin"}, "aspirin": {"aspirin"}}


def assess_patient_risk(patient_age: int, allergies: list[str], conditions: list[str], drug_list: list[str]) -> dict:
    generics, resolutions = normalize_drug_list(drug_list)
    conflicts = []
    for allergy, drugs in ALLERGY_DIRECT.items():
        if any(allergy in item.lower() for item in allergies):
            for drug in generics:
                if drug in drugs: conflicts.append({"severity": "CONTRAINDICATED", "message": f"Curated allergy rule: {allergy} allergy with {drug}."})
    warnings: list[str] = []
    condition_text = " ".join(conditions).lower()
    for drug in generics:
        if any(x in condition_text for x in ("ckd", "kidney", "renal")) and "ibuprofen" in drug: warnings.append(f"[HIGH] Curated renal rule: kidney disease with {drug}.")
    beers = [{"risk": "HIGH", "title": f"Curated older-adult rule: {drug}"} for drug in generics if patient_age >= 65 and "ibuprofen" in drug]
    items = [*warnings, *[c["message"] for c in conflicts], *[b["title"] for b in beers]]
    return {"overall_risk": "HIGH_RISK" if any("[HIGH]" in x for x in warnings) or conflicts or beers else "NO_CURATED_MATCH", "allergy_conflicts": conflicts, "beers_flags": beers, "condition_warnings": warnings, "risk_items": items, "resolutions": resolutions, "summary": "Patient rules: " + ("; ".join(items) if items else "no match. Limited rules do not establish safety.")}
