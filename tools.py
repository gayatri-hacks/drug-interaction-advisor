"""
tools.py — The "API Sandwich"
==============================
Three tools powering the ReAct agent:

  1. rxnav_normalize(drug_name)
     → RxNorm API (NLM) — resolves brand/generic names, e.g. "Advil" → "ibuprofen"
     → FREE, no API key required

  2. openfda_lookup(drug_name)
     → OpenFDA Drug Label API — fetches official FDA adverse events + interactions
     → FREE (40 req/min without key, 240/min with key)

  3. check_local_interactions(drug_list)
     → Instant local high-risk dictionary — 25+ curated pairs with severity + mechanism

  4. assess_patient_risk(age, allergies, conditions, drug_list)
     → Allergy cross-reactivity + Beers Criteria (FAISS) + condition-drug rules
"""

import os
import re
import json
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OPENFDA_KEY     = os.getenv("OPENFDA_API_KEY", "")
REQUEST_TIMEOUT = 10  # seconds

# ══════════════════════════════════════════════════════════════════════════════
# TOOL 1 — RxNav / RxNorm  (National Library of Medicine)
# "Advil" → "ibuprofen", "Tylenol" → "acetaminophen"
# ══════════════════════════════════════════════════════════════════════════════

RXNAV_BASE = "https://rxnav.nlm.nih.gov/REST"
_rxnorm_cache: dict[str, dict] = {}  # in-memory cache


def rxnav_normalize(drug_name: str) -> dict:
    """
    Resolves a brand or generic drug name to its canonical RxNorm entry.

    Steps:
      1. Strip dose: "Ibuprofen 400mg" → "Ibuprofen"
      2. Approximate match → get RxCUI
      3. Resolve to ingredient (IN) if branded term
      4. Fetch ATC drug class (best-effort)

    Returns dict with: generic_name, rxcui, drug_classes, summary, normalized
    """
    # Strip dose suffixes: "Advil 200mg", "Warfarin 5 mg", "Metformin XR 500mg"
    clean = re.sub(
        r"\s+\d+\s*(?:mg|mcg|ml|g|iu|units?|tab|cap|er|xr|sr)\b.*",
        "", drug_name, flags=re.I
    ).strip()

    cache_key = clean.lower()
    if cache_key in _rxnorm_cache:
        return _rxnorm_cache[cache_key]

    result = {
        "generic_name": clean.lower(),
        "brand_name":   drug_name,
        "rxcui":        None,
        "tty":          None,
        "drug_classes": [],
        "summary":      f"RxNorm: Could not resolve '{drug_name}' — using as-is.",
        "normalized":   False,
        "status":       "unavailable",
        "source_url":   None,
    }

    try:
        # Step 1 — approximate match
        resp = requests.get(
            f"{RXNAV_BASE}/approximateTerm.json",
            params={"term": clean, "maxEntries": 1, "option": 1},
            timeout=REQUEST_TIMEOUT,
        )
        result["source_url"] = resp.url
        resp.raise_for_status()
        candidates = resp.json().get("approximateGroup", {}).get("candidate", [])

        if not candidates:
            result["status"] = "not_found"
            _rxnorm_cache[cache_key] = result
            return result

        rxcui = candidates[0].get("rxcui")
        if not rxcui:
            result["status"] = "not_found"
            _rxnorm_cache[cache_key] = result
            return result

        # Step 2 — get properties
        props_resp = requests.get(
            f"{RXNAV_BASE}/rxcui/{rxcui}/properties.json",
            timeout=REQUEST_TIMEOUT,
        )
        props_resp.raise_for_status()
        props        = props_resp.json().get("properties", {})
        generic_name = props.get("name", clean).lower()
        tty          = props.get("tty", "")

        # Step 3 — resolve branded → ingredient
        if tty in ("SBD", "BPCK", "BN", "SBD"):
            rel_resp = requests.get(
                f"{RXNAV_BASE}/rxcui/{rxcui}/related.json",
                params={"tty": "IN"},
                timeout=REQUEST_TIMEOUT,
            )
            rel_resp.raise_for_status()
            groups = rel_resp.json().get("relatedGroup", {}).get("conceptGroup", [])
            for grp in groups:
                concepts = grp.get("conceptProperties", [])
                if concepts:
                    generic_name = concepts[0].get("name", generic_name).lower()
                    rxcui        = concepts[0].get("rxcui", rxcui)
                    tty          = "IN"
                    break

        # Step 4 — drug class (best-effort, won't fail the whole call)
        drug_classes = []
        try:
            cls_resp = requests.get(
                f"{RXNAV_BASE}/rxclass/class/byRxcui.json",
                params={"rxcui": rxcui, "relaSource": "ATC1-4"},
                timeout=REQUEST_TIMEOUT,
            )
            cls_resp.raise_for_status()
            cls_data = (
                cls_resp.json()
                .get("rxclassDrugInfoList", {})
                .get("rxclassDrugInfo", [])
            )
            drug_classes = list({
                c.get("rxclassMinConceptItem", {}).get("className", "")
                for c in cls_data
                if c.get("rxclassMinConceptItem", {}).get("className")
            })[:3]
        except Exception:
            pass

        result.update({
            "generic_name": generic_name,
            "rxcui":        rxcui,
            "tty":          tty,
            "drug_classes": drug_classes,
            "normalized":   True,
            "status":       "success",
            "summary": (
                f"RxNorm: '{drug_name}' → '{generic_name}' "
                f"(RxCUI {rxcui}"
                + (f", {', '.join(drug_classes)}" if drug_classes else "")
                + ")"
            ),
        })

    except requests.exceptions.ConnectionError:
        result["status"] = "network_error"
        result["summary"] = f"RxNorm: Network unavailable — using '{clean.lower()}' as-is."
    except requests.exceptions.Timeout:
        result["status"] = "timeout"
        result["summary"] = f"RxNorm: Timeout for '{drug_name}' — using as-is."
    except Exception as e:
        result["status"] = "api_error"
        result["summary"] = f"RxNorm: Error for '{drug_name}': {str(e)[:80]}"

    _rxnorm_cache[cache_key] = result
    return result


def normalize_drug_list(drug_list: list[str]) -> tuple[list[str], list[dict]]:
    """Normalize a list via RxNav. Returns (generics_list, full_resolution_dicts)."""
    generics, resolutions = [], []
    for drug in drug_list:
        res = rxnav_normalize(drug)
        generics.append(res["generic_name"])
        resolutions.append(res)
    return generics, resolutions


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 2 — OpenFDA Drug Label API
# ══════════════════════════════════════════════════════════════════════════════

OPENFDA_BASE = "https://api.fda.gov/drug/label.json"


def openfda_lookup(drug_name: str) -> dict:
    """
    Queries OpenFDA Drug Label for adverse reactions, drug interactions,
    warnings, and black-box warnings.

    Normalizes the drug name via RxNorm first for best search results.
    """
    rxnorm      = rxnav_normalize(drug_name)
    search_name = rxnorm["generic_name"]

    params: dict = {"limit": 1}
    if OPENFDA_KEY and OPENFDA_KEY not in ("your-openfda-key-here", ""):
        params["api_key"] = OPENFDA_KEY

    attempts: list[dict] = []

    def _try(field: str, value: str):
        try:
            p = {**params, "search": f'{field}:"{value}"'}
            r = requests.get(OPENFDA_BASE, params=p, timeout=REQUEST_TIMEOUT)
            attempts.append({"field": field, "status_code": r.status_code, "url": r.url})
            if r.status_code == 200:
                data = r.json()
                if data.get("results"):
                    return data["results"][0], "success", r.url
            if r.status_code == 404:
                return None, "not_found", r.url
            if r.status_code == 429:
                return None, "rate_limited", r.url
            return None, "api_error", r.url
        except requests.exceptions.Timeout:
            attempts.append({"field": field, "status_code": "timeout", "url": None})
            return None, "timeout", None
        except requests.exceptions.ConnectionError:
            attempts.append({"field": field, "status_code": "network_error", "url": None})
            return None, "network_error", None
        except requests.RequestException:
            attempts.append({"field": field, "status_code": "api_error", "url": None})
            return None, "api_error", None
        return None, "api_error", None

    label = None
    status, source_url = "not_found", None
    for field, value in (
        ("openfda.generic_name", search_name),
        ("openfda.generic_name", drug_name.lower()),
        ("openfda.brand_name", drug_name),
        ("openfda.substance_name", search_name),
    ):
        label, attempt_status, source_url = _try(field, value)
        if label:
            status = "success"
            break
        # A definitive service failure should not be presented as "no record".
        if attempt_status in {"timeout", "network_error", "rate_limited", "api_error"}:
            status = attempt_status
            break

    base = {
        "drug": drug_name, "generic_name": search_name,
        "rxnorm": rxnorm,
        "adverse_reactions": [], "drug_interactions": [],
        "warnings": [], "boxed_warning": [],
        "status": status, "source_url": source_url, "attempts": attempts,
    }

    if not label:
        return {
            **base, "found": False,
            "summary": (
                (f"OpenFDA: No label record found for '{drug_name}' "
                 f"(searched: '{search_name}')." if status == "not_found" else
                 f"OpenFDA: {status.replace('_', ' ')} while retrieving '{drug_name}'. "
                 "FDA label evidence is unavailable; this is not a negative finding.")
            ),
        }

    def _extract(key: str, max_chars: int = 500) -> list[str]:
        val  = label.get(key, [])
        if not val:
            return []
        text = " ".join(val) if isinstance(val, list) else str(val)
        return [text[:max_chars] + ("..." if len(text) > max_chars else "")]

    brand_names   = label.get("openfda", {}).get("brand_name",   [drug_name])
    generic_names = label.get("openfda", {}).get("generic_name", [search_name])
    boxed         = _extract("boxed_warning",         300)
    interactions  = _extract("drug_interactions",     500)
    adverse       = _extract("adverse_reactions",     400)
    warnings      = _extract("warnings",              400) or _extract("warnings_and_cautions", 400)

    parts = [
        f"OpenFDA — {generic_names[0].title()} "
        f"({brand_names[0] if brand_names else 'N/A'}) "
        f"[RxNorm: {rxnorm['summary']}]:"
    ]
    if boxed:
        parts.append(f"⚠ BLACK BOX: {boxed[0]}")
    if interactions:
        parts.append(f"Drug Interactions: {interactions[0]}")
    if adverse:
        parts.append(f"Adverse Reactions: {adverse[0]}")
    if warnings and not boxed:
        parts.append(f"Warnings: {warnings[0]}")

    return {
        **base, "found": True, "status": "success",
        "brand_name":        brand_names[0] if brand_names else drug_name,
        "adverse_reactions": adverse,
        "drug_interactions": interactions,
        "warnings":          warnings,
        "boxed_warning":     boxed,
        "summary":           "\n".join(parts),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 3 — Local High-Risk Interaction Dictionary
# ══════════════════════════════════════════════════════════════════════════════

HIGH_RISK_PAIRS: dict[frozenset, dict] = {
    frozenset(["warfarin",    "aspirin"]):         {"severity":"HIGH",          "effect":"Major bleeding — warfarin INR elevation + platelet inhibition.",                        "mechanism":"Protein displacement + COX-1",       "action":"Avoid. Use acetaminophen. Monitor INR."},
    frozenset(["warfarin",    "ibuprofen"]):        {"severity":"HIGH",          "effect":"Increased bleeding — INR elevation and platelet inhibition.",                           "mechanism":"Protein displacement + COX-1",       "action":"Avoid NSAIDs. Acetaminophen preferred."},
    frozenset(["warfarin",    "naproxen"]):         {"severity":"HIGH",          "effect":"GI bleeding and elevated INR.",                                                         "mechanism":"Protein displacement + COX-1",       "action":"Avoid. Acetaminophen preferred."},
    frozenset(["warfarin",    "fluconazole"]):      {"severity":"HIGH",          "effect":"CYP2C9 inhibition — INR may triple within days.",                                       "mechanism":"CYP2C9 inhibition",                  "action":"Reduce warfarin 25-50%. Daily INR monitoring."},
    frozenset(["warfarin",    "acetaminophen"]):    {"severity":"MODERATE",      "effect":"High-dose acetaminophen (>2g/day) potentiates warfarin.",                               "mechanism":"CYP modulation",                     "action":"Limit acetaminophen ≤2g/day. Monitor INR."},
    frozenset(["sertraline",  "tramadol"]):         {"severity":"HIGH",          "effect":"Serotonin syndrome — agitation, hyperthermia, seizures.",                               "mechanism":"Additive serotonergic stimulation",  "action":"Contraindicated. Use alternative analgesic."},
    frozenset(["fluoxetine",  "tramadol"]):         {"severity":"HIGH",          "effect":"Serotonin syndrome.",                                                                   "mechanism":"Additive SERT inhibition",           "action":"Do not co-prescribe."},
    frozenset(["ssri",        "maoi"]):             {"severity":"CONTRAINDICATED","effect":"Fatal serotonin syndrome.",                                                             "mechanism":"Massive serotonin excess",            "action":"Absolutely contraindicated. 14-day washout."},
    frozenset(["sildenafil",  "nitrate"]):          {"severity":"CONTRAINDICATED","effect":"Life-threatening hypotension.",                                                         "mechanism":"Additive cGMP vasodilation",          "action":"Never co-prescribe."},
    frozenset(["sildenafil",  "isosorbide"]):       {"severity":"CONTRAINDICATED","effect":"Life-threatening hypotension — PDE5 + nitrate.",                                        "mechanism":"Additive cGMP vasodilation",          "action":"Never co-prescribe."},
    frozenset(["tadalafil",   "isosorbide"]):       {"severity":"CONTRAINDICATED","effect":"All PDE5 inhibitors contraindicated with any nitrate.",                                 "mechanism":"Additive cGMP vasodilation",          "action":"Never co-prescribe."},
    frozenset(["metformin",   "ibuprofen"]):        {"severity":"MODERATE",      "effect":"NSAIDs reduce metformin clearance → lactic acidosis, especially in CKD.",               "mechanism":"Reduced renal prostaglandins → ↓GFR", "action":"Avoid in CKD. Use acetaminophen."},
    frozenset(["lisinopril",  "ibuprofen"]):        {"severity":"MODERATE",      "effect":"NSAIDs blunt ACE inhibitor effect and may cause AKI.",                                  "mechanism":"Renal prostaglandin inhibition",       "action":"Monitor renal function and BP."},
    frozenset(["enalapril",   "ibuprofen"]):        {"severity":"MODERATE",      "effect":"ACE inhibitor + NSAID: reduced BP control and AKI risk.",                               "mechanism":"Renal prostaglandin inhibition",       "action":"Monitor renal function and BP."},
    frozenset(["digoxin",     "amiodarone"]):       {"severity":"HIGH",          "effect":"Digoxin levels 1.5-2x → toxicity: bradycardia, arrhythmia.",                            "mechanism":"P-glycoprotein inhibition",            "action":"Reduce digoxin 30-50%. Monitor levels."},
    frozenset(["simvastatin", "amiodarone"]):       {"severity":"HIGH",          "effect":"Statin accumulation → myopathy, rhabdomyolysis.",                                       "mechanism":"CYP3A4 inhibition",                  "action":"Use rosuvastatin or pravastatin instead."},
    frozenset(["atorvastatin","clarithromycin"]):   {"severity":"HIGH",          "effect":"Statin levels increase → myopathy.",                                                    "mechanism":"CYP3A4 inhibition",                  "action":"Hold statin during clarithromycin course."},
    frozenset(["lithium",     "ibuprofen"]):        {"severity":"HIGH",          "effect":"NSAIDs reduce lithium clearance → toxicity (tremor, seizures).",                        "mechanism":"Renal prostaglandin inhibition",       "action":"Avoid NSAIDs. Use acetaminophen."},
    frozenset(["azithromycin","hydroxychloroquine"]):{"severity":"HIGH",         "effect":"Additive QT prolongation → torsades de pointes.",                                       "mechanism":"Additive hERG blockade",              "action":"Avoid. ECG monitoring if required."},
    frozenset(["methotrexate","aspirin"]):          {"severity":"HIGH",          "effect":"NSAIDs reduce methotrexate clearance → bone marrow toxicity.",                          "mechanism":"Organic anion transporter competition","action":"Avoid. Monitor CBC."},
    frozenset(["metoprolol",  "verapamil"]):        {"severity":"HIGH",          "effect":"Complete heart block or severe bradycardia.",                                            "mechanism":"Additive AV nodal suppression",       "action":"Avoid. Continuous monitoring if required."},
    frozenset(["clopidogrel", "omeprazole"]):       {"severity":"MODERATE",      "effect":"Reduced clopidogrel activation → less antiplatelet effect.",                            "mechanism":"CYP2C19 inhibition",                 "action":"Use pantoprazole instead of omeprazole."},
    frozenset(["ciprofloxacin","theophylline"]):    {"severity":"HIGH",          "effect":"Theophylline toxicity — seizures, arrhythmias.",                                        "mechanism":"CYP1A2 inhibition",                  "action":"Avoid. Reduce theophylline dose 30-50%."},
    frozenset(["heparin",     "aspirin"]):          {"severity":"MODERATE",      "effect":"Combined anticoagulant + antiplatelet raises bleeding risk.",                            "mechanism":"Pharmacodynamic synergy",             "action":"Often intentional in ACS — monitor bleeding."},
    frozenset(["metformin",   "contrast"]):         {"severity":"HIGH",          "effect":"IV contrast → AKI → metformin accumulation → lactic acidosis.",                         "mechanism":"Renal clearance reduction",           "action":"Hold metformin 48h before/after contrast."},
}


def check_local_interactions(drug_list: list[str]) -> dict:
    """
    Normalizes all drugs via RxNorm then checks every pair against HIGH_RISK_PAIRS.
    """
    generics, resolutions = normalize_drug_list(drug_list)
    norm_log = [f"  '{r['brand_name']}' → '{r['generic_name']}'" for r in resolutions]

    found:   list[dict]      = []
    checked: set[frozenset]  = set()

    for i, a in enumerate(generics):
        for j, b in enumerate(generics):
            if i >= j:
                continue
            pair_idx = frozenset([i, j])
            if pair_idx in checked:
                continue
            checked.add(pair_idx)

            # Exact match
            exact = frozenset([a, b])
            if exact in HIGH_RISK_PAIRS:
                found.append({"drug_a": drug_list[i], "drug_b": drug_list[j],
                               "generic_a": a, "generic_b": b, **HIGH_RISK_PAIRS[exact]})
                continue

            # Fuzzy: key fragment contained in generic name
            for kpair, interaction in HIGH_RISK_PAIRS.items():
                keys = list(kpair)
                hit_a = any(k in a for k in keys)
                hit_b = any(k in b for k in keys)
                if hit_a and hit_b and a != b:
                    # Make sure a and b match *different* keys
                    keys_matching_a = [k for k in keys if k in a]
                    keys_matching_b = [k for k in keys if k in b]
                    if set(keys_matching_a) != set(keys_matching_b):
                        found.append({"drug_a": drug_list[i], "drug_b": drug_list[j],
                                       "generic_a": a, "generic_b": b, **interaction})
                        break

    if not found:
        summary = ("Local DB: No matching curated interactions found. Coverage is incomplete; this does NOT establish safety.\n"
                   "RxNorm resolutions:\n" + "\n".join(norm_log))
    else:
        lines = [
            f"[{f['severity']}] {f['drug_a']} + {f['drug_b']} "
            f"({f['generic_a']} + {f['generic_b']}): "
            f"{f['effect']} → {f['action']}"
            for f in found
        ]
        summary = (
            f"Local DB: {len(found)} interaction(s) found.\n"
            "RxNorm resolutions:\n" + "\n".join(norm_log)
            + "\n\n" + "\n".join(lines)
        )

    return {"interactions": found, "count": len(found), "coverage": "incomplete",
            "resolutions": resolutions, "summary": summary}


# ══════════════════════════════════════════════════════════════════════════════
# TOOL 4 — Patient Risk Assessment
# ══════════════════════════════════════════════════════════════════════════════

ALLERGY_MAP: dict[str, dict] = {
    "penicillin": {"direct": ["amoxicillin","ampicillin","piperacillin","flucloxacillin","nafcillin","oxacillin"],
                   "cross":  ["cephalexin","cefazolin","ceftriaxone","cefuroxime","cefdinir","cefepime"],
                   "severity_direct":"CONTRAINDICATED","severity_cross":"HIGH",
                   "mechanism":"Beta-lactam ring cross-reactivity"},
    "sulfa":      {"direct": ["sulfamethoxazole","sulfadiazine","sulfisoxazole"],
                   "cross":  [],
                   "severity_direct":"HIGH","severity_cross":"MODERATE",
                   "mechanism":"Antimicrobial sulfonamide allergy; non-antimicrobial cross-reactivity is not assumed"},
    "aspirin":    {"direct": ["aspirin","acetylsalicylic"],
                   "cross":  ["ibuprofen","naproxen","ketorolac","indomethacin","diclofenac","meloxicam"],
                   "severity_direct":"HIGH","severity_cross":"HIGH",
                   "mechanism":"COX-1 hypersensitivity (AERD)"},
    "codeine":    {"direct": ["codeine","dihydrocodeine"],
                   "cross":  [],
                   "severity_direct":"HIGH","severity_cross":"MODERATE",
                   "mechanism":"Reported opioid allergy requires clinical history; class-wide cross-reactivity is not assumed"},
}


def assess_patient_risk(
    patient_age: int,
    allergies:   list[str],
    conditions:  list[str],
    drug_list:   list[str],
) -> dict:
    """
    Checks allergy cross-reactivity, Beers Criteria (via FAISS), and
    condition-specific contraindications for the given drug list.
    """
    from build_knowledge_base import search_guidelines

    generics, resolutions = normalize_drug_list(drug_list)

    risks:              list[str]  = []
    allergy_conflicts:  list[dict] = []
    beers_flags:        list[dict] = []
    condition_warnings: list[str]  = []

    # ── Allergy checks ────────────────────────────────────────────────────────
    for allergy in allergies:
        ak = allergy.lower().strip()
        for map_key, data in ALLERGY_MAP.items():
            if map_key not in ak and ak not in map_key:
                continue
            for generic in generics:
                for d in data["direct"]:
                    if d in generic or generic.startswith(d[:6]):
                        msg = (f"ALLERGY [{data['severity_direct']}]: '{generic}' is in the same "
                               f"class as '{allergy}' ({data['mechanism']}).")
                        allergy_conflicts.append({"type":"DIRECT","allergen":allergy,
                                                   "drug":generic,"severity":data["severity_direct"],"message":msg})
                        risks.append(msg)
                        break
                for d in data["cross"]:
                    if d in generic or generic.startswith(d[:6]):
                        msg = (f"CROSS-REACTIVITY [{data['severity_cross']}]: '{generic}' "
                               f"cross-reacts with '{allergy}' allergy ({data['mechanism']}).")
                        allergy_conflicts.append({"type":"CROSS","allergen":allergy,
                                                   "drug":generic,"severity":data["severity_cross"],"message":msg})
                        risks.append(msg)
                        break

    # ── Beers Criteria (FAISS) ────────────────────────────────────────────────
    if patient_age >= 65:
        for i, drug in enumerate(drug_list):
            generic = generics[i]
            results = search_guidelines(f"elderly patient {patient_age} taking {generic}", top_k=2)
            for r in results:
                if r.get("category") == "elderly" and r.get("score", 0) > 0.25:
                    beers_flags.append({
                        "drug": drug, "generic": generic,
                        "rule_id": r["rule_id"], "risk": r["risk"],
                        "title": r["title"], "snippet": r["text"][:220] + "...",
                    })
                    risks.append(f"[BEERS {r['risk']}] {drug} ({generic}): {r['title']}")

    # ── Condition-specific rules ──────────────────────────────────────────────
    conds    = [c.lower() for c in conditions]
    ckd      = any(k in c for c in conds for k in ["kidney","renal","ckd","nephropathy"])
    liver    = any(k in c for c in conds for k in ["liver","hepatic","cirrhosis"])
    hf       = any(k in c for c in conds for k in ["heart failure","chf","hfref"])
    diabetes = any(k in c for c in conds for k in ["diabetes","diabetic"])
    epilepsy = any(k in c for c in conds for k in ["epilepsy","seizure"])
    arrhy    = any(k in c for c in conds for k in ["atrial fibrillation","arrhythmia","long qt"])
    gi       = any(k in c for c in conds for k in ["ulcer","gi bleed","gastritis"])

    for g in generics:
        is_nsaid = any(k in g for k in ["ibuprofen","naproxen","ketorolac","diclofenac","indomethacin"])
        if ckd      and is_nsaid:            w=f"[HIGH] CKD + {g}: NSAIDs reduce GFR → AKI.";                              condition_warnings.append(w); risks.append(w)
        if ckd      and "metformin"   in g:  w=f"[HIGH] CKD + {g}: lactic acidosis risk.";                                 condition_warnings.append(w); risks.append(w)
        if hf       and is_nsaid:            w=f"[HIGH] Heart Failure + {g}: sodium retention, worsens HF.";               condition_warnings.append(w); risks.append(w)
        if liver    and "acetaminophen" in g: w=f"[MODERATE] Liver disease + {g}: limit ≤2g/day.";                         condition_warnings.append(w); risks.append(w)
        if diabetes and any(k in g for k in ["prednisone","dexamethasone"]): w=f"[MODERATE] Diabetes + {g}: hyperglycemia."; condition_warnings.append(w); risks.append(w)
        if epilepsy and any(k in g for k in ["tramadol","bupropion","meperidine"]): w=f"[HIGH] Epilepsy + {g}: lowers seizure threshold."; condition_warnings.append(w); risks.append(w)
        if arrhy    and any(k in g for k in ["azithromycin","ciprofloxacin","hydroxychloroquine","ondansetron"]): w=f"[HIGH] Arrhythmia + {g}: QT prolongation."; condition_warnings.append(w); risks.append(w)
        if gi       and is_nsaid:            w=f"[HIGH] GI disease + {g}: NSAIDs contraindicated with active ulcer.";       condition_warnings.append(w); risks.append(w)

    # ── Overall risk ──────────────────────────────────────────────────────────
    has_contra   = any(c["severity"]=="CONTRAINDICATED" for c in allergy_conflicts)
    has_high     = (any(c["severity"] in ("HIGH","CONTRAINDICATED") for c in allergy_conflicts)
                    or any(b["risk"]=="HIGH" for b in beers_flags)
                    or any("[HIGH]" in w for w in condition_warnings))
    has_moderate = (any(c["severity"]=="MODERATE" for c in allergy_conflicts)
                    or any(b["risk"]=="MODERATE" for b in beers_flags)
                    or any("[MODERATE]" in w for w in condition_warnings)
                    or bool(beers_flags))

    overall = ("CONTRAINDICATED" if has_contra else
               "HIGH RISK"       if has_high   else
               "USE WITH CAUTION" if has_moderate else "LOW RISK")

    summary = (
        f"Patient Risk [{overall}]: No flags found."
        if not risks else
        f"Patient Risk Assessment [{overall}]:\n" + "\n".join(f"  • {r}" for r in risks)
    )

    return {"overall_risk": overall, "allergy_conflicts": allergy_conflicts,
            "beers_flags": beers_flags, "condition_warnings": condition_warnings,
            "risk_items": risks, "resolutions": resolutions, "summary": summary}
