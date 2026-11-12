"""Streamlit presentation layer; uses Streamlit's escaped text components."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import streamlit as st
from engine import InputValidationError, get_ai_provider, run_analysis

st.set_page_config(page_title="Drug Interaction Advisor", page_icon="⚕️", layout="wide")
RISK_ICONS = {"HIGH_RISK": "🔴", "CAUTION": "⚠️", "NO_CURATED_MATCH": "🔎", "INSUFFICIENT_EVIDENCE": "❓", "UNKNOWN": "❓"}

def split_lines(text: str) -> list[str]:
    """Presentation is plain escaped Streamlit Markdown."""
    return [line for line in text.splitlines() if line.strip()]

def main() -> None:
    st.title("⚕️ Drug Interaction Advisor")
    st.caption("Educational prototype · RxNorm/NLM + OpenFDA + limited local rules")
    st.warning("Not clinically validated prescribing software. Do not use it as a substitute for pharmacist or clinician review.")
    provider = get_ai_provider()
    st.sidebar.subheader("Configuration")
    st.sidebar.info(f"AI provider: {provider}. Local deterministic analysis is the default and requires no LLM key.")
    st.sidebar.success("RxNorm/NLM and OpenFDA use live public REST APIs; availability is reported in the result.")
    with st.form("analysis"):
        age = st.number_input("Patient age", min_value=1, max_value=120, value=65, step=1)
        conditions_raw = st.text_area("Medical conditions (one per line)", placeholder="CKD Stage 3")
        allergies_raw = st.text_area("Known allergies (one per line)", placeholder="Penicillin")
        medications_raw = st.text_area("Current medications (one per line)", placeholder="Warfarin 5 mg")
        new_drug = st.text_input("Proposed new medication", placeholder="Ibuprofen 400 mg")
        submitted = st.form_submit_button("Check evidence")
    if not submitted: return
    conditions = [item.strip() for item in conditions_raw.splitlines() if item.strip()]
    allergies = [item.strip() for item in allergies_raw.splitlines() if item.strip()]
    medications = [item.strip() for item in medications_raw.splitlines() if item.strip()]
    try:
        with st.spinner("Querying RxNorm and OpenFDA, then evaluating local rules…"):
            result = run_analysis(int(age), allergies, conditions, medications, new_drug)
    except (InputValidationError, ValueError) as exc:
        st.error(str(exc)); return
    except Exception:
        st.error("The analysis could not be completed. Please try again later."); return
    risk = result["risk_level"]
    st.subheader(f"{RISK_ICONS.get(risk, '❓')} {risk.replace('_', ' ')}")
    st.info("A local-rule miss is reported as NO CURATED MATCH, never as safe.")
    st.subheader("Evidence report")
    for line in split_lines(result["final_answer"]):
        st.markdown(line)  # Streamlit escapes user and external text by default.
    with st.expander("Execution / tool trace"):
        for step in result["react_steps"]:
            st.write(f"{step['action']} — {step['status']}")
            st.code(str(step["observation"]), language=None)

if __name__ == "__main__": main()
