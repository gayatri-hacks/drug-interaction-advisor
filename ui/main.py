"""
ui/main.py — Drug Interaction Advisor Dashboard
================================================
Run: streamlit run ui/main.py

Shows:
  • Patient profile input form
  • Color-coded risk banner
  • API pipeline status (RxNorm → Local DB → OpenFDA)
  • Structured safety report with per-section color coding
  • Expandable ReAct thinking process trace
"""

import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import streamlit as st
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drug Interaction Advisor",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,400;0,600;1,400&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, .stApp { background:#05080f; color:#c8d8f0; font-family:'Inter',sans-serif; }
h1,h2,h3,h4       { font-family:'Inter',sans-serif; color:#e0eaff; }

/* Risk banners */
.risk-SAFE              {background:#001f0a;border:2px solid #22c55e;color:#22c55e;}
.risk-LOW\ RISK         {background:#001f0a;border:2px solid #22c55e;color:#22c55e;}
.risk-USE\ WITH\ CAUTION{background:#1c1000;border:2px solid #f59e0b;color:#f59e0b;}
.risk-NOT\ RECOMMENDED  {background:#1c0000;border:2px solid #ef4444;color:#ef4444;}
.risk-HIGH\ RISK        {background:#1c0000;border:2px solid #ef4444;color:#ef4444;}
.risk-CONTRAINDICATED   {background:#1a0025;border:2px solid #a855f7;color:#a855f7;}
.risk-UNKNOWN           {background:#111827;border:2px solid #6b7280;color:#9ca3af;}
.risk-banner            {padding:18px 24px;border-radius:12px;text-align:center;
                         font-size:22px;font-weight:700;letter-spacing:.04em;margin:12px 0;}

/* Pipeline steps */
.pipe-step {display:inline-flex;align-items:center;gap:8px;
            background:#0a1628;border:1px solid #1e3050;
            border-radius:8px;padding:8px 14px;font-size:12px;
            font-family:'JetBrains Mono',monospace;}
.pipe-ok   {color:#22c55e;} .pipe-warn {color:#f59e0b;} .pipe-err {color:#ef4444;}

/* Section cards */
.section-card {border-left:3px solid;padding:10px 16px;margin:10px 0;
               background:#0a1220;border-radius:0 8px 8px 0;}
.sc-label     {font-size:10px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
               margin-bottom:6px;font-family:'JetBrains Mono',monospace;}
.sc-body      {color:#c8d8f0;line-height:1.8;font-size:14px;}

/* ReAct trace */
.trace-thought {border-left:3px solid #60a5fa;background:#0a1628;padding:10px 14px;margin:6px 0;border-radius:0 8px 8px 0;}
.trace-action  {border-left:3px solid #c084fc;background:#14082a;padding:10px 14px;margin:6px 0;border-radius:0 8px 8px 0;}
.trace-obs     {border-left:3px solid #34d399;background:#051a10;padding:10px 14px;margin:6px 0;border-radius:0 8px 8px 0;}
.trace-label   {font-size:10px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;
                font-family:'JetBrains Mono',monospace;margin-bottom:4px;}
.trace-body    {font-size:12px;color:#c8d8f0;line-height:1.6;font-family:'JetBrains Mono',monospace;white-space:pre-wrap;}

/* RxNorm tag */
.rxnorm-tag {display:inline-block;background:#0f2040;border:1px solid #3b82f6;color:#60a5fa;
             border-radius:20px;padding:2px 10px;font-size:11px;font-family:'JetBrains Mono',monospace;margin:2px;}

/* Inputs */
.stTextInput>div>div>input,.stTextArea textarea,.stSelectbox>div>div {
    background:#0a1220!important;border:1px solid #1e3050!important;
    color:#c8d8f0!important;border-radius:8px!important;}
.stSlider>div {color:#c8d8f0;}
section[data-testid="stSidebar"] {background:#080e1a;border-right:1px solid #1a2a4a;}
div[data-testid="stExpander"]    {background:#0a1220;border:1px solid #1a2a4a;border-radius:10px;}

/* Run button */
.stButton>button {
    background:linear-gradient(135deg,#1e3a6e,#2563eb);
    border:2px solid #3b82f6;border-radius:10px;
    color:#93c5fd;font-family:'Inter',sans-serif;
    font-weight:600;font-size:14px;letter-spacing:.04em;
    text-transform:uppercase;padding:12px 24px;width:100%;}
.stButton>button:hover {border-color:#60a5fa;color:#dbeafe;}
</style>
""", unsafe_allow_html=True)


# ── Presets ───────────────────────────────────────────────────────────────────
PRESETS = {
    "👴 Elderly Diabetic + Advil":    dict(age=80, cond=["Type 2 Diabetes","CKD Stage 3"], allerg=[], meds=["Metformin 500mg","Lisinopril 10mg"], new="Advil 400mg"),
    "💊 Warfarin + Aspirin":          dict(age=65, cond=["Atrial Fibrillation"], allerg=[], meds=["Warfarin 5mg"], new="Aspirin 325mg"),
    "🧬 Penicillin → Amoxicillin":    dict(age=35, cond=["Respiratory Infection"], allerg=["Penicillin"], meds=[], new="Amoxicillin 500mg"),
    "⚡ Serotonin Syndrome":           dict(age=45, cond=["Depression","Chronic Back Pain"], allerg=["Codeine"], meds=["Sertraline 100mg"], new="Tramadol 50mg"),
    "💓 QT Prolongation":             dict(age=52, cond=["Atrial Fibrillation","Pneumonia"], allerg=[], meds=["Hydroxychloroquine 200mg"], new="Azithromycin 500mg"),
    "🫀 Nitrate + Sildenafil":        dict(age=62, cond=["Coronary Artery Disease"], allerg=[], meds=["Isosorbide Mononitrate 20mg"], new="Sildenafil 50mg"),
}

RISK_ICONS = {
    "SAFE":"✅","LOW RISK":"✅","USE WITH CAUTION":"⚠️",
    "NOT RECOMMENDED":"🚫","HIGH RISK":"🔴","CONTRAINDICATED":"⛔","UNKNOWN":"❓",
}

SECTION_COLORS = {
    "OVERALL RISK:":         "#fbbf24",
    "DRUG INTERACTIONS:":    "#f87171",
    "PATIENT-SPECIFIC RISKS:":"#c084fc",
    "FDA LABEL DATA:":       "#60a5fa",
    "CLINICAL GUIDANCE:":    "#34d399",
    "TRACEABILITY:":         "#94a3b8",
}


def parse_final_answer(text: str) -> list[tuple[str, str, str]]:
    """Split the final answer into labeled sections."""
    keys   = list(SECTION_COLORS.keys())
    parsed = []
    for i, key in enumerate(keys):
        idx = text.find(key)
        if idx == -1:
            continue
        end = len(text)
        for nk in keys[i+1:]:
            ni = text.find(nk, idx + len(key))
            if ni != -1 and ni < end:
                end = ni
        content = text[idx+len(key):end].strip()
        parsed.append((key.rstrip(":"), SECTION_COLORS[key], content))
    return parsed


def render_api_pipeline(rxnorm_new: dict | None, steps: list[dict]):
    """Show which APIs were called in the agent run."""
    tools_used = {s["action"] for s in steps if s.get("action")}

    def badge(label: str, ok: bool, detail: str = ""):
        cls = "pipe-ok" if ok else "pipe-warn"
        tick = "✓" if ok else "–"
        return (f'<span class="pipe-step"><span class="{cls}">{tick}</span> '
                f'<strong>{label}</strong>'
                + (f'<span style="color:#4a6080;"> → {detail}</span>' if detail else "")
                + '</span>')

    rxnorm_ok = bool(rxnorm_new and rxnorm_new.get("normalized"))
    rxnorm_det = rxnorm_new["generic_name"] if rxnorm_ok else "network unavailable"

    local_ok = "check_local_interactions" in tools_used
    fda_ok   = "openfda_lookup" in tools_used
    risk_ok  = "assess_patient_risk" in tools_used

    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;gap:8px;margin:12px 0;">'
        + badge("RxNorm / NLM", rxnorm_ok, rxnorm_det)
        + badge("Local Interaction DB", local_ok)
        + badge("OpenFDA Label API", fda_ok)
        + badge("Patient Risk (Beers + Allergy)", risk_ok)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_rxnorm_tags(resolutions: list[dict]):
    """Show brand → generic resolution tags."""
    if not resolutions:
        return
    tags = ""
    for r in resolutions:
        if r.get("normalized") and r["brand_name"].lower() != r["generic_name"]:
            tags += f'<span class="rxnorm-tag">{r["brand_name"]} → {r["generic_name"]}</span> '
    if tags:
        st.markdown(
            f'<div style="margin:8px 0;"><span style="font-size:11px;color:#4a6080;'
            f'font-family:JetBrains Mono,monospace;">RxNorm resolved: </span>{tags}</div>',
            unsafe_allow_html=True,
        )


# ── App ───────────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div style="display:flex;align-items:center;gap:14px;padding:4px 0 20px;">
        <div style="width:44px;height:44px;background:linear-gradient(135deg,#1e3a6e,#2563eb);
             border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:24px;">⚕</div>
        <div>
            <div style="font-size:22px;font-weight:700;color:#e0eaff;letter-spacing:-.02em;">Drug Interaction Advisor</div>
            <div style="font-size:11px;color:#3a6090;letter-spacing:.06em;font-family:'JetBrains Mono',monospace;">
                GPT-4o · LANGCHAIN REACT · RXNORM (NLM) · OPENFDA · FAISS BEERS CRITERIA
            </div>
        </div>
        <div style="margin-left:auto;text-align:right;">
            <div style="font-size:11px;color:#3a5070;">API Sandwich Architecture</div>
            <div style="font-size:10px;color:#2a3a50;font-family:'JetBrains Mono',monospace;">
                RxNorm → Local DB → OpenFDA → GPT-4o
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.warning("⚠️ **Clinical decision support prototype only.** Not a substitute for pharmacist review.")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ API Status")
        provider = st.selectbox("Analysis mode", ["local", "openai", "groq"], index=0,
                                help="Local is keyless and uses transparent rules. AI modes add a supplemental summary only.")
        if provider == "local":
            st.info("ℹ️ Local-rule mode — no paid AI key required")
        elif provider == "openai":
            st.info("OpenAI selected. A missing key will be reported in the result.")
        else:
            st.info("Groq selected. A missing key will be reported in the result.")

        fda_key = os.getenv("OPENFDA_API_KEY", "")
        if fda_key and fda_key not in ("your-openfda-key-here", ""):
            st.success("✅ OpenFDA (keyed — 240 req/min)")
        else:
            st.info("ℹ️ OpenFDA (public — 40 req/min)")

        st.success("✅ RxNorm / NLM (free, no key)")

        st.divider()
        st.markdown("### 🧪 Test Cases")
        for name, data in PRESETS.items():
            if st.button(name, key=f"p_{name}"):
                st.session_state.update({
                    "s_age":  data["age"],
                    "s_cond": "\n".join(data["cond"]),
                    "s_allg": "\n".join(data["allerg"]),
                    "s_meds": "\n".join(data["meds"]),
                    "s_new":  data["new"],
                })
                st.session_state.pop("result", None)
                st.rerun()

        st.divider()
        st.markdown("### 🏗️ API Sandwich")
        st.markdown("""
**1. RxNorm (NLM)**  
Resolves "Advil" → "ibuprofen"  
*Free, no key needed*

**2. Local Interaction DB**  
25+ curated high-risk pairs  
*Instant, offline*

**3. OpenFDA**  
Official FDA drug label  
*Free, optional key*

**4. Optional AI summary**  
OpenAI or Groq can summarize retrieved evidence.  
*It cannot override local high-risk warnings.*
        """)

    # ── Input Form ────────────────────────────────────────────────────────────
    col_in, col_out = st.columns([1, 1], gap="large")

    with col_in:
        st.markdown("### 👤 Patient Profile")

        age = st.slider("Patient Age", 1, 100,
                        value=st.session_state.get("s_age", 65))
        if age >= 65:
            st.info("⚠️ Beers Criteria applies (patient ≥65)")

        cond_raw = st.text_area(
            "Medical Conditions *(one per line)*",
            value=st.session_state.get("s_cond", "Type 2 Diabetes\nCKD Stage 3"),
            height=90, placeholder="Type 2 Diabetes\nHypertension\n...",
        )
        allg_raw = st.text_area(
            "Known Allergies *(one per line)*",
            value=st.session_state.get("s_allg", ""),
            height=60, placeholder="Penicillin\nSulfa\n...",
        )

        st.markdown("### 💊 Medications")
        meds_raw = st.text_area(
            "Current Medications *(one per line)*",
            value=st.session_state.get("s_meds", "Metformin 500mg\nLisinopril 10mg"),
            height=90, placeholder="Metformin 500mg\nWarfarin 5mg\n...",
        )
        new_drug = st.text_input(
            "🆕 Proposed New Drug",
            value=st.session_state.get("s_new", "Advil 400mg"),
            placeholder='e.g. "Advil 400mg" or "ibuprofen 400mg"',
        )
        st.caption('💡 You can use brand names — RxNorm resolves them automatically.')

        run = st.button("▶  CHECK DRUG SAFETY", use_container_width=True)

    # Parse inputs
    conditions = [c.strip() for c in cond_raw.splitlines() if c.strip()]
    allergies  = [a.strip() for a in allg_raw.splitlines() if a.strip()]
    cur_meds   = [m.strip() for m in meds_raw.splitlines() if m.strip()]

    # ── Results ───────────────────────────────────────────────────────────────
    with col_out:
        if not run and "result" not in st.session_state:
            st.markdown("""
            <div style="background:#0a1220;border:1px dashed #1a2a4a;border-radius:14px;
                        padding:48px;text-align:center;color:#2a4a7a;margin-top:40px;">
                <div style="font-size:52px;margin-bottom:16px;">⚕️</div>
                <div style="font-size:15px;color:#3a5a8a;">Fill in the patient profile and click<br>
                    <strong style="color:#5a7ab8;">Check Drug Safety</strong></div>
                <div style="font-size:11px;margin-top:10px;color:#1a3050;">
                    Brand names are auto-resolved via RxNorm
                </div>
            </div>""", unsafe_allow_html=True)

        if run:
            if not new_drug.strip():
                st.error("Please enter a proposed new drug.")
            else:
                st.session_state.pop("result", None)
                with st.spinner("🔄 Running API Sandwich: RxNorm → Local DB → OpenFDA → GPT-4o..."):
                    try:
                        from engine import run_analysis
                        result = run_analysis(
                            age=age, allergies=allergies, conditions=conditions,
                            current_drugs=cur_meds, new_drug=new_drug, provider=provider,
                        )
                        st.session_state["result"] = result
                    except Exception as e:
                        st.error(f"Agent error: {e}")
                        st.session_state.pop("result", None)

        if "result" in st.session_state:
            res  = st.session_state["result"]
            risk = res.get("risk_level", "UNKNOWN")
            icon = RISK_ICONS.get(risk, "❓")
            cls  = f"risk-{risk}"

            # Risk banner
            st.markdown(
                f'<div class="risk-banner {cls}">{icon}  {risk}</div>',
                unsafe_allow_html=True,
            )

            # API pipeline status
            st.markdown("#### 🔌 API Pipeline")
            render_api_pipeline(res.get("rxnorm_new"), res.get("react_steps", []))

            # RxNorm resolution tags
            all_resolutions = []
            for step in res.get("react_steps", []):
                # extract from trace if available
                pass
            if res.get("rxnorm_new"):
                render_rxnorm_tags([res["rxnorm_new"]])

            st.markdown("---")

            # ── Safety Report ─────────────────────────────────────────────────
            st.markdown("### 📋 Safety Report")
            final = res.get("final_answer", "")
            sections = parse_final_answer(final)

            if sections:
                for label, color, content in sections:
                    st.markdown(
                        f'<div class="section-card" style="border-color:{color};">'
                        f'<div class="sc-label" style="color:{color};">{label}</div>'
                        f'<div class="sc-body">{content}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    f'<div style="background:#0a1220;border:1px solid #1a2a4a;border-radius:10px;'
                    f'padding:16px;color:#c8d8f0;font-size:14px;line-height:1.8;">{final}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # ── Evidence / tool summaries ─────────────────────────────────────
            with st.expander("🔎 Evidence and tool summaries", expanded=False):
                st.caption("Retrieved source summaries and rule matches. Internal model reasoning is not displayed.")
                steps = res.get("react_steps", [])
                if steps:
                    for i, step in enumerate(steps, 1):
                        st.markdown(
                            f'<div class="trace-action">'
                            f'<div class="trace-label" style="color:#c084fc;">⚡ Tool → {step.get("action","")}</div>'
                            f'<div class="trace-body">{step.get("action_input","")[:300]}</div>'
                            f'</div>'
                            f'<div class="trace-obs">'
                            f'<div class="trace-label" style="color:#34d399;">👁 Observation</div>'
                            f'<div class="trace-body">{step.get("observation","")[:500]}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No step trace available.")

            # ── Tools panel ───────────────────────────────────────────────────
            with st.expander("🔧 Tools & APIs Called", expanded=False):
                api_info = {
                    "check_local_interactions": ("🔍 Local Interaction DB + RxNorm",  "25+ curated pairs · RxNorm normalization · No network required for DB lookup",  "#60a5fa"),
                    "assess_patient_risk":       ("👤 Patient Risk (FAISS + Allergy)", "Beers Criteria via FAISS · Allergy cross-reactivity map · Condition rules",       "#c084fc"),
                    "openfda_lookup":            ("🏛 OpenFDA Drug Label API",          "open.fda.gov · Official FDA labels · Adverse events · Black box warnings",        "#34d399"),
                }
                tools_used = {s["action"] for s in res.get("react_steps", []) if s.get("action")}
                for tool_name, (title, desc, color) in api_info.items():
                    used = tool_name in tools_used
                    st.markdown(
                        f'<div style="background:#0a1220;border:1px solid {"#1a2a4a" if not used else color+"55"};'
                        f'border-radius:8px;padding:10px 14px;margin:6px 0;">'
                        f'<strong style="color:{color if used else "#4a6080"};">{title}</strong>'
                        f'{"" if not used else " <span style=\"color:#22c55e;font-size:11px;\">✓ called</span>"}'
                        f'<br><span style="color:#3a5070;font-size:12px;font-family:JetBrains Mono,monospace;">{desc}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            st.caption(
                "Powered by GPT-4o · LangChain ReAct · RxNorm NLM API · OpenFDA · "
                "FAISS Beers Criteria · Not for clinical use without pharmacist verification"
            )


if __name__ == "__main__":
    main()
