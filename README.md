# ⚕️ Drug Interaction Advisor
### GPT-4o · LangChain ReAct · OpenFDA · FAISS · Beers Criteria · Streamlit

> Clinical decision support prototype — for educational/development use only.

---

## 📁 Project Structure

```
drug_advisor/
├── data/
│   └── guidelines.txt          # Medical safety rules (source for FAISS)
├── faiss_store/                 # Auto-created by build_knowledge_base.py
│   ├── guidelines.index
│   └── chunks.pkl
├── ui/
│   └── main.py                 # Streamlit dashboard (Sprint 4)
├── build_knowledge_base.py     # Sprint 1: Parse + embed guidelines → FAISS
├── tools.py                    # Sprint 2: openfda_lookup, check_local_interactions, assess_patient_risk
├── engine.py                   # Sprint 3: LangChain ReAct AgentExecutor
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API keys
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Build the FAISS knowledge base (run once)
```bash
python build_knowledge_base.py
```

### 4. Launch the dashboard
```bash
streamlit run ui/main.py
```

Opens at `http://localhost:8501`

---

## 🧠 Architecture

```
Doctor inputs patient profile + new drug
         │
         ▼
    Streamlit UI (ui/main.py)
         │
         ▼
  engine.py — GPT-4o ReAct AgentExecutor
         │
    ┌────┴────────────────────────┐
    │                             │
    ▼                             ▼
check_local_interactions    assess_patient_risk
(HIGH_RISK_PAIRS dict)      (Allergy map + FAISS
                             Beers Criteria +
                             condition rules)
    │                             │
    └────────────┬────────────────┘
                 │
                 ▼
          openfda_lookup
        (OpenFDA REST API)
                 │
                 ▼
       GPT-4o synthesizes
       Final Safety Verdict
```

### ReAct Loop in Terminal
```
Thought: I need to check if Ibuprofen interacts with Metformin and Lisinopril.
Action: check_local_interactions
Action Input: ["Metformin 500mg", "Lisinopril 10mg", "Ibuprofen 400mg"]
Observation: [MODERATE] metformin + ibuprofen: NSAIDs reduce metformin clearance...
             [MODERATE] lisinopril + ibuprofen: NSAIDs blunt ACE inhibitor effect...

Thought: Patient is 80 years old — must check Beers Criteria and allergies.
Action: assess_patient_risk
Action Input: ["Metformin 500mg", "Lisinopril 10mg", "Ibuprofen 400mg"]
Observation: [BEERS HIGH] ibuprofen in elderly...
             [CONDITION-HIGH] CKD + Ibuprofen: NSAIDs reduce GFR...

Thought: I should check the FDA label for ibuprofen for additional warnings.
Action: openfda_lookup
Action Input: ibuprofen
Observation: OpenFDA: NSAIDs may cause serious cardiovascular thrombotic events...

Thought: I now have all findings needed for a complete safety assessment.
Final Answer: OVERALL RISK: NOT RECOMMENDED...
```

---

## 🔧 Tools (Sprint 2)

| Tool | Source | What it checks |
|------|--------|----------------|
| `check_local_interactions` | Local dict (25+ pairs) | Drug-drug interactions with severity + mechanism |
| `assess_patient_risk` | FAISS + allergy map | Beers Criteria (≥65), allergy conflicts, disease-drug contraindications |
| `openfda_lookup` | OpenFDA REST API | Official FDA adverse reactions + black box warnings |

---

## 🎨 Dashboard Features (Sprint 4)

- **Risk Banner** — color-coded: 🟢 Safe / 🟡 Caution / 🔴 Not Recommended / 🟣 Contraindicated
- **Safety Report** — structured sections: Drug Interactions, Patient Risks, FDA Data, Clinical Guidance
- **AI Thinking Process expander** — full ReAct step-by-step trace (Thought → Action → Observation)
- **Tools Called panel** — shows which tools the agent used
- **6 preset test cases** — one-click loading of clinical edge cases

---

## 🧪 Test Cases

| Case | Expected |
|------|----------|
| Elderly Diabetic + NSAID (80y, CKD + Metformin → Ibuprofen) | NOT RECOMMENDED |
| Warfarin + Aspirin | NOT RECOMMENDED |
| Penicillin allergy + Amoxicillin | CONTRAINDICATED |
| SSRI + Tramadol (serotonin syndrome) | NOT RECOMMENDED |
| QT Prolongation (Hydroxychloroquine + Azithromycin) | NOT RECOMMENDED |
| Nitrate + Sildenafil | CONTRAINDICATED |

---

## 🔌 OpenFDA API

OpenFDA is free and works without a key (limited to 40 requests/minute).
For higher rate limits, register at: https://open.fda.gov/apis/authentication/
Then add to `.env`: `OPENFDA_API_KEY=your-key`

---

## ⚠️ Disclaimer

For educational and development purposes only.
Drug interaction databases are illustrative, not comprehensive.
Always verify with a licensed clinical pharmacist before making prescribing decisions.
