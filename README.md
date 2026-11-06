# Drug Interaction Advisor

An educational healthcare research prototype built with Streamlit, LangChain, FAISS, RxNorm, and OpenFDA. It is **not clinically validated**, does not grant prescribing approval, and is not medical advice. A licensed pharmacist or prescriber must review any treatment decision.

## Setup

```bash
git clone https://github.com/gayatri-hacks/drug-interaction-advisor.git
cd drug-interaction-advisor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run ui/main.py
```

The default `LLM_PROVIDER=local` starts without an OpenAI or Groq key. It runs rule-based evidence reporting with public RxNorm and OpenFDA calls. Select `openai` or `groq` in the UI only after setting that provider's key; the selected provider is never silently substituted.

Run tests:

```bash
pytest                         # deterministic, mocked unit tests
pytest -m live                 # opt-in calls to real RxNorm and OpenFDA
```

## Architecture

```text
Streamlit → engine.py → RxNorm / local curated rules / FAISS guidelines / OpenFDA
                         ↓
                evidence-first local report
                         ↓
          optional LangChain OpenAI or Groq summary
```

`tools.py` uses the live [RxNorm REST API](https://lhncbc.nlm.nih.gov/RxNav/APIs/index.html) for drug-name normalization and the live [OpenFDA drug-label API](https://open.fda.gov/apis/drug/label/) for label evidence. OpenFDA works without a key, subject to public rate limits. Each result explicitly distinguishes a record found, no record, timeout, network error, rate limit, and API error, and retains a source URL where available.

## Safety model and limitations

- The local interaction and patient-risk rules are curated, incomplete screening aids. A missing rule is reported as **UNKNOWN**, never SAFE.
- OpenFDA labels and RxNorm resolution are evidence sources, not complete interaction screening or patient-specific recommendations.
- High-risk local findings control the displayed risk level. An optional AI summary is supplemental and cannot downgrade them.
- The UI exposes tool summaries and source evidence, not private model reasoning.
- Some FAISS guidance depends on the committed index; rebuild it with `python build_knowledge_base.py` when changing `data/guidelines.txt`.

## Configuration

```dotenv
LLM_PROVIDER=local             # local | openai | groq
OPENAI_API_KEY=...             # needed only for openai
GROQ_API_KEY=...               # needed only for groq
OPENFDA_API_KEY=...            # optional; public access works without it
```

GitHub Actions installs dependencies and runs the non-live pytest suite on pushes and pull requests.
