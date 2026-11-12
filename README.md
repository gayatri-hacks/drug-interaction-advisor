# Drug Interaction Advisor

An educational software prototype that evaluates a proposed medication against a deliberately limited curated rule set, patient-entered context, and live public medication sources. It is not clinically validated prescribing software.

## Quick start (no paid LLM key)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run ui/main.py
```

To rebuild the optional FAISS knowledge base, install `pip install -r requirements-kb.txt` first. It is a standalone retrieval experiment: the runtime analysis path does **not** query this store or use it to determine risk.

`AI_PROVIDER=local` is the default. It makes no LLM request and requires no LLM API key. RxNorm/NLM and OpenFDA are still called as real public REST integrations.

Optional providers are selected explicitly in `.env`:

```bash
AI_PROVIDER=openai
OPENAI_API_KEY=...
# or AI_PROVIDER=groq and GROQ_API_KEY=...
```

The deterministic assessment remains authoritative; optional-provider configuration never changes a risk flag.

## Architecture

```text
Streamlit presentation (ui/main.py)
  -> orchestration and input validation (engine.py)
     -> RxNorm/NLM normalization + OpenFDA labels (tools.py)
     -> deterministic curated interaction/patient rules (tools.py)
     -> evidence-limited report
```

RxNorm is used to normalize medication names where its service responds. OpenFDA supplies FDA label data for review. A lookup failure, empty label, or unresolved medication is reported as missing/insufficient evidence; it is not converted to a negative clinical finding.

## Risk outcomes

| Outcome | Meaning |
| --- | --- |
| `HIGH_RISK` | A high-severity curated interaction, allergy, or patient-specific rule matched. |
| `CAUTION` | A lower-severity curated rule matched. |
| `NO_CURATED_MATCH` | No matching local rule was found while a label was available; not a safety claim. |
| `INSUFFICIENT_EVIDENCE` | Required upstream label evidence was unavailable or unreliable. |
| `UNKNOWN` | Reserved for unexpected states. |

FDA boxed warnings are displayed as label evidence, not automatically treated as a patient-specific contraindication.

Example: warfarin plus aspirin produces `HIGH_RISK` from the curated interaction rule. An unknown pair with a retrieved label produces `NO_CURATED_MATCH`, not “safe.”

## Testing and quality checks

```bash
python -m compileall -q engine.py tools.py ui build_knowledge_base.py
python -m pytest
RUN_LIVE_API_TESTS=1 python -m pytest tests/test_live_apis.py
```

Unit tests mock all HTTP calls. Live tests are opt-in and skipped by default because public API availability is external to the test suite. CI performs dependency installation, compilation, unit tests, and a no-key local smoke test.

## Limitations and error behavior

- Curated rules are illustrative and incomplete; they cannot establish medication safety.
- Public API calls use an 8-second timeout and two bounded retries for transient failures. Successful RxNorm resolutions use a one-hour in-process TTL; transient failures are never cached.
- HTTP 404, 429, 5xx, malformed JSON, timeouts, empty labels, and unresolved names are surfaced in the report’s execution trace without stack traces or credentials.
- The optional `build_knowledge_base.py` rebuild requires its listed FAISS/sentence-transformer dependencies and may download its embedding model. It produces a searchable local index, but is not connected to deterministic runtime assessment.

See [engineering decisions](docs/ENGINEERING_DECISIONS.md) and the [interview walkthrough](docs/INTERVIEW_WALKTHROUGH.md).
