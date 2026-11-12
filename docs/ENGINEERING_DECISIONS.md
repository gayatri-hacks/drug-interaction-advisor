# Engineering decisions

## Audit findings and fixes

| Finding | Fix |
| --- | --- |
| UI blocked all use when `OPENAI_API_KEY` was absent. | Added `AI_PROVIDER=local` default and lazy optional-provider initialization. |
| A local database miss produced `SAFE`. | Added conservative structured risk outcomes; a miss is `NO_CURATED_MATCH` or `INSUFFICIENT_EVIDENCE`. |
| API failures were swallowed and had inconsistent timeout behavior. | Centralized request handling with timeout, bounded transient retries, status classification, and RxNorm cache. |
| FDA boxed warnings were escalated automatically. | They are now evidence for review only; deterministic patient/rule evidence controls the outcome. |
| UI inserted external/user content into custom HTML and showed "AI Thinking Process." | The UI uses escaped Streamlit text components and labels the display as an execution/tool trace. |
| LLM/LangChain imports were mandatory and agent flow was not reproducible. | The local deterministic pipeline is the primary path; clients are imported only for explicitly selected providers. |

## Tradeoffs

The rule set is intentionally small and transparent. This is preferable to implying complete clinical coverage. The project retains real NLM and FDA services rather than fabricating responses; tests mock HTTP at the boundary for repeatability.

Retries apply only to likely transient transport and HTTP conditions (429/5xx) and are bounded to two retries. Successful RxNorm responses use a one-hour in-process TTL; failures are deliberately not cached so a later analysis can retry a recovered service.

## Curated-rule review

The original repository contained 25 interaction entries. Eight ingredient-level pairs remain: warfarin/aspirin, warfarin/ibuprofen, sertraline/tramadol, fluoxetine/tramadol, sildenafil/isosorbide, azithromycin/hydroxychloroquine, metformin/ibuprofen, and lisinopril/ibuprofen. Their wording is limited to neutral risk mechanisms; legacy dosing, substitution, monitoring, and prescribing instructions were removed because this educational project has no maintained clinical-citation or patient-specific dosing context.

Seventeen entries were intentionally removed: class placeholders (`ssri`/`maoi`, `nitrate`), a procedure rather than a medication (`contrast`), and pairs whose legacy text made unsupported dose, timing, or therapeutic recommendations. A missing rule is `NO_CURATED_MATCH`, not safe. The matcher now accepts only exact, distinct RxNorm active ingredients and never uses substring matching.

## Optional FAISS knowledge base

`build_knowledge_base.py` remains a standalone index-builder/search utility after installing `requirements-kb.txt` and allowing its sentence-transformer model download. It is not imported by `engine.py` or `tools.py` and does not contribute to runtime risk classification.
