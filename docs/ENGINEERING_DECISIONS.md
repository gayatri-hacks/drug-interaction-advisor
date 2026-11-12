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

Retries apply only to likely transient transport and HTTP conditions (429/5xx) and are bounded to two retries. This avoids an unresponsive UI and avoids masking unavailable sources.
