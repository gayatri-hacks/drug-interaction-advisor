# Interview walkthrough

This project demonstrates a reliability-focused refactor of a small clinical-information prototype.

1. `engine.py` validates inputs and orchestrates providers separately from deterministic risk evaluation and report rendering.
2. `tools.py` is the API boundary. It normalizes names through RxNorm/NLM, retrieves actual OpenFDA labels, classifies failure modes, applies bounded retry/backoff, and caches normalized results in process.
3. The deterministic evaluator returns explicit evidence-limited outcomes. It never turns absence of a limited curated rule into a safety recommendation.
4. `ui/main.py` is presentation-only: it avoids HTML interpolation, gives a no-key local path, and shows the execution trace rather than hidden reasoning.
5. Pytest uses mock response fixtures so retries, malformed payloads, missing labels, and provider selection are reproducible. Live API verification is explicitly opt-in.

When discussing tradeoffs: real public API calls improve realism but introduce external availability concerns, so the result distinguishes source failures from negative evidence. The implementation does not claim clinical validation, production deployment, or performance benchmarks.
