from build_knowledge_base import GUIDELINES, parse_guidelines


def test_optional_knowledge_base_source_parses_without_runtime_dependencies():
    """The optional builder's source data remains usable without FAISS installed."""
    chunks = parse_guidelines(GUIDELINES)
    assert chunks
    assert all({"rule_id", "title", "text", "risk", "category"} <= chunk.keys() for chunk in chunks)
