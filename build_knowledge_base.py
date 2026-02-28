"""
Sprint 1, Step 2 — build_knowledge_base.py
==========================================
Converts guidelines.txt into a FAISS vector store.

Run once before starting the app:
    python build_knowledge_base.py

Requires: faiss-cpu, sentence-transformers (or openai embeddings)
"""

import os
import re
import json
import pickle
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATA_DIR   = Path(__file__).parent / "data"
FAISS_DIR  = Path(__file__).parent / "faiss_store"
GUIDELINES = DATA_DIR / "guidelines.txt"


# ── Step 1: Parse guidelines.txt into chunks ──────────────────────────────────
def parse_guidelines(filepath: Path) -> list[dict]:
    """
    Split the guidelines file into meaningful rule chunks.
    Each chunk = one rule block (RULE-E01, RULE-D01, etc.)
    """
    text = filepath.read_text(encoding="utf-8")
    
    # Split on rule headers like "RULE-E01:", "RULE-D01:", "PRINCIPLE-1:"
    pattern = r"((?:RULE-[A-Z]+\d+|PRINCIPLE-\d+):.*?)(?=(?:RULE-[A-Z]+\d+|PRINCIPLE-\d+):|$)"
    raw_chunks = re.findall(pattern, text, re.DOTALL)
    
    chunks = []
    for raw in raw_chunks:
        raw = raw.strip()
        if len(raw) < 50:
            continue
        
        # Extract rule ID and first line as title
        lines = raw.split("\n")
        rule_id = lines[0].split(":")[0].strip()
        title = lines[0].replace(rule_id + ":", "").strip()
        
        # Determine risk level from content
        risk = "MODERATE"
        if "CONTRAINDICATED" in raw:
            risk = "CONTRAINDICATED"
        elif "HIGH" in raw:
            risk = "HIGH"
        elif "LOW" in raw:
            risk = "LOW"
        
        # Determine category
        if rule_id.startswith("RULE-E"):
            category = "elderly"
        elif rule_id.startswith("RULE-D"):
            category = "drug_interaction"
        elif rule_id.startswith("RULE-R"):
            category = "renal"
        elif rule_id.startswith("RULE-A"):
            category = "allergy"
        elif rule_id.startswith("RULE-C"):
            category = "cardiac"
        elif rule_id.startswith("RULE-DB"):
            category = "diabetes"
        else:
            category = "general"
        
        chunks.append({
            "rule_id": rule_id,
            "title": title,
            "text": raw,
            "risk": risk,
            "category": category,
        })
    
    print(f"  Parsed {len(chunks)} rule chunks from {filepath.name}")
    return chunks


# ── Step 2: Build FAISS index ─────────────────────────────────────────────────
def build_faiss_index(chunks: list[dict]):
    """
    Embed each chunk and store in a FAISS index.
    Uses sentence-transformers (all-MiniLM-L6-v2) — no API key needed.
    """
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"Missing package: {e}")
        print("Install with: pip install faiss-cpu sentence-transformers")
        raise

    FAISS_DIR.mkdir(exist_ok=True)
    
    print("  Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    texts = [c["text"] for c in chunks]
    print(f"  Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    embeddings = np.array(embeddings, dtype="float32")
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Build flat inner-product index (cosine similarity after normalization)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    # Save index and metadata
    faiss.write_index(index, str(FAISS_DIR / "guidelines.index"))
    with open(FAISS_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    with open(FAISS_DIR / "meta.json", "w") as f:
        json.dump({"num_chunks": len(chunks), "dim": dim, "model": "all-MiniLM-L6-v2"}, f)
    
    print(f"  ✅ FAISS index saved → {FAISS_DIR}/")
    print(f"     {index.ntotal} vectors, dim={dim}")
    return index


# ── Step 3: Test retrieval ────────────────────────────────────────────────────
def test_retrieval():
    """
    Sprint 1, Step 3 — verify the FAISS index returns correct rules.
    """
    print("\n── Testing FAISS retrieval ───────────────────────────────────────────")
    
    test_queries = [
        "elderly patients and ibuprofen safety",
        "warfarin bleeding risk",
        "serotonin syndrome SSRI combination",
        "metformin renal impairment kidney disease",
        "sildenafil nitrates contraindication",
    ]
    
    for query in test_queries:
        results = search_guidelines(query, top_k=2)
        print(f"\n  Query: '{query}'")
        for r in results:
            print(f"    [{r['risk']}] {r['rule_id']}: {r['title'][:60]}...")
            print(f"           score={r['score']:.3f} | category={r['category']}")


def search_guidelines(query: str, top_k: int = 5) -> list[dict]:
    """
    Search the FAISS index. Returns top_k most relevant guideline chunks.
    Falls back to keyword search if index not built.
    """
    index_path = FAISS_DIR / "guidelines.index"
    chunks_path = FAISS_DIR / "chunks.pkl"
    
    if not index_path.exists():
        print("  ⚠ FAISS index not found — run build_knowledge_base.py first")
        return _keyword_fallback(query)
    
    try:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer("all-MiniLM-L6-v2")
        index = faiss.read_index(str(index_path))
        
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        
        query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vec)
        scores, indices = index.search(query_vec, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                chunk = dict(chunks[idx])
                chunk["score"] = float(score)
                results.append(chunk)
        return results
    
    except Exception as e:
        print(f"  FAISS search error: {e}")
        return _keyword_fallback(query)


def _keyword_fallback(query: str) -> list[dict]:
    """Keyword-based fallback when FAISS is not available."""
    try:
        chunks_path = FAISS_DIR / "chunks.pkl"
        if chunks_path.exists():
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
            words = set(query.lower().split())
            scored = []
            for c in chunks:
                text_words = set(c["text"].lower().split())
                overlap = len(words & text_words)
                if overlap > 0:
                    scored.append((overlap, c))
            scored.sort(key=lambda x: -x[0])
            return [dict(c, score=float(s)) for s, c in scored[:5]]
    except Exception:
        pass
    return []


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Drug Interaction Advisor — Knowledge Base Builder")
    print("  Sprint 1, Step 2-3")
    print("=" * 60)
    
    print("\n── Step 1: Parsing guidelines.txt ───────────────────────────────────")
    chunks = parse_guidelines(GUIDELINES)
    
    print("\n── Step 2: Building FAISS vector store ──────────────────────────────")
    build_faiss_index(chunks)
    
    print("\n── Step 3: Testing retrieval ─────────────────────────────────────────")
    test_retrieval()
    
    print("\n✅ Knowledge base ready. Run: streamlit run ui/main.py")
