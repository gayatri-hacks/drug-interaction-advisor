import requests
import tools


class Response:
    def __init__(self, code, data=None, url="https://evidence.test"):
        self.status_code, self._data, self.url = code, data or {}, url
    def json(self): return self._data
    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


def test_normalizes_and_matches_curated_pair(monkeypatch):
    monkeypatch.setattr(tools, "normalize_drug_list", lambda drugs: (["warfarin", "aspirin"], [{"brand_name": x, "generic_name": x} for x in drugs]))
    result = tools.check_local_interactions(["warfarin", "aspirin"])
    assert result["count"] == 1
    assert result["interactions"][0]["severity"] == "HIGH"


def test_allergy_and_condition_risk(monkeypatch):
    monkeypatch.setattr(tools, "normalize_drug_list", lambda drugs: (["amoxicillin", "ibuprofen"], [{"brand_name": x, "generic_name": x} for x in drugs]))
    monkeypatch.setattr("build_knowledge_base.search_guidelines", lambda *a, **k: [])
    result = tools.assess_patient_risk(40, ["penicillin"], ["CKD stage 3"], ["amoxicillin", "ibuprofen"])
    assert any(x["severity"] == "CONTRAINDICATED" for x in result["allergy_conflicts"])
    assert any("CKD" in x for x in result["condition_warnings"])


def test_rxnorm_timeout_is_distinguished(monkeypatch):
    tools._rxnorm_cache.clear()
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.Timeout()))
    assert tools.rxnav_normalize("ibuprofen")["status"] == "timeout"


def test_openfda_rate_limit_is_not_no_record(monkeypatch):
    monkeypatch.setattr(tools, "rxnav_normalize", lambda name: {"generic_name": name, "summary": "rx", "status": "success", "rxcui": "1"})
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: Response(429))
    result = tools.openfda_lookup("ibuprofen")
    assert result["found"] is False
    assert result["status"] == "rate_limited"
    assert "unavailable" in result["summary"]


def test_openfda_success_has_evidence_url(monkeypatch):
    monkeypatch.setattr(tools, "rxnav_normalize", lambda name: {"generic_name": name, "summary": "rx", "status": "success", "rxcui": "1"})
    label = {"results": [{"openfda": {"brand_name": ["Advil"], "generic_name": ["ibuprofen"]}, "warnings": ["warning"]}]}
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: Response(200, label, "https://api.fda.gov/evidence"))
    result = tools.openfda_lookup("ibuprofen")
    assert result["found"] is True
    assert result["status"] == "success"
    assert result["source_url"] == "https://api.fda.gov/evidence"
