import requests
import tools

class Response:
    def __init__(self, status=200, data=None, bad_json=False): self.status_code, self.data, self.bad_json = status, data or {}, bad_json
    def json(self):
        if self.bad_json: raise ValueError("bad json")
        return self.data

def test_rxnorm_success(monkeypatch):
    tools.rxnav_normalize.cache_clear()
    calls = iter([Response(data={"approximateGroup": {"candidate": [{"rxcui": "1"}]}}), Response(data={"properties": {"name": "Ibuprofen", "tty": "IN"}})])
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: next(calls))
    assert tools.rxnav_normalize("Advil 200 mg")["generic_name"] == "ibuprofen"

def test_rxnorm_unknown_and_timeout(monkeypatch):
    tools.rxnav_normalize.cache_clear(); monkeypatch.setattr(tools.requests, "get", lambda *a, **k: Response(data={"approximateGroup": {"candidate": []}}))
    assert tools.rxnav_normalize("not-a-drug")["evidence_status"] == "unresolved"
    tools.rxnav_normalize.cache_clear(); monkeypatch.setattr(tools.requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.exceptions.Timeout()))
    assert tools.rxnav_normalize("x")["evidence_status"] == "timeout"

def test_openfda_success_empty_and_malformed(monkeypatch):
    monkeypatch.setattr(tools, "rxnav_normalize", lambda n: {"generic_name": "drug", "evidence_status": "ok"})
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: Response(data={"results": [{"warnings": ["warning"]}]}))
    assert tools.openfda_lookup("drug")["found"]
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: Response(data={"results": []}))
    assert tools.openfda_lookup("drug")["evidence_status"] == "not_found"
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: Response(bad_json=True))
    assert tools.openfda_lookup("drug")["evidence_status"] == "malformed_response"

def test_rate_limit_and_server_retry_are_bounded(monkeypatch):
    monkeypatch.setattr(tools, "MAX_RETRIES", 2); calls = []
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: calls.append(1) or Response(status=429))
    assert tools._request_json("x")[1] == "rate_limited" and len(calls) == 3
    calls.clear(); monkeypatch.setattr(tools.requests, "get", lambda *a, **k: calls.append(1) or Response(status=503))
    assert tools._request_json("x")[1] == "http_error" and len(calls) == 3

def test_known_interaction(monkeypatch):
    monkeypatch.setattr(tools, "normalize_drug_list", lambda d: (["warfarin", "aspirin"], [{}, {}]))
    assert tools.check_local_interactions(["a", "b"])["interactions"][0]["severity"] == "HIGH"
