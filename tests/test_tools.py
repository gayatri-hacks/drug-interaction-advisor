import requests
import tools

class Response:
    def __init__(self, status=200, data=None, bad_json=False): self.status_code, self.data, self.bad_json = status, data or {}, bad_json
    def json(self):
        if self.bad_json: raise ValueError("bad json")
        return self.data

def test_rxnorm_success(monkeypatch):
    tools._rxnorm_cache.clear()
    calls = iter([Response(data={"approximateGroup": {"candidate": [{"rxcui": "1"}]}}), Response(data={"properties": {"name": "Ibuprofen", "tty": "IN"}})])
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: next(calls))
    assert tools.rxnav_normalize("Advil 200 mg")["generic_name"] == "ibuprofen"

def test_rxnorm_unknown_and_timeout(monkeypatch):
    tools._rxnorm_cache.clear(); monkeypatch.setattr(tools.requests, "get", lambda *a, **k: Response(data={"approximateGroup": {"candidate": []}}))
    assert tools.rxnav_normalize("not-a-drug")["evidence_status"] == "unresolved"
    tools._rxnorm_cache.clear(); monkeypatch.setattr(tools.requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.exceptions.Timeout()))
    assert tools.rxnav_normalize("x")["evidence_status"] == "timeout"

def test_rxnorm_brand_resolves_related_active_ingredient(monkeypatch):
    tools._rxnorm_cache.clear()
    calls = iter([
        Response(data={"approximateGroup": {"candidate": [{"rxcui": "2"}]}}),
        Response(data={"properties": {"name": "Brand product", "tty": "SBD"}}),
        Response(data={"relatedGroup": {"conceptGroup": [{"conceptProperties": [{"name": "warfarin"}]}]}}),
    ])
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: next(calls))
    result = tools.rxnav_normalize("Brand product")
    assert result["generic_name"] == "warfarin"
    assert result["ingredients"] == ["warfarin"]

def test_transient_rxnorm_failure_is_not_cached(monkeypatch):
    tools._rxnorm_cache.clear(); calls = []
    monkeypatch.setattr(tools, "MAX_RETRIES", 0)
    monkeypatch.setattr(tools.requests, "get", lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(requests.exceptions.Timeout()))
    tools.rxnav_normalize("retry drug")
    tools.rxnav_normalize("retry drug")
    assert len(calls) == 2

def test_successful_rxnorm_cache_expires(monkeypatch):
    tools._rxnorm_cache.clear(); clock = [0]; calls = []
    monkeypatch.setattr(tools, "RXNORM_CACHE_TTL_SECONDS", 10)
    monkeypatch.setattr(tools.time, "monotonic", lambda: clock[0])
    def fake_get(*args, **kwargs):
        calls.append(1)
        return Response(data={"approximateGroup": {"candidate": [{"rxcui": "1"}]}}) if len(calls) % 2 else Response(data={"properties": {"name": "Drug", "tty": "IN"}})
    monkeypatch.setattr(tools.requests, "get", fake_get)
    tools.rxnav_normalize("cache drug")
    tools.rxnav_normalize("cache drug")
    clock[0] = 11
    tools.rxnav_normalize("cache drug")
    assert len(calls) == 4

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

def test_duplicate_medications_do_not_match_two_rule_sides(monkeypatch):
    monkeypatch.setattr(tools, "normalize_drug_list", lambda d: (["warfarin", "warfarin"], [{}, {}]))
    assert tools.check_local_interactions(["warfarin", "warfarin"])["count"] == 0

def test_brand_and_generic_ingredients_match(monkeypatch):
    monkeypatch.setattr(tools, "normalize_drug_list", lambda d: (
        ["brand aspirin", "warfarin"],
        [{"ingredients": ["aspirin"]}, {"ingredients": ["warfarin"]}],
    ))
    assert tools.check_local_interactions(["brand", "warfarin"])["count"] == 1

def test_unrelated_and_substring_names_do_not_match(monkeypatch):
    monkeypatch.setattr(tools, "normalize_drug_list", lambda d: (["aspirin-like", "warfarin-salt"], [{}, {}]))
    assert tools.check_local_interactions(["aspirin-like", "warfarin-salt"])["count"] == 0
    monkeypatch.setattr(tools, "normalize_drug_list", lambda d: (["warfarin", "vitamin c"], [{}, {}]))
    assert tools.check_local_interactions(["warfarin", "vitamin c"])["count"] == 0
