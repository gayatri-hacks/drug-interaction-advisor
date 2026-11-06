import pytest
import tools


@pytest.mark.live
def test_live_rxnorm_ibuprofen():
    result = tools.rxnav_normalize("ibuprofen")
    assert result["status"] == "success"
    assert result["rxcui"]


@pytest.mark.live
def test_live_openfda_ibuprofen():
    result = tools.openfda_lookup("ibuprofen")
    assert result["status"] == "success"
    assert result["found"] is True
