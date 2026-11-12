import os
import pytest
from tools import openfda_lookup, rxnav_normalize

pytestmark = pytest.mark.skipif(os.getenv("RUN_LIVE_API_TESTS") != "1", reason="Set RUN_LIVE_API_TESTS=1 to use live public APIs.")

def test_live_rxnorm_and_openfda():
    assert rxnav_normalize("ibuprofen")["generic_name"]
    assert "evidence_status" in openfda_lookup("ibuprofen")
