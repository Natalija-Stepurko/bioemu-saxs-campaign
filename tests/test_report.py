"""Figure conventions of the report stage."""
from bsc import report


def test_every_model_has_a_colour():
    missing = [m for m in report.MODEL_NAME if m not in report.MODEL_COL]
    assert not missing, missing
    assert all(c.startswith("#") and len(c) == 7 for c in report.MODEL_COL.values())
    # the model under study is the one saturated colour; it is also the page accent
    assert report.MODEL_COL["bioemu"] == "#2F5D8A"
