from backend.models import issue_category

def test_taxonomy_fallback():
    assert issue_category("future rule") == "General data quality"
