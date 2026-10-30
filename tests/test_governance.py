from pathlib import Path
import pandas as pd
from backend.pipeline import heal
from backend.governance import root_cause_groups
ROOT=Path(__file__).resolve().parents[1]
def test_governance_fields_and_nonempty_clean():
    df=pd.read_csv(ROOT/"sample messy datasets/02_measurements_units.csv")
    h=heal(df)
    assert h["cleaned"].shape[0]>0
    assert "validation" in h and "root_causes" in h and "evidence_summary" in h
def test_root_cause_prefers_unit_cause():
    class I:
        def __init__(self,t): self.row=1; self.column="temperature"; self.issue_type=t; self.category="x"; self.rule_id="R"
    assert root_cause_groups([I("Statistical outlier"),I("Unit normalization")],[])[0]["root_cause"]=="Unit inconsistency"
