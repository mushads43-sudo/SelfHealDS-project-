import pandas as pd
from backend.pipeline import assess, heal

def test_pipeline_roundtrip():
    df=pd.DataFrame({"age":[-25,30],"phone":[" 98765-43210 ","123"],"city":["Mumbai","Pune"],"country":["France","India"]})
    a=assess(df)
    assert len(a["issues"])>0
    h=heal(df)
    assert "cleaned" in h and "review_required" in h
