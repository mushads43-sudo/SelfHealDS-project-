
import numpy as np
import pandas as pd
from backend.ml_anomaly import detect_ml_anomalies

def test_isolation_forest_is_deterministic():
    rng=np.random.default_rng(123)
    x=rng.normal(size=50)
    x[-1]=12
    df=pd.DataFrame({"value":x})
    a,meta1=detect_ml_anomalies(df)
    b,meta2=detect_ml_anomalies(df)
    assert meta1["status"]=="complete"
    assert meta2["status"]=="complete"
    assert [(z["row"],z["column"]) for z in a] == [(z["row"],z["column"]) for z in b]
    assert meta1["random_state"]==42
    assert meta1["n_estimators"]==200

def test_ml_does_not_modify_input():
    df=pd.DataFrame({"a":[1.,2.,3.,4.,5.]})
    original=df.copy()
    detect_ml_anomalies(df)
    pd.testing.assert_frame_equal(df,original)
