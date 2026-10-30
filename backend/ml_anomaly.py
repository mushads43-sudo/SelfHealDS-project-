
"""Machine-learning anomaly detection layer for SelfHealDS.

Isolation Forest is used only for statistical anomaly detection. It does not
overwrite values and never makes a repair decision. The deterministic rule
engine remains responsible for domain, format, semantic and unit validity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_NAME = "Isolation Forest"
MODEL_VERSION = "sklearn IsolationForest (project configuration v1)"
RANDOM_STATE = 42
CONTAMINATION = "auto"
MIN_ROWS = 20
MIN_NUMERIC_COLUMNS = 1

def detect_ml_anomalies(df: pd.DataFrame):
    """Return row-level ML anomaly records.

    Only sufficiently populated numeric columns are used. Missing/non-finite
    values are median-imputed for model input; the original dataframe is never
    changed. Results are evidence flags, not proof of erroneous data.
    """
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        return [], {"status":"unavailable","reason":"scikit-learn is not installed"}

    numeric = df.select_dtypes(include=[np.number]).copy()
    if len(df) < MIN_ROWS or numeric.shape[1] < MIN_NUMERIC_COLUMNS:
        return [], {
            "status":"skipped",
            "reason": f"Requires at least {MIN_ROWS} rows and one numeric column.",
            "model": MODEL_NAME, "random_state": RANDOM_STATE,
        }

    usable=[]
    for c in numeric.columns:
        s=numeric[c].replace([np.inf,-np.inf],np.nan)
        if s.notna().sum() >= max(5, int(len(df)*0.5)) and s.nunique(dropna=True)>1:
            usable.append(c)
    if not usable:
        return [], {
            "status":"skipped",
            "reason":"No numeric column has sufficient finite variation.",
            "model": MODEL_NAME, "random_state": RANDOM_STATE,
        }

    X=numeric[usable].replace([np.inf,-np.inf],np.nan)
    X=X.fillna(X.median()).fillna(0.0)

    # Deterministic model configuration for reproducible research runs.
    model=IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    labels=model.fit_predict(X)
    scores=model.decision_function(X)

    anomalies=[]
    for pos,label in enumerate(labels):
        if label != -1:
            continue
        row=df.index[pos]
        # Attach the strongest contributing observed numeric feature as the
        # primary location while retaining the complete feature set as evidence.
        vals=X.iloc[pos]
        med=X.median()
        scale=(X-med).abs().median().replace(0,np.nan)
        deviations=((vals-med).abs()/scale).replace([np.inf,-np.inf],np.nan).fillna(0)
        col=str(deviations.idxmax())
        value=df.iloc[pos][col]
        anomalies.append({
            "row":row,
            "column":col,
            "value":value,
            "score":float(scores[pos]),
            "features":usable,
            "model":MODEL_NAME,
        })

    return anomalies, {
        "status":"complete",
        "model":MODEL_NAME,
        "model_version":MODEL_VERSION,
        "random_state":RANDOM_STATE,
        "n_estimators":200,
        "contamination":CONTAMINATION,
        "numeric_features":usable,
        "rows_scored":len(df),
        "anomaly_count":len(anomalies),
        "note":"Anomaly status indicates statistical unusualness, not proof of a data error."
    }
