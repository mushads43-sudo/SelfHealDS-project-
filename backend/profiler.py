import hashlib
from datetime import datetime, timezone
import pandas as pd
from .rules import semantic_type

def dataset_fingerprint(df):
    payload=pd.util.hash_pandas_object(df,index=True).values.tobytes()+"|".join(map(str,df.columns)).encode()
    return hashlib.sha256(payload).hexdigest()[:16]

def profile_dataset(df):
    records=[]
    for c in df.columns:
        s=df[c]
        missing=int(s.isna().sum()+s.astype("string").str.strip().eq("").sum())
        numeric=pd.to_numeric(s,errors="coerce")
        records.append({"column":c,"dtype":str(s.dtype),"semantic_type":semantic_type(c),"rows":len(s),
            "missing":missing,"missing_rate_pct":round(missing/max(1,len(s))*100,2),
            "unique":int(s.nunique(dropna=True)),"unique_rate_pct":round(s.nunique(dropna=True)/max(1,len(s))*100,2),
            "numeric_compatibility_pct":round(float(numeric.notna().mean())*100,2) if len(s) else 0})
    return pd.DataFrame(records)

def dataset_summary(df):
    # Whitespace-only strings count as missing here too, matching
    # profile_dataset's per-column definition. Previously this used a plain
    # isna() count, so the dataset-level total could silently disagree with
    # the sum of the per-column missing counts shown on the Data Profile page.
    missing_cells=sum(
        int(df[c].isna().sum()+df[c].astype("string").str.strip().eq("").sum())
        for c in df.columns
    )
    return {"rows":int(df.shape[0]),"columns":int(df.shape[1]),"cells":int(df.shape[0]*df.shape[1]),
            "duplicates":int(df.duplicated().sum()),"missing_cells":int(missing_cells),
            "fingerprint":dataset_fingerprint(df),"profiled_at":datetime.now(timezone.utc).isoformat()}
