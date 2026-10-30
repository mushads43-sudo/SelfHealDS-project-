
from pathlib import Path
from datetime import datetime, timezone
import json
from backend.evaluation import evaluate_bundled_suite
from backend.research import run_research_suite, to_jsonable

root=Path(__file__).parent
out=root/"research_outputs"; out.mkdir(exist_ok=True)
timestamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

suite=evaluate_bundled_suite(root/"sample messy datasets")
csv_path=out/f"bundled_suite_{timestamp}.csv"
suite.to_csv(csv_path,index=False)
print(suite.to_string(index=False))
print(f"Saved: {csv_path}")

# Save one complete reproducible research manifest per bundled dataset.
for path in sorted((root/"sample messy datasets").glob("*.csv")):
    import pandas as pd
    df=pd.read_csv(path)
    result=run_research_suite(df)
    manifest_path=out/f"{path.stem}_research_{timestamp}.json"
    # to_jsonable recursively turns embedded DataFrames (sweep, ablation,
    # category_breakdown, bundled_suite, the benchmark's messy dataset) and
    # Issue dataclass instances into plain JSON-safe structures, instead of
    # the previous partial per-key .to_dict() pass that missed nested
    # DataFrames/dataclasses inside "evaluation" and left them to be
    # silently stringified by json.dumps(default=str).
    manifest_path.write_text(json.dumps(to_jsonable(result),indent=2,default=str))
    print(f"Saved: {manifest_path}")
