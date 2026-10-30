from datetime import datetime, timezone
from .profiler import profile_dataset, dataset_summary
from .detector import detect_issues
from .repair import repair_dataset
from .quality import quality_summary, quality_dimensions
from .models import issue_category
from .ml_anomaly import MODEL_NAME, MODEL_VERSION, RANDOM_STATE, detect_ml_anomalies
from .governance import root_cause_groups, enrich_repair_log, validation_summary, evidence_score_summary

def taxonomy_summary(issues):
    counts={}
    for i in issues: counts[i.category]=counts.get(i.category,0)+1
    return counts

def _issue_table(issues):
    return [{"row":i.row,"column":i.column,"issue_type":i.issue_type,"category":i.category,"severity":i.severity,
             "value":i.value,"confidence":i.confidence,"confidence_basis":i.confidence_basis,"repairable":i.repairable,"rule_id":i.rule_id} for i in issues]

def assess(df, context=None):
    started=datetime.now(timezone.utc)
    issues=detect_issues(df, context=context)
    return {"profile":profile_dataset(df),"summary":dataset_summary(df),"issues":issues,
            "issue_table":_issue_table(issues),"quality":quality_summary(df,issues),
            "dimensions":quality_dimensions(df,issues),"taxonomy":taxonomy_summary(issues),
            "run_at":started.isoformat(),"scope":"Current uploaded dataset; pre-repair assessment",
             "ml_model":{"name":MODEL_NAME,"version":MODEL_VERSION,"random_state":RANDOM_STATE,"status":detect_ml_anomalies(df)[1]}}

def heal(df, context=None):
    cleaned,before,after,repair_log,review_df,post=repair_dataset(df, context=context)
    repair_log=enrich_repair_log(repair_log,post)
    return {"cleaned":cleaned,"before_issues":before,"after_issues":after,"post_repair_issues":post,
            "repair_log":repair_log,"review_required":review_df,"quarantined_rows":len(review_df),
            "validation":validation_summary(df,cleaned,before,post,repair_log),
            "root_causes":root_cause_groups(before,repair_log),
            "evidence_summary":evidence_score_summary(before),
            "before_quality":quality_summary(df,before),"after_quality":quality_summary(cleaned,after),
            "before_dimensions":quality_dimensions(df,before),"after_dimensions":quality_dimensions(cleaned,after),
            "run_at":datetime.now(timezone.utc).isoformat(),"scope":"Current uploaded dataset; post-repair validation",
             "ml_model":{"name":MODEL_NAME,"version":MODEL_VERSION,"random_state":RANDOM_STATE,"status":detect_ml_anomalies(df)[1]}}
