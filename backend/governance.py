from collections import defaultdict

def root_cause_groups(before_issues, repair_log):
    priority={
        "Unit normalization":"Unit inconsistency","Unrecognized measurement value":"Unit inconsistency",
        "Documented missing sentinel":"Missingness sentinel","City-country semantic inconsistency":"Geographic semantic inconsistency",
        "Text geographic contradiction":"Geographic semantic inconsistency","Cross-sentence semantic contradiction":"Cross-record semantic inconsistency",
        "Contextual spelling error":"Text lexical inconsistency","Duplicate record":"Duplicate data","Duplicate text line":"Duplicate text",
        "Invalid date":"Date validity","Invalid time":"Time validity","Non-numeric value in numeric field":"Type inconsistency",
        "Non-numeric sensor value":"Sensor type inconsistency"}
    by=defaultdict(list)
    for i in before_issues: by[(i.row,str(i.column))].append(i)
    out=[]
    for (row,col),items in by.items():
        types=[i.issue_type for i in items]
        cause=next((priority[t] for t in types if t in priority),items[0].category)
        out.append({"row":row,"column":col,"root_cause":cause,"observed_symptoms":", ".join(types),"rule_evidence":[i.rule_id for i in items],"issue_count":len(items)})
    return out

def enrich_repair_log(repair_log, after_issues):
    error_cells={(i.row,str(i.column)) for i in after_issues if i.severity=="Error"}
    out=[]
    for e in repair_log:
        x=dict(e)
        if str(x.get("status","")).startswith("Automatically"):
            x["validation_status"]="FAILED" if (x.get("row"),str(x.get("column"))) in error_cells else "PASSED"
        out.append(x)
    return out

def validation_summary(df_before,df_after,before,after,log):
    be=sum(i.severity=="Error" for i in before); ae=sum(i.severity=="Error" for i in after)
    auto=[e for e in log if str(e.get("status","")).startswith("Automatically")]
    failed=[e for e in auto if e.get("validation_status")=="FAILED"]
    return {"before_error_findings":be,"after_error_findings":ae,"error_reduction":be-ae,"automatic_repairs":len(auto),"unsafe_repairs":len(failed),"validation_passed":ae<=be and not failed,"rows_before":len(df_before),"rows_after":len(df_after),"columns_before":len(df_before.columns),"columns_after":len(df_after.columns)}

def evidence_score_summary(issues):
    vals=[float(i.confidence) for i in issues]
    return {"findings":len(vals),"mean":round(sum(vals)/len(vals),3) if vals else None,"min":round(min(vals),3) if vals else None,"max":round(max(vals),3) if vals else None}
