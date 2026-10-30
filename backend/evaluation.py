from datetime import datetime, timezone
import glob, os, random, math, re
import pandas as pd
from .detector import detect_issues
from .repair import repair_dataset
from .profiler import dataset_fingerprint
from .ml_anomaly import detect_ml_anomalies

EVAL_VERSION = "2026.08-eval-v3"
DEFAULT_RATES = (0.05, 0.10, 0.20, 0.30)
DEFAULT_SEEDS = (42, 43, 44, 45, 46)
MIN_RESEARCH_ROWS = 30
MAX_INJECTED_CASES = 5000


def _now(): return datetime.now(timezone.utc).isoformat()
def _set_findings(items): return {(int(i.row), str(i.column), str(i.issue_type)) for i in items}

def _scores(pred, truth):
    tp=len(pred&truth); fp=len(pred-truth); fn=len(truth-pred)
    p=tp/(tp+fp) if tp+fp else None; r=tp/(tp+fn) if tp+fn else None
    f=2*p*r/(p+r) if p is not None and r is not None and p+r else None
    return {"true_positives":tp,"false_positives":fp,"false_negatives":fn,
            "precision_pct":None if p is None else round(p*100,2),
            "recall_pct":None if r is None else round(r*100,2),
            "f1_pct":None if f is None else round(f*100,2),
            "no_positive_predictions":tp+fp==0}

def _has_token(c,tokens):
    # Split camelCase/PascalCase boundaries first (e.g. "PhoneNumber" ->
    # "Phone Number"), matching backend.rules.norm_col's normalization. Without
    # this, camelCase-named real-world columns (PhoneNumber, ProviderNumber,
    # HospitalName, ZipCode, ...) never matched any token below, so the
    # benchmark injector silently produced zero candidate specs -- and the
    # Research Evaluation page showed all-None precision/recall/F1 -- on any
    # dataset that uses camelCase headers instead of snake_case.
    raw=str(c)
    raw=re.sub(r"(?<=[a-z0-9])(?=[A-Z])"," ",raw)
    name=re.sub(r"[^a-z0-9]+","_",raw.lower()).strip("_")
    parts=set(name.split("_")); return any(t in parts or name==t for t in tokens)

def _candidate_specs(df):
    cols=list(df.columns); specs=[]
    age=[c for c in cols if _has_token(c,{"age","years_old","customer_age"}) or "age" in str(c).lower()]
    phone=[c for c in cols if _has_token(c,{"phone","mobile","telephone","contact_number"})]
    email=[c for c in cols if "email" in str(c).lower()]
    qty=[c for c in cols if _has_token(c,{"quantity","qty","count","units"})]
    weight=[c for c in cols if _has_token(c,{"weight","mass"})]
    date=[c for c in cols if _has_token(c,{"date","dob"}) or "date" in str(c).lower()]
    city=[c for c in cols if _has_token(c,{"city","town"})]
    country=[c for c in cols if _has_token(c,{"country","nation"})]
    if age: specs += [{"issue_type":"Negative age","column":age[0],"kind":"negative_age","difficulty":"obvious"},{"issue_type":"Plausible age corruption","column":age[0],"kind":"subtle_age_shift","difficulty":"plausible"}]
    if phone: specs += [{"issue_type":"Invalid phone length","column":phone[0],"kind":"short_phone","difficulty":"obvious"},{"issue_type":"Plausible phone corruption","column":phone[0],"kind":"digit_swap","difficulty":"plausible"}]
    if email: specs += [{"issue_type":"Malformed email","column":email[0],"kind":"bad_email","difficulty":"obvious"},{"issue_type":"Plausible email corruption","column":email[0],"kind":"domain_typo","difficulty":"plausible"}]
    if qty: specs += [{"issue_type":"Negative quantity","column":qty[0],"kind":"negative_quantity","difficulty":"obvious"},{"issue_type":"Plausible quantity corruption","column":qty[0],"kind":"subtle_quantity_shift","difficulty":"plausible"}]
    if weight: specs += [{"issue_type":"Unrecognized measurement value","column":weight[0],"kind":"bad_unit","difficulty":"obvious"},{"issue_type":"Plausible measurement corruption","column":weight[0],"kind":"subtle_measurement_shift","difficulty":"plausible"}]
    if date: specs += [{"issue_type":"Invalid date","column":date[0],"kind":"bad_date","difficulty":"obvious"},{"issue_type":"Plausible date corruption","column":date[0],"kind":"date_shift","difficulty":"plausible"}]
    if city and country: specs.append({"issue_type":"City-country semantic inconsistency","column":country[0],"kind":"geo","difficulty":"obvious"})
    return specs

def _inject(base,row,spec):
    col,kind=spec["column"],spec["kind"]; original=base.at[row,col]
    if pd.isna(original) or str(original).strip()=="": return None
    if kind=="negative_age":
        n=pd.to_numeric(pd.Series([original]),errors="coerce").iloc[0]
        if pd.isna(n) or n==0 or abs(float(n))>120:return None
        value,expected=-abs(float(n)),float(n)
    elif kind=="subtle_age_shift":
        n=pd.to_numeric(pd.Series([original]),errors="coerce").iloc[0]
        if pd.isna(n) or not 1<=float(n)<=119:return None
        value,expected=float(n)+1,float(n)
    elif kind=="short_phone":
        digits=re.sub(r"\D","",str(original))
        if len(digits)<8:return None
        value,expected=digits[:3],None
    elif kind=="digit_swap":
        digits=re.sub(r"\D","",str(original))
        if len(digits)<7 or len(set(digits))<2:return None
        chars=list(digits); j=next(i for i in range(1,len(chars)) if chars[i]!=chars[0]); chars[0],chars[j]=chars[j],chars[0]
        value,expected="".join(chars),str(original)
    elif kind=="bad_email": value,expected="not-an-email",str(original)
    elif kind=="domain_typo":
        raw=str(original).strip()
        if "@" not in raw or "." not in raw.rsplit("@",1)[-1]:return None
        local,domain=raw.rsplit("@",1); domain2=domain.replace("gmail","gmai1",1) if "gmail" in domain else domain.replace(".","1.",1)
        value,expected=f"{local}@{domain2}",raw
    elif kind=="negative_quantity":
        n=pd.to_numeric(pd.Series([original]),errors="coerce").iloc[0]
        if pd.isna(n) or float(n)==0:return None
        value,expected=-abs(float(n)),float(n)
    elif kind=="subtle_quantity_shift":
        n=pd.to_numeric(pd.Series([original]),errors="coerce").iloc[0]
        if pd.isna(n):return None
        value,expected=float(n)+1,float(n)
    elif kind=="bad_unit":
        m=re.match(r"^([-+]?\d+(?:\.\d+)?)",str(original).strip())
        if not m:return None
        value,expected=f"{m.group(1)} stone",str(original)
    elif kind=="subtle_measurement_shift":
        m=re.match(r"^([-+]?\d+(?:\.\d+)?)(.*)$",str(original).strip())
        if not m:return None
        value,expected=f"{float(m.group(1))*1.05:g}{m.group(2)}",str(original)
    elif kind=="bad_date": value,expected="not-a-date",str(original)
    elif kind=="date_shift":
        parsed=pd.to_datetime(pd.Series([original]),errors="coerce",dayfirst=False).iloc[0]
        if pd.isna(parsed):return None
        value,expected=(parsed+pd.Timedelta(days=1)).strftime("%Y-%m-%d"),str(original)
    elif kind=="geo":
        city_cols=[c for c in base.columns if _has_token(c,{"city","town"})]
        if not city_cols:return None
        base.at[row,city_cols[0]]="Mumbai"; value,expected="France","India"
    else:return None
    base.at[row,col]=value
    return {"row":int(row),"column":str(col),"issue_type":spec["issue_type"],"difficulty":spec["difficulty"],"original_value":original,"corrupted_value":value,"expected_repair":expected}

def build_benchmark(df,corruption_rate=0.10,seed=42):
    rng=random.Random(seed); base=df.copy().reset_index(drop=True); specs=_candidate_specs(base)
    if base.empty or not specs:return base,[]
    target=max(1,min(MAX_INJECTED_CASES,math.ceil(len(base)*max(1,len(specs))*float(corruption_rate))))
    pairs=[(r,s) for r in base.index for s in specs]; rng.shuffle(pairs); cases=[]; used=set()
    for row,spec in pairs:
        key=(row,spec["column"])
        if key in used:continue
        trial=base.copy(); case=_inject(trial,row,spec)
        if case is None:continue
        base=trial; cases.append(case); used.add(key)
        if len(cases)>=target:break
    return base,cases

def _truth_set(cases):return {(int(x["row"]),str(x["column"]),str(x["issue_type"])) for x in cases}

def ground_truth_table(cases):
    """Return a display-ready DataFrame of injected benchmark cases.

    Kept as a standalone, testable function so any caller (UI or script)
    gets the same well-formed table instead of re-deriving column names.
    """
    cols=["row","column","issue_type","difficulty","original_value","corrupted_value","expected_repair"]
    if not cases:
        return pd.DataFrame(columns=["Row","Column","Issue type","Difficulty","Original value","Corrupted value","Expected repair"])
    rows=[{
        "Row":c.get("row"),"Column":c.get("column"),"Issue type":c.get("issue_type"),
        "Difficulty":c.get("difficulty"),"Original value":c.get("original_value"),
        "Corrupted value":c.get("corrupted_value"),"Expected repair":c.get("expected_repair"),
    } for c in cases]
    return pd.DataFrame(rows)

def _clean_false_positive_rate(reference, truth, predicted):
    """Rate of spurious detections among cells that were NOT injected with a corruption.

    Denominator is every (row, column) cell in the reference dataset minus the
    cells that were deliberately corrupted (the only cells where a positive
    detection is expected). Numerator is detections in that same negative
    space, i.e. detector output that is not attributable to an injected case
    (matches the false-positive set already used for precision/recall).
    Returns None only when the reference is empty and no rate is defined.
    """
    total_cells = int(reference.shape[0]) * int(reference.shape[1])
    negatives = max(0, total_cells - len(truth))
    if negatives <= 0:
        return None
    fp = len(predicted - truth)
    return round(fp / negatives * 100, 4)

def _repair_results(post_issues,repair_log,cases):
    """Score only benchmark cases with an explicit safe target.

    Cases whose expected repair is None are deliberately non-inferable (for example,
    a three-digit phone created by deleting unknown digits). An automatic repair of
    such a case is unsafe, never correct.
    """
    post_set=_set_findings(post_issues)
    cmap={(int(c["row"]),str(c["column"]),str(c["issue_type"])):c for c in cases}
    attempted=[];correct=[];unsafe=[]
    repairable_cases=[c for c in cases if c.get("expected_repair") is not None]
    repairable_keys={(int(c["row"]),str(c["column"]),str(c["issue_type"])) for c in repairable_cases}
    for event in repair_log:
        key=(event.get("row"),str(event.get("column")),str(event.get("issue_type")))
        if not isinstance(event.get("row"),int) or key not in cmap or not str(event.get("status","")).startswith("Automatically"):
            continue
        attempted.append(event); c=cmap[key]; expected=c.get("expected_repair")
        if expected is None:
            unsafe.append(event)
            continue
        if key in post_set:
            unsafe.append(event)
            continue
        try:
            ok=abs(float(event.get("repaired_value"))-float(expected))<=max(1e-9,abs(float(expected))*1e-9)
        except Exception:
            ok=str(event.get("repaired_value"))==str(expected)
        (correct if ok else unsafe).append(event)
    return attempted,correct,unsafe,repairable_cases

def _benchmark_reference(df):
    return df.copy().reset_index(drop=True),"uploaded_reference_no_preclean",0

def _baseline_for_scoring(original, cases):
    """Return the pre-existing findings set adjusted for cell reuse.

    A benchmark cell chosen for corruption injection can, by chance, already
    carry a pre-existing baseline finding of the SAME issue type (e.g. a
    dataset with an existing city/country mismatch happens to be corrupted at
    that same cell into a different city/country mismatch). Naively
    subtracting the full 'original' findings set would then silently discard
    the post-injection detection as if it were pre-existing noise, undercounting
    recall for a corruption the detector actually caught. Any (row, column)
    that was deliberately corrupted must never be subtracted this way, so the
    resulting detection is scored against the injected ground truth instead of
    disappearing.
    """
    injected_cells = {(int(c["row"]), str(c["column"])) for c in cases}
    return {t for t in original if (t[0], t[1]) not in injected_cells}

def evaluate_detection(df,corruption_rate=0.10,seed=42):
    reference,reference_mode,excluded=_benchmark_reference(df); messy,cases=build_benchmark(reference,corruption_rate,seed); truth=_truth_set(cases)
    original=_set_findings(detect_issues(reference)); detected=detect_issues(messy)
    predicted=_set_findings(detected)-_baseline_for_scoring(original,cases); score=_scores(predicted,truth)
    cleaned,_,remaining,log,review,post=repair_dataset(messy); attempted,correct,unsafe,repairable_cases=_repair_results(post,log,cases)
    repair_accuracy=len(correct)/len(attempted)*100 if attempted else None
    safe_coverage=len(correct)/len(repairable_cases)*100 if repairable_cases else None
    unsafe_rate=len(unsafe)/len(attempted)*100 if attempted else None
    clean_fpr=_clean_false_positive_rate(reference,truth,predicted)
    # Built directly from each case's own key (not by zipping the truth set
    # against the cases list, which pairs elements positionally and silently
    # scrambles the obvious/plausible split because set iteration order has
    # no relationship to the original case list order).
    easy_truth={(int(c["row"]),str(c["column"]),str(c["issue_type"])) for c in cases if c["difficulty"]=="obvious"}
    hard_truth=truth-easy_truth
    easy_scores=_scores(predicted&easy_truth,easy_truth); hard_scores=_scores(predicted&hard_truth,hard_truth)
    return {"generated_at":_now(),"evaluation_version":EVAL_VERSION,"dataset_fingerprint":dataset_fingerprint(df),"benchmark_reference_fingerprint":dataset_fingerprint(reference),"reference_mode":reference_mode,"preexisting_rows_excluded":excluded,"corruption_rate":float(corruption_rate),"seed":int(seed),"actual_corruption_cases":len(cases),"obvious_cases":len(easy_truth),"plausible_cases":len(hard_truth),"messy":messy,"ground_truth":cases,"detected":detected,"new_detected":sorted(predicted),"baseline_findings_before_injection":len(original),**score,"clean_false_positive_rate_pct":clean_fpr,"repair_attempts":len(attempted),"repairable_cases":len(repairable_cases),"repair_correct":len(correct),"repair_unsafe":len(unsafe),"repair_accuracy_pct":None if repair_accuracy is None else round(repair_accuracy,2),"safe_repair_coverage_pct":None if safe_coverage is None else round(safe_coverage,2),"unsafe_repair_rate_pct":None if unsafe_rate is None else round(unsafe_rate,2),"blocked_or_reviewed":max(0,len(cases)-len(attempted)),"quarantined_rows":len(review),"post_repair_findings":len(remaining),"obvious_case_metrics":easy_scores,"plausible_case_metrics":hard_scores,"research_readiness":{"status":"recommended" if len(reference)>=MIN_RESEARCH_ROWS else "small_sample_warning","reference_rows":len(reference),"minimum_recommended_rows":MIN_RESEARCH_ROWS,"reason":f"Below {MIN_RESEARCH_ROWS} rows, percentage-based corruption scenarios remain coarse and should be treated as a demonstration. Publication claims should use larger independently sourced datasets."},"method":"Controlled corruptions are applied directly to the uploaded reference without pre-cleaning by SelfHealDS. Each case stores its original value, corrupted value, issue label and expected repair where applicable. Natural pre-existing findings are subtracted from the prediction set so they do not become benchmark false positives. The corruption mix includes obvious rule violations and plausible-but-wrong values to test the abstention boundary.","metric_definitions":{"precision":"TP/(TP+FP), conditional on the system making a positive detection.","recall":"TP/(TP+FN), proportion of injected labelled corruptions detected with the matching issue label.","f1":"Harmonic mean of precision and recall.","repair_accuracy":"Correct repaired target divided by automatic repair attempts; conditional on automation. Non-inferable cases are never counted as correct.","safe_repair_coverage":"Correct automatic repairs divided by injected cases with an explicitly known safe target. This measures how much of the safely repairable benchmark was actually repaired correctly.","clean_false_positive_rate":"False positives divided by the cells that were NOT deliberately corrupted (total reference cells minus injected cases). Measures how often the detector raises a finding on data it had no reason to flag.","unsafe_repair_rate":"Unsafe/incorrect automatic repairs divided by automatic repair attempts; N/A when no automatic repair was attempted."},"precision":score["precision_pct"],"recall":score["recall_pct"],"f1":score["f1_pct"]}

def run_corruption_sweep(df,rates=DEFAULT_RATES,seeds=DEFAULT_SEEDS):
    rows=[]
    for rate in rates:
        for seed in seeds:
            e=evaluate_detection(df,rate,seed); rows.append({"corruption_rate_pct":rate*100,"seed":seed,"injected_cases":e["actual_corruption_cases"],"obvious_cases":e["obvious_cases"],"plausible_cases":e["plausible_cases"],"precision_pct":e["precision_pct"],"recall_pct":e["recall_pct"],"f1_pct":e["f1_pct"],"repair_accuracy_pct":e["repair_accuracy_pct"],"safe_repair_coverage_pct":e["safe_repair_coverage_pct"],"unsafe_repair_rate_pct":e["unsafe_repair_rate_pct"]})
    return pd.DataFrame(rows)

def summarize_sweep(sweep):
    if sweep is None or sweep.empty:return pd.DataFrame()
    metrics=["precision_pct","recall_pct","f1_pct","repair_accuracy_pct","safe_repair_coverage_pct","unsafe_repair_rate_pct"]; rows=[]
    for rate,g in sweep.groupby("corruption_rate_pct"):
        row={"corruption_rate_pct":rate,"trials":len(g),"injected_cases_mean":round(g["injected_cases"].mean(),2)}
        for m in metrics:
            vals=pd.to_numeric(g[m],errors="coerce").dropna(); row[m+"_mean"]=round(vals.mean(),2) if len(vals) else None; row[m+"_sd"]=round(vals.std(ddof=1),2) if len(vals)>1 else None
        rows.append(row)
    return pd.DataFrame(rows)

def baseline_detect(df):
    findings=set()
    for c in df.columns:
        for idx,v in df[c].items():
            if pd.isna(v) or (isinstance(v,str) and not v.strip()):findings.add((int(idx),c,"Missing value"))
    for idx in df.index[df.duplicated(keep="first")]:findings.add((int(idx),"(entire row)","Duplicate record"))
    return findings

def compare_with_baseline(df,corruption_rate=0.10,seed=42):
    reference,mode,excluded=_benchmark_reference(df); messy,cases=build_benchmark(reference,corruption_rate,seed); truth=_truth_set(cases); original=_set_findings(detect_issues(reference)); ours=_set_findings(detect_issues(messy))-_baseline_for_scoring(original,cases); base_before=baseline_detect(reference); base=baseline_detect(messy)-base_before
    def scores(pred):s=_scores(pred,truth);s["predictions"]=len(pred);return s
    return {"SelfHealDS":scores(ours),"Minimal baseline":scores(base),"ground_truth":len(truth),"generated_at":_now(),"dataset_fingerprint":dataset_fingerprint(df),"benchmark_reference_fingerprint":dataset_fingerprint(reference),"reference_mode":mode,"preexisting_rows_excluded":excluded,"corruption_rate":corruption_rate,"seed":seed,"baseline_definition":"Missing-value and exact-duplicate detection only. This is a transparent internal baseline, not a reproduction of a commercial system. A metric is N/A when the baseline makes no positive predictions rather than being awarded perfect precision."}

def run_ablation(df,corruption_rate=0.10,seed=42):
    reference,_,_= _benchmark_reference(df); messy,cases=build_benchmark(reference,corruption_rate,seed); truth=_truth_set(cases); original=_set_findings(detect_issues(reference)); adjusted_original=_baseline_for_scoring(original,cases); full=_set_findings(detect_issues(messy))-adjusted_original
    det_clean=[i for i in detect_issues(messy) if i.issue_type!="ML statistical anomaly"];det_base=[i for i in detect_issues(reference) if i.issue_type!="ML statistical anomaly"];deterministic=_set_findings(det_clean)-{t for t in _set_findings(det_base) if (t[0],t[1]) not in {(int(c["row"]),str(c["column"])) for c in cases}}
    ml_messy,_=detect_ml_anomalies(messy);ml_clean,_=detect_ml_anomalies(reference);ml_pred={(int(x["row"]),str(x["column"]),"ML statistical anomaly") for x in ml_messy};ml_base={(int(x["row"]),str(x["column"]),"ML statistical anomaly") for x in ml_clean};ml_only=ml_pred-ml_base
    rows=[]
    for name,pred,note in [("Deterministic rules only",deterministic,"Domain, format, unit and semantic rules without ML anomaly findings."),("Hybrid SelfHealDS",full,"Full detector including deterministic rules and Isolation Forest statistical anomaly flags.")]:rows.append({"Variant":name,**_scores(pred,truth),"note":note})
    rows.append({"Variant":"Isolation Forest only","true_positives":"N/A","false_positives":"N/A","false_negatives":"N/A","precision_pct":"N/A","recall_pct":"N/A","f1_pct":"N/A","note":f"{len(ml_only)} statistical anomaly prediction(s) produced. Its taxonomy does not map one-to-one to deterministic corruption labels, so an F1 score would be invalid."})
    return pd.DataFrame(rows)

def evaluate_bundled_suite(data_dir):
    rows=[]
    if not os.path.isdir(data_dir):
        return pd.DataFrame([{"dataset":None,"error":f"Bundled sample directory not found: {data_dir}","generated_at":_now()}])
    csv_paths=sorted(glob.glob(os.path.join(str(data_dir),"*.csv")))
    if not csv_paths:
        return pd.DataFrame([{"dataset":None,"error":f"No CSV files found in bundled sample directory: {data_dir}","generated_at":_now()}])
    for path in csv_paths:
        try:
            df=pd.read_csv(path);comp=compare_with_baseline(df)
            rows.append({"dataset":os.path.basename(path),"rows":len(df),"columns":len(df.columns),"ground_truth":comp["ground_truth"],"SelfHealDS precision (%)":comp["SelfHealDS"]["precision_pct"],"SelfHealDS recall (%)":comp["SelfHealDS"]["recall_pct"],"SelfHealDS F1 (%)":comp["SelfHealDS"]["f1_pct"],"Baseline precision (%)":comp["Minimal baseline"]["precision_pct"],"Baseline recall (%)":comp["Minimal baseline"]["recall_pct"],"Baseline F1 (%)":comp["Minimal baseline"]["f1_pct"],"seed":comp["seed"],"generated_at":comp["generated_at"],"interpretation":"Small controlled sample; not suitable as the paper's final performance claim." if len(df)<MIN_RESEARCH_ROWS else "Candidate for paper benchmarking."})
        except Exception as exc:rows.append({"dataset":os.path.basename(path),"error":str(exc),"generated_at":_now()})
    return pd.DataFrame(rows)
