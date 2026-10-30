import re
from difflib import get_close_matches
import pandas as pd
import numpy as np
from .models import Issue, issue_category
from .rules import CITY_COUNTRY, COUNTRY_ALIASES, KNOWN_COUNTRIES, WORLD_COUNTRIES, is_known_country, normalize_country_token, UNIT_MAP, POLICY, POLICY_RATIONALE, PLACEHOLDERS, is_placeholder_token, semantic_type, parse_numeric, unit_parse, missing_fallback
from .ml_anomaly import detect_ml_anomalies

def add(issues,row,column,issue_type,severity,value,reason,evidence,repair,repairable,confidence,rule,confidence_basis=None):
    if confidence_basis is None:
        if repairable:
            confidence_basis = "Deterministic rule evidence is sufficient under the active repair policy; this score is an evidence strength, not a calibrated probability."
        elif issue_type == "ML statistical anomaly":
            confidence_basis = "Model anomaly score supplies statistical evidence only; the value is intentionally not treated as a repairable error."
        else:
            confidence_basis = "The rule establishes a violation, but the available evidence is insufficient to infer a safe replacement."
    issues.append(Issue(row,column,issue_type,issue_category(issue_type),severity,value,reason,evidence,repair,repairable,float(confidence),rule,False,None,confidence_basis))

def detect_issues(df, context=None):
    if context == "text_document":
        from .text_processing import detect_text_issues
        return detect_text_issues(df)
    issues=[]
    context = context or {}
    context_id = context if isinstance(context, str) else context.get("id")
    if df.empty: return issues

    # Structural integrity
    normalized=[re.sub(r"[^a-z0-9]+","_",str(c).strip().lower()).strip("_") for c in df.columns]
    seen={}
    for pos,name in enumerate(normalized):
        if name in seen:
            add(issues,df.index[0],str(df.columns[pos]),"Duplicate column name","Error",str(df.columns[pos]),
                "Two column names become identical after normalization. This makes field identity ambiguous for downstream transformations.",
                f"Normalized name '{name}' already occurs at column position {seen[name]}.",
                "Rename the columns before downstream processing.",False,.99,"STR-001")
        else: seen[name]=pos
    for c in df.columns:
        s=df[c]
        if s.nunique(dropna=True)<=1:
            add(issues,df.index[0],c,"Constant column","Info",s.dropna().iloc[0] if s.notna().any() else None,
                "The field contains zero or one distinct non-null value. Constant fields may be intentional, but they provide no observed variance for most analyses.",
                f"Distinct non-null values: {s.nunique(dropna=True)}.",
                "Review whether the field is intentionally constant.",False,.99,"STR-002")

    # Row-level duplication
    for idx in df.index[df.duplicated(keep="first")]:
        add(issues,idx,"(entire row)","Duplicate record","Warning","Duplicate row",
            "The complete record is repeated. Retaining it can double-count observations and bias aggregates or models.",
            "Pandas duplicated(keep='first') identifies this row as a repeat of an earlier record.",
            "Remove the later duplicate while retaining the first occurrence.",True,.99,"UNI-001")

    for c in df.columns:
        s=df[c]; st=semantic_type(c)
        if context_id == "uci_adult_census" and c == "salary":
            st="generic"
        # Missingness and placeholders. Some published schemas intentionally
        # leave descriptive fields blank (e.g. Hospital Address2/Address3 and
        # Chicago Violations when no violation text is recorded); these are
        # structural optional fields, not data errors.
        optional_missing = (
            (context_id == "holoclean_hospital" and c in {"Address2","Address3"}) or
            (context_id == "chicago_food_inspections" and c == "Violations")
        )
        if optional_missing:
            continue
        for idx,v in s.items():
            if pd.isna(v) or (isinstance(v,str) and v.strip()==""):
                candidate,method,fb_evidence=missing_fallback(s,st)
                rate=float((s.isna() | s.astype("string").str.strip().eq("")).mean())
                if candidate is not None and rate <= POLICY["missing_max_rate"]:
                    treatment=f"Impute with {candidate!r} using the observed {method} under MIS-001."
                    repairable=True
                    conf=.92
                    basis=f"Candidate is fully derived from observed values: {fb_evidence} Missingness is within the {POLICY['missing_max_rate']*100:.0f}% policy ceiling."
                else:
                    treatment="Do not fabricate a value. Keep the record for review because the active evidence threshold for automatic imputation is not met."
                    repairable=False
                    conf=.95
                    basis=f"Automatic imputation is blocked. {fb_evidence} The policy ceiling is {POLICY['missing_max_rate']*100:.0f}% missingness."
                add(issues,idx,c,"Missing value","Error",v,
                    "The cell contains no usable value. Missingness can affect calculations, joins, summaries and model inputs.",
                    f"Value is null or whitespace-only. {fb_evidence}",
                    treatment,repairable,conf,"MIS-001",basis)
            elif isinstance(v,str) and v.strip().lower() in PLACEHOLDERS:
                add(issues,idx,c,"Placeholder value","Warning",v,
                    "A textual placeholder is being used instead of a canonical missing value. Multiple placeholder forms fragment missing-data handling.",
                    f"Normalized placeholder token: '{v.strip().lower()}'.",
                    "Convert to canonical missingness, then apply the missing-data policy.",True,.98,"MIS-002")

        # String hygiene
        if s.dtype=="object" or pd.api.types.is_string_dtype(s):
            for idx,v in s.items():
                if isinstance(v,str) and v!=v.strip():
                    add(issues,idx,c,"Whitespace inconsistency","Info",v,
                        "Leading or trailing whitespace can create duplicate categories, break exact joins and make identifiers appear different when they are semantically equal.",
                        f"Raw representation has surrounding whitespace: {v!r}.",
                        "Trim surrounding whitespace while preserving internal characters.",True,.99,"FMT-001")

        if st=="age":
            nums=s.map(parse_numeric)
            for idx,n in nums.items():
                if pd.isna(n): continue
                if n<0:
                    safe=abs(n)<=POLICY["age_upper_bound"]
                    add(issues,idx,c,"Negative age","Error",s.loc[idx],
                        "Age is a non-negative duration. A negative value violates the domain constraint.",
                        f"Observed numeric value: {n:g}; absolute magnitude is {'plausible' if safe else 'not plausibly an age'}.",
                        "Remove the erroneous negative sign and revalidate the magnitude." if safe else "Do not infer a replacement; quarantine for review.",safe,.98 if safe else .95,"DOM-AGE-001")
                elif n>POLICY["age_upper_bound"]:
                    add(issues,idx,c,"Impossible age","Error",s.loc[idx],
                        "The value exceeds the configured conservative plausibility boundary for an ordinary human age.",
                        f"Observed value {n:g} exceeds the configured upper plausibility boundary of {POLICY['age_upper_bound']} years.",
                        "Quarantine for review; the correct age cannot be inferred safely from the value alone.",False,.99,"DOM-AGE-002")

        elif st=="phone":
            for idx,v in s.items():
                if pd.isna(v) or str(v).strip()=="" or is_placeholder_token(v): continue
                raw=str(v).strip(); digits=re.sub(r"\D","",raw.lstrip("+"))
                negative=raw.startswith("-") or (pd.notna(parse_numeric(raw)) and parse_numeric(raw)<0)
                if negative:
                    add(issues,idx,c,"Negative phone number","Error",v,
                        "A phone number is an identifier, not a signed measurement. A negative sign makes the identifier structurally invalid.",
                        f"Raw value begins with a negative sign: {raw!r}.",
                        "Remove the sign only when the remaining digit sequence is a valid-length identifier; otherwise quarantine.",POLICY["phone_min_digits"]<=len(digits)<=POLICY["phone_max_digits"],.98 if POLICY["phone_min_digits"]<=len(digits)<=POLICY["phone_max_digits"] else .96,"DOM-PHONE-001")
                if raw!=digits:
                    add(issues,idx,c,"Phone formatting inconsistency","Warning",v,
                        "The identifier contains presentation characters such as spaces, hyphens, brackets or a plus sign. These can create false mismatches in joins and duplicate checks.",
                        f"Canonical digit sequence contains {len(digits)} digits.",
                        f"Normalize to canonical digits-only form: {digits}." if POLICY["phone_min_digits"]<=len(digits)<=POLICY["phone_max_digits"] else "Normalize only after the identifier passes length validation.",POLICY["phone_min_digits"]<=len(digits)<=POLICY["phone_max_digits"],.99,"FMT-PHONE-001")
                if not POLICY["phone_min_digits"]<=len(digits)<=POLICY["phone_max_digits"]:
                    add(issues,idx,c,"Invalid phone length","Error",v,
                        "After harmless formatting characters are removed, the identifier falls outside the conservative 7–15 digit range.",
                        f"Detected {len(digits)} digits; expected {POLICY['phone_min_digits']}–{POLICY['phone_max_digits']} for this generic international validation.",
                        "Quarantine for review because missing digits cannot be inferred safely.",False,.99,"DOM-PHONE-002")

        elif st=="email":
            for idx,v in s.items():
                if pd.isna(v) or str(v).strip()=="" or is_placeholder_token(v): continue
                raw=str(v).strip()
                valid=bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+",raw))
                if not valid:
                    add(issues,idx,c,"Malformed email","Error",v,
                        "The value does not satisfy the minimum local-part@domain structure expected of an email identifier.",
                        f"Regex validation failed for {raw!r}.",
                        "Trim and lowercase only; do not invent a missing address component.",False,.99,"FMT-EMAIL-001")
                elif raw!=str(v):
                    pass

        elif st=="time":
            for idx,v in s.items():
                if pd.isna(v) or str(v).strip()=="" or is_placeholder_token(v): continue
                raw=str(v).strip()
                valid=False
                if context_id=="holoclean_flights":
                    try:
                        n=float(raw); valid=(n.is_integer() and 0<=n<=2359 and int(n)%100<60)
                    except Exception: valid=False
                else:
                    valid=bool(re.fullmatch(r"([01]\d|2[0-3])\.([0-5]\d)\.([0-5]\d)",raw))
                if not valid:
                    add(issues,idx,c,"Invalid time","Error",v,
                        "The value does not match the time representation required by this case-study dataset.",
                        f"Observed time token {v!r}.","Use an observed valid time only when the replacement can be supported by the case-study fallback policy.",
                        True,.97,"CASE-TIME-001","Explicit time-format validation.")
        elif st=="date":
            parsed=pd.to_datetime(s,errors="coerce",dayfirst=False,format="mixed")
            for idx,v in s.items():
                if pd.isna(v) or str(v).strip()=="" or is_placeholder_token(v): continue
                if pd.isna(parsed.loc[idx]):
                    add(issues,idx,c,"Invalid date","Error",v,
                        "The value cannot be interpreted as a valid calendar date under the deterministic parser.",
                        f"Parser result: NaT for {v!r}.",
                        "Quarantine unless the intended date can be determined unambiguously from source context.",False,.99,"FMT-DATE-001")
                else:
                    canonical=parsed.loc[idx].strftime("%Y-%m-%d")
                    if str(v).strip()!=canonical:
                        add(issues,idx,c,"Inconsistent date format","Warning",v,
                            "The date is valid but represented in a non-canonical form, which can complicate sorting and downstream parsing.",
                            f"Parsed date is {canonical}.","Normalize to ISO YYYY-MM-DD.",True,.98,"FMT-DATE-002")

        elif st in {"currency_numeric","quantity"}:
            nums=s.map(parse_numeric)
            for idx,v in s.items():
                if pd.isna(v) or str(v).strip()=="" or is_placeholder_token(v): continue
                n=nums.loc[idx]
                if pd.isna(n):
                    add(issues,idx,c,"Non-numeric value in numeric field","Error",v,
                        "The field is inferred to represent a numeric measure, but this value cannot be parsed as a number.",
                        f"Deterministic numeric parser returned NaN for {v!r}.",
                        "Parse only unambiguous currency/thousands formatting; otherwise quarantine.",False,.99,"TYPE-NUM-001")
                elif st=="quantity" and n<0 and context_id != "uci_online_retail":
                    safe=abs(n)>=0
                    add(issues,idx,c,"Negative quantity","Error",v,
                        "A quantity/count is non-negative under ordinary measurement semantics.",
                        f"Observed numeric value: {n:g}.",
                        "Remove the erroneous negative sign and validate the resulting quantity." if safe else "Quarantine for review.",True,.97,"DOM-QTY-001")
            clean=nums.dropna()
            if len(clean)>=8 and clean.nunique()>3:
                q1,q3=clean.quantile(.25),clean.quantile(.75); iqr=q3-q1
                if iqr>0:
                    lo,hi=q1-POLICY["outlier_iqr_multiplier"]*iqr,q3+POLICY["outlier_iqr_multiplier"]*iqr
                    for idx,n in nums.items():
                        if pd.notna(n) and (n<lo or n>hi):
                            add(issues,idx,c,"Statistical outlier","Warning",s.loc[idx],
                                "The value is unusual relative to the column distribution. An outlier is evidence of unusualness, not proof of an error.",
                                f"IQR rule: observed {n:g}; expected exploratory fence [{lo:.3g}, {hi:.3g}] using {POLICY['outlier_iqr_multiplier']}×IQR. This is a screening rule, not a proof of corruption.",
                                "Flag for review rather than automatically deleting or replacing a potentially legitimate observation.",False,.90,"STAT-IQR-001")

        elif st in {"weight","length","temperature"}:
            parsed=[unit_parse(v,st) for v in s]
            units={p[1] for p in parsed if p and p[1]}
            if len(units)>1:
                canonical={"weight":"kg","length":"m","temperature":"c"}[st]
                for idx,p in zip(df.index,parsed):
                    if p and p[1] and p[1]!=canonical:
                        add(issues,idx,c,"Mixed measurement units","Error",df.at[idx,c],
                            f"The column contains multiple units ({', '.join(sorted(units))}), so raw magnitudes are not directly comparable.",
                            f"Observed unit '{p[1]}'; canonical unit is '{canonical}'.",
                            f"Convert to canonical unit {canonical} using the deterministic conversion table.",True,.99,"UNIT-001")
            for idx,v in s.items():
                if pd.isna(v) or str(v).strip()=="" or is_placeholder_token(v): continue
                if unit_parse(v,st) is None:
                    add(issues,idx,c,"Unrecognized measurement value","Error",v,
                        f"The value cannot be parsed as a numeric {st} measurement with a recognized unit.",
                        f"Unit parser rejected {v!r}.","Quarantine rather than inventing a conversion.",False,.99,"UNIT-002")

        elif st=="country":
            for idx,v in s.items():
                if pd.isna(v) or str(v).strip()=="" or is_placeholder_token(v): continue
                if not is_known_country(v):
                    norm=normalize_country_token(v)
                    ref=WORLD_COUNTRIES|{c2.lower() for c2 in KNOWN_COUNTRIES}
                    close=get_close_matches(norm,list(ref),n=1,cutoff=.86)
                    repair=f"Normalize to '{close[0].title()}' because it is the unique high-similarity known country." if close else "Review against an authoritative country reference."
                    add(issues,idx,c,"Unknown country","Warning",v,
                        "The country value is not present in the reference vocabulary, which covers UN member states and common aliases.",
                        f"Normalized candidate: {close[0].title() if close else 'none'}.",repair,bool(close),.90 if close else .70,"SEM-COUNTRY-001")

    # Cross-field semantic validation
    city_cols=[c for c in df.columns if semantic_type(c)=="city"]
    country_cols=[c for c in df.columns if semantic_type(c)=="country"]
    if city_cols and country_cols:
        city,country=city_cols[0],country_cols[0]
        for idx in df.index:
            cv,kv=df.at[idx,city],df.at[idx,country]
            if pd.isna(cv) or pd.isna(kv): continue
            city_norm=str(cv).strip().lower(); country_norm=COUNTRY_ALIASES.get(str(kv).strip().lower(),str(kv).strip())
            expected=CITY_COUNTRY.get(city_norm)
            if expected and country_norm.lower()!=expected.lower():
                add(issues,idx,country,"City-country semantic inconsistency","Error",kv,
                    f"The city '{cv}' is mapped to {expected}, but the row states '{kv}'. The two fields contradict one another.",
                    f"Reference relationship: {cv} -> {expected}; observed country: {kv}.",
                    f"Set country to '{expected}' because the configured city-country relationship is high-confidence.",True,.99,"SEM-GEO-001")
            elif not expected:
                close=get_close_matches(city_norm,list(CITY_COUNTRY),n=1,cutoff=.88)
                if close and CITY_COUNTRY[close[0]].lower()==country_norm.lower() and close[0]!=city_norm:
                    add(issues,idx,city,"Near-match categorical value","Warning",cv,
                        "The city is not an exact reference value, but a unique close match agrees with the row's country. This is a candidate typo, not proof of corruption.",
                        f"Candidate '{close[0]}' is the unique close match and maps to {country_norm}.",
                        f"Normalize city to '{close[0]}' only under high-confidence fuzzy matching.",True,.93,"SEM-GEO-002")
                elif city_norm not in CITY_COUNTRY:
                    add(issues,idx,city,"Unknown city","Info",cv,
                        "The city is outside the configured reference vocabulary. This does not prove the value is wrong because the reference is intentionally limited.",
                        "No high-confidence reference mapping was found.","Review against an authoritative geographic reference if semantic validation is required.",False,.70,"SEM-GEO-003")
    # Case-study integrity constraints. These are explicit domain rules, not
    # hidden dataset-specific repairs. They are activated only when a curated
    # case-study context is supplied by the UI.
    if context_id == "holoclean_hospital":
        id_cols=[c for c in df.columns if str(c).strip().lower() in {"providernumber","provider number"}]
        if id_cols:
            key=id_cols[0]
            for idx,v in df[key].items():
                if pd.isna(v): continue
                if not re.fullmatch(r"\d{5}", str(v).strip()):
                    add(issues,idx,key,"Invalid provider identifier","Error",v,
                        "The published Hospital benchmark treats ProviderNumber as a five-digit provider identifier; this value is not structurally valid.",
                        f"Observed {v!r}; expected five digits.","Quarantine/review because the missing identifier digits cannot be inferred safely.",False,.99,"CASE-HOSP-001")
        # Detect cross-row contradictions for fields that should be stable for a provider.
        if id_cols:
            key=id_cols[0]
            stable=[c for c in ["HospitalName","Address1","City","State","ZipCode","CountyName","PhoneNumber","HospitalType","HospitalOwner","EmergencyService"] if c in df.columns]
            for c in stable:
                nunique=df.groupby(key,dropna=False)[c].nunique(dropna=True)
                bad=nunique[nunique>1].index
                for provider in bad:
                    for idx in df.index[df[key].eq(provider)]:
                        add(issues,idx,c,"Cross-record consistency conflict","Warning",df.at[idx,c],
                            f"Provider {provider!r} has multiple observed values for {c}, which conflicts with a provider-level consistency constraint.",
                            f"Distinct observed non-null values: {int(nunique.loc[provider])}.",
                            "Review against the authoritative source; no value is fabricated automatically.",False,.86,"CASE-HOSP-002")
    elif context_id == "holoclean_flights":
        for dep,arr in [("DepTime","ArrTime"),("CRSDepTime","CRSArrTime")]:
            if dep in df.columns and arr in df.columns:
                d=pd.to_numeric(df[dep],errors="coerce"); a=pd.to_numeric(df[arr],errors="coerce")
                for idx in df.index[(d.notna()) & (a.notna()) & (d>2359) | (a>2359)]:
                    add(issues,idx,dep if d.loc[idx]>2359 else arr,"Invalid flight time","Error",df.at[idx,dep if d.loc[idx]>2359 else arr],
                        "Flight time is outside the HHMM-style 0000–2359 representation used by this benchmark.",
                        f"Observed departure={df.at[idx,dep]!r}, arrival={df.at[idx,arr]!r}.","Normalize only when the source format is unambiguous; otherwise review.",False,.98,"CASE-FLT-001")
    elif context_id == "chicago_food_inspections":
        # The city portal explicitly warns that duplicate inspection reports may remain.
        # We surface them rather than silently deleting them.
        for idx in df.index[df.duplicated(keep=False)]:
            add(issues,idx,"(entire row)","Duplicate inspection record","Warning",None,
                "The Chicago Open Data portal warns that duplicate inspection reports may remain in the dataset.",
                "Exact row duplicate detected by pandas duplicated(keep=False).","Review duplicate provenance before removal.",False,.99,"CASE-CHI-001")
    elif context_id == "uci_air_quality":
        # UCI Air Quality explicitly uses -200 as a missing-value sentinel.
        # This MUST be reported on the raw dataset before normalization so the
        # case study demonstrates that SelfHealDS detects the documented issue.
        for c in df.columns:
            for idx,v in df[c].items():
                try:
                    is_sentinel = (pd.notna(v) and float(v) == -200)
                except Exception:
                    is_sentinel = str(v).strip() == "-200"
                if is_sentinel:
                    add(issues,idx,c,"Documented missing sentinel","Error",v,
                        "UCI documents -200 as the missing-value marker in this dataset. Treating -200 as a real measurement would bias sensor statistics and downstream models.",
                        "Exact documented UCI sentinel match: -200.",
                        "Convert -200 to canonical missingness, then apply the conservative observed-data imputation policy.",
                        True,.99,"CASE-AQ-001",
                        "Dataset documentation identifies -200 as missing; no statistical assumption is needed.")
    elif context_id == "uci_air_quality":
        # All measurement columns in this case (everything except Date/Time)
        # are numeric sensor/chemistry measurements. Detect malformed tokens,
        # including text that pandas would otherwise leave as an object value.
        for c in df.columns:
            if c in {"Date","Time"}: continue
            nums=df[c].map(parse_numeric)
            for idx,v in df[c].items():
                if pd.notna(v) and pd.isna(nums.loc[idx]) and str(v).strip()!="":
                    add(issues,idx,c,"Non-numeric sensor value","Error",v,
                        "The sensor measurement column contains a token that cannot be parsed as a numeric reading.",
                        f"Deterministic numeric parser rejected {v!r}.",
                        "Use the observed median of valid readings when sufficient evidence exists.",
                        True,.98,"CASE-AQ-002","Numeric sensor schema + deterministic parse failure.")
    elif context_id == "holoclean_flights":
        # Case-study integrity checks for malformed values in the supplied benchmark.
        if "ArrDelay" in df.columns:
            for idx,v in df["ArrDelay"].items():
                if pd.notna(v) and pd.isna(parse_numeric(v)):
                    add(issues,idx,"ArrDelay","Non-numeric value in numeric field","Error",v,
                        "Arrival delay is a numeric operational measure, but this value cannot be parsed as a number.",
                        f"Deterministic numeric parser rejected {v!r}.",
                        "Impute from observed flight-delay values when the column has sufficient evidence.",
                        True,.99,"CASE-FLT-002","Numeric semantic type + deterministic parse failure.")
        if "DepTime" in df.columns:
            for idx,v in df["DepTime"].items():
                try: ok=pd.notna(v) and 0 <= float(v) <= 2359
                except Exception: ok=False
                if pd.notna(v) and not ok:
                    add(issues,idx,"DepTime","Invalid flight time","Error",v,
                        "Departure time is outside the benchmark's HHMM-style 0000–2359 representation.",
                        f"Observed value {v!r} is not a valid numeric HHMM value.",
                        "Replace with an observed valid departure time only when the case-study fallback policy has sufficient evidence.",
                        True,.98,"CASE-FLT-001","Explicit HHMM domain constraint.")
        if "Cancelled" in df.columns:
            for idx,v in df["Cancelled"].items():
                if isinstance(v,str) and v.strip().lower() in {"yes","no","true","false"}:
                    add(issues,idx,"Cancelled","Boolean representation inconsistency","Warning",v,
                        "The cancellation indicator uses a textual boolean representation instead of the canonical 0/1 representation.",
                        f"Observed boolean token {v!r}.","Normalize yes/no/true/false to 1/0.",True,.99,"CASE-FLT-003","Deterministic boolean mapping.")
    elif context_id == "chicago_food_inspections":
        if "Zip" in df.columns:
            for idx,v in df["Zip"].items():
                if pd.notna(v) and not re.fullmatch(r"\d{5}",str(v).strip()):
                    add(issues,idx,"Zip","Invalid ZIP code","Error",v,
                        "Chicago ZIP codes in this dataset are represented as five digits; malformed values can break geographic grouping and joins.",
                        f"Observed {v!r} does not match the five-digit structural contract.",
                        "Use the observed modal ZIP for the same city/state when the grouping provides sufficient evidence.",
                        True,.96,"CASE-CHI-002","Five-digit structural validation plus within-group observed-value fallback.")
        if "Inspection Date" in df.columns:
            parsed=pd.to_datetime(df["Inspection Date"],errors="coerce",format="mixed")
            for idx,v in df["Inspection Date"].items():
                if pd.notna(v) and pd.isna(parsed.loc[idx]):
                    add(issues,idx,"Inspection Date","Invalid date","Error",v,
                        "Inspection Date cannot be parsed as a calendar date.",
                        f"Parser returned NaT for {v!r}.","Use the observed median inspection date when sufficient evidence exists.",True,.97,"CASE-CHI-003","Deterministic parse failure + observed-date fallback.")
        if "Risk" in df.columns:
            valid={"Risk 1 (High)","Risk 2 (Medium)","Risk 3 (Low)"}
            for idx,v in df["Risk"].items():
                if pd.notna(v) and str(v).strip() not in valid:
                    add(issues,idx,"Risk","Unknown risk category","Error",v,
                        "Risk must use one of the documented Chicago inspection risk categories.",
                        f"Observed category {v!r} is outside the configured domain vocabulary.",
                        "Replace with the observed modal risk category because the dataset provides a closed operational vocabulary.",
                        True,.95,"CASE-CHI-004","Closed vocabulary + observed mode.")
        if "Results" in df.columns:
            for idx,v in df["Results"].items():
                if isinstance(v,str) and v.strip().lower() in {"pass","fail","no entry","pass w/ conditions"} and v not in {"Pass","Fail","No Entry","Pass w/ Conditions"}:
                    add(issues,idx,"Results","Categorical representation inconsistency","Warning",v,
                        "Inspection results use a case/format variant of a known operational category.",
                        f"Case-insensitive match found for {v!r}.","Normalize to the canonical category spelling.",True,.99,"CASE-CHI-005","Exact case-insensitive category mapping.")
    elif context_id == "uci_online_retail":
        if "Quantity" in df.columns:
            for idx,v in df["Quantity"].items():
                if pd.notna(v) and pd.isna(parse_numeric(v)):
                    add(issues,idx,"Quantity","Non-numeric value in quantity field","Error",v,
                        "Transaction quantity must be numerically parseable; legitimate negative cancellation quantities are handled separately.",
                        f"Deterministic numeric parser rejected {v!r}.","Impute from observed non-cancellation quantities when sufficient evidence exists.",True,.99,"CASE-RET-001","Numeric parse failure.")
        if "UnitPrice" in df.columns:
            for idx,v in df["UnitPrice"].items():
                n=parse_numeric(v)
                if pd.notna(n) and n < 0:
                    add(issues,idx,"UnitPrice","Negative unit price","Error",v,
                        "A negative unit price is outside the transaction-price domain used for this experimental case. This rule does not treat negative Quantity as erroneous because UCI documents cancellation semantics.",
                        f"Observed unit price {n:g} is negative.","Remove only the invalid sign when the resulting price is otherwise plausible.",True,.97,"CASE-RET-002","Domain constraint with magnitude-preserving correction.")
    elif context_id == "uci_adult_census":
        if "age" in df.columns:
            for idx,v in df["age"].items():
                if pd.notna(v) and pd.isna(parse_numeric(v)):
                    add(issues,idx,"age","Non-numeric age","Error",v,
                        "Age is a numeric demographic field, but this value cannot be parsed as a number.",
                        f"Deterministic numeric parser rejected {v!r}.","Impute from the observed age distribution when sufficient evidence exists.",True,.98,"CASE-ADULT-001","Numeric semantic type + parse failure.")
                else:
                    n=parse_numeric(v)
                    if pd.notna(n) and not 0 <= n <= 120:
                        add(issues,idx,"age","Impossible age","Error",v,
                            "Age falls outside the configured human-age plausibility range.",
                            f"Observed age {n:g}.","Replace with the observed median age for the case when sufficient evidence exists.",True,.95,"CASE-ADULT-002","Explicit domain constraint + observed median.")
        if "hours-per-week" in df.columns:
            for idx,v in df["hours-per-week"].items():
                n=parse_numeric(v)
                if pd.notna(n) and not 0 <= n <= 99:
                    add(issues,idx,"hours-per-week","Invalid work-hours value","Error",v,
                        "Weekly work hours are outside the 0–99 operational range used for this dataset.",
                        f"Observed {n:g} hours/week.","Replace with the observed median when sufficient evidence exists.",True,.95,"CASE-ADULT-003","Explicit domain constraint + observed median.")

    # Machine-learning statistical anomaly layer.
    # This layer only flags unusual observations; it never performs a repair.
    ml_anomalies, ml_meta = detect_ml_anomalies(df)
    for a in ml_anomalies:
        score=a["score"]
        # Isolation Forest decision_function is negative for anomalies.
        strength=min(0.99, max(0.70, 0.70 + min(0.29, abs(score))))
        add(issues,a["row"],a["column"],"ML statistical anomaly","Warning",a["value"],
            "Isolation Forest identified this observation as statistically unusual relative to the selected numeric features. Statistical unusualness is not proof that the value is incorrect.",
            f"Model: {ml_meta.get('model')}; decision score: {score:.6f}; numeric features: {', '.join(a['features'])}. Lower decision scores indicate stronger isolation.",
            "Flag for human review; do not automatically replace a statistically unusual value.",
            False,strength,"ML-IF-001")
    return issues
