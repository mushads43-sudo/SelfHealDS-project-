"""Curated real-world data-quality case studies.

The catalog intentionally separates documented provenance from the experiment:
- natural_issue: a problem explicitly documented by the dataset/paper/source;
- controlled_corruption: any synthetic corruption introduced by SelfHealDS for
  ground-truth evaluation. It is never presented as a historical incident.

Remote datasets are cached locally after the first successful download. NASA's
Mars Climate Orbiter is kept as a historical unit-mismatch case; its exact
mission telemetry dataset is not currently downloadable from NASA, so the UI
labels it as an analogous/controlled experiment rather than pretending the
C-MAPSS data contains the incident itself.
"""
from __future__ import annotations
from pathlib import Path
from io import BytesIO
import json, time
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "real_world"
CACHE.mkdir(parents=True, exist_ok=True)

CASE_STUDIES = [
    {
        "id":"nasa_mars_climate_orbiter",
        "name":"Mars Climate Orbiter — unit mismatch",
        "organization":"NASA / JPL",
        "domain":"Aerospace",
        "year":1999,
        "incident":"Mars Climate Orbiter was lost during Mars orbit insertion after a navigation-data unit mismatch between English and metric units.",
        "issue":"Unit inconsistency / interface contract failure: ground software supplied English-unit small-force data while navigation software expected metric units.",
        "impact":"The navigation error put the spacecraft on an unsafe trajectory; NASA reports that the spacecraft entered the atmosphere at about 60 km rather than the planned 150 km and was lost.",
        "source":"NASA/JPL mission and failure-board documentation",
        "source_url":"https://science.nasa.gov/mission/mars-climate-orbiter/",
        "dataset_url":None,
        "dataset_name":"NASA C-MAPSS Aircraft Engine Simulator Data (CSV supplied locally; analogous to the documented MCO unit-contract failure)",
        "local_filename":"01_nasa_cmapss.csv",
        "dataset_url_public":"https://data.nasa.gov/dataset/c-mapss-aircraft-engine-simulator-data",
        "availability":"NASA currently marks C-MAPSS/C-MAPSS40K unavailable for download. The exact historical MCO telemetry is therefore not claimed to be included.",
        "natural_issue":"Historical incident is real and documented; the exact mission telemetry file is not publicly downloadable here.",
        "controlled_issue":"Optional controlled unit corruption on a public NASA aerospace dataset, explicitly labelled synthetic.",
        "detect":"Unit tokens, mixed measurement units, numeric parsing, statistical anomalies and schema consistency where a downloadable analogous dataset is supplied.",
        "repair":"Deterministic unit conversion and normalization only when units are explicitly recognized; otherwise review/quarantine.",
        "case_constraints":{},
    },
    {
        "id":"holoclean_hospital",
        "name":"Hospital quality benchmark",
        "organization":"CMS Hospital Compare / HoloClean benchmark",
        "domain":"Healthcare",
        "year":2017,
        "incident":"A published data-cleaning benchmark uses the Hospital dataset to study real structured-data errors, with ground truth available for all labelled cells.",
        "issue":"Documented benchmark errors include duplicated information and corrupted provider/location/measure fields; the HoloClean paper reports roughly 5% errors and full ground truth.",
        "impact":"Incorrect provider, location and measure values can corrupt healthcare quality reporting and downstream analysis.",
        "source":"HoloClean PVLDB evaluation and HoloClean testdata repository",
        "source_url":"https://cs.uwaterloo.ca/~ilyas/papers/RekatsinasHoloClean.pdf",
        "dataset_url":"https://raw.githubusercontent.com/HoloClean/holoclean/master/testdata/hospital.csv",
        "dataset_name":"HoloClean hospital.csv",
        "local_filename":"02_hospital.csv",
        "ground_truth_filename":"02_hospital_clean.csv",
        "ground_truth_url":"https://raw.githubusercontent.com/HoloClean/holoclean/master/testdata/hospital_clean.csv",
        "availability":"Public CSV; benchmark-curated dirty data. It is not presented as a current operational CMS export.",
        "natural_issue":"Errors are part of a published dirty-data benchmark; they are not newly injected by this application.",
        "controlled_issue":"A separate optional corruption sweep may be applied for reproducible ground-truth stress tests.",
        "detect":"Duplicate records, malformed identifiers, phone/ZIP structure, missingness, categorical/semantic inconsistencies and text typos where supported.",
        "repair":"Safe formatting/normalization, deterministic missing-value treatment and high-confidence semantic corrections; ambiguous provider values remain for review.",
        "case_constraints":{"id_columns":["ProviderNumber"],"consistent_groups":["HospitalName","Address1","City","State","ZipCode","CountyName","PhoneNumber","HospitalType","HospitalOwner","EmergencyService"]},
    },
    {
        "id":"holoclean_flights",
        "name":"Flights multi-source consistency benchmark",
        "organization":"HoloClean / flight-data benchmark",
        "domain":"Transportation",
        "year":2017,
        "incident":"The published benchmark combines flight departure/arrival information from different sources and evaluates conflicts between those sources.",
        "issue":"Conflicting scheduled/actual departure and arrival values across sources; the paper states that most cells are noisy and ground truth is available.",
        "impact":"Conflicting flight times can distort delay analysis, schedules, operational reporting and downstream models.",
        "source":"HoloClean PVLDB evaluation and public testdata repository",
        "source_url":"https://cs.uwaterloo.ca/~ilyas/papers/RekatsinasHoloClean.pdf",
        "dataset_url":"https://raw.githubusercontent.com/HoloClean/holoclean/master/testdata/flight.csv",
        "dataset_name":"HoloClean flight.csv",
        "local_filename":"03_flights.csv",
        "ground_truth_filename":"03_flights_clean.csv",
        "ground_truth_url":"https://raw.githubusercontent.com/HoloClean/holoclean/master/testdata/flight_clean.csv",
        "availability":"Public CSV benchmark dataset.",
        "natural_issue":"Documented multi-source conflicts in the published benchmark; not newly fabricated by SelfHealDS.",
        "controlled_issue":"Optional controlled corruption is clearly separated from the benchmark's existing noise.",
        "detect":"Numeric/type inconsistencies, missingness, duplicate records and flight-time/domain constraints when matching columns are present.",
        "repair":"Canonicalize parseable time/date values and repair only deterministic contradictions; source conflicts without authoritative evidence are review-only.",
        "case_constraints":{"time_pairs":[["DepTime","ArrTime"],["CRSDepTime","CRSArrTime"]]},
    },
    {
        "id":"chicago_food_inspections",
        "name":"Chicago Food Inspections data-quality case",
        "organization":"City of Chicago Department of Public Health",
        "domain":"Government / Food Safety",
        "year":2010,
        "incident":"Chicago publishes restaurant and food-establishment inspection records through its open-data portal; the portal explicitly warns that duplicate inspection reports may remain.",
        "issue":"Duplicate records and cross-record consistency problems. The HoloClean literature documents conflicting ZIP codes, inspection results and facility attributes in this real dataset.",
        "impact":"Duplicate or conflicting inspection records can distort establishment histories, risk summaries and food-safety analytics.",
        "source":"City of Chicago Open Data + published data-cleaning literature",
        "source_url":"https://catalog.data.gov/dataset/food-inspections",
        "dataset_url":"https://data.cityofchicago.org/api/v3/views/4ijn-s7e5/export.csv?accessType=DOWNLOAD",
        "dataset_name":"Chicago Food Inspections",
        "local_filename":"04_chicago_food_inspections.csv",
        "availability":"Public CSV; the live portal is updated over time, so a research run must record the download timestamp/fingerprint.",
        "natural_issue":"The city itself documents possible duplicate reports; published data-cleaning work documents additional integrity conflicts.",
        "controlled_issue":"Any injected errors are labelled synthetic and excluded from the natural-issue counts.",
        "detect":"Duplicates, missingness, date parsing, numeric fields, categorical inconsistencies and selected cross-record functional dependencies.",
        "repair":"Deterministic formatting and exact duplicate handling; conflicting establishment attributes are surfaced for evidence-backed review rather than guessed.",
        "case_constraints":{"id_columns":["License #","License"],"consistent_groups":["DBA Name","Facility Type","Risk","Address","Zip"]},
    },
    {
        "id":"uci_air_quality",
        "name":"UCI Air Quality sensor data",
        "organization":"UCI Machine Learning Repository / urban air-quality study",
        "domain":"Environmental / Sensor",
        "year":2008,
        "incident":"The underlying urban pollution-monitoring study documents sensor cross-sensitivities and sensor/concept drift that affect concentration estimation.",
        "issue":"Missing values are encoded as -200, while sensor drift and cross-sensitivity affect measurement reliability.",
        "impact":"Unrecognized missing sentinels and sensor drift can bias time-series summaries and concentration estimates.",
        "source":"UCI Machine Learning Repository and cited Sensors & Actuators B study",
        "source_url":"https://archive.ics.uci.edu/dataset/360/air%2Bquality",
        "dataset_url":"https://archive.ics.uci.edu/static/public/360/air%2Bquality.zip",
        "dataset_name":"AirQualityUCI.csv",
        "local_filename":"05_air_quality.csv",
        "availability":"Public research dataset; UCI restricts commercial use.",
        "natural_issue":"UCI explicitly documents -200 as the missing-value marker and notes sensor/concept drift.",
        "controlled_issue":"Optional synthetic perturbations may be used for exact repair scoring; they are never described as sensor faults from the original study.",
        "detect":"-200 missing sentinel, numeric parsing, missingness, date/time parsing and statistical anomalies.",
        "repair":"Convert documented -200 sentinel values to missing, then apply the existing conservative missing-value policy; sensor-drift anomalies remain review-only.",
        "case_constraints":{"missing_sentinel":-200},
    },
    {
        "id":"uci_online_retail",
        "name":"UCI Online Retail transaction semantics",
        "organization":"UCI Machine Learning Repository / UK online retailer",
        "domain":"Business / Finance-adjacent Transactions",
        "year":2011,
        "incident":"The dataset records all transactions for a UK-based non-store online retailer. Its data dictionary explicitly defines invoice numbers beginning with C as cancellations.",
        "issue":"Domain-semantic trap: negative quantities and cancellation records are legitimate transaction events in this dataset, not automatically erroneous values.",
        "impact":"A cleaner that blindly treats negative quantities as errors could destroy legitimate returns/cancellations and bias revenue or inventory analysis.",
        "source":"UCI Machine Learning Repository",
        "source_url":"https://uci-ics-mlr-prod.aws.uci.edu/dataset/352/online%2Bretail",
        "dataset_url":"https://archive.ics.uci.edu/static/public/352/online%2Bretail.zip",
        "dataset_name":"Online Retail.csv (locally supplied experimental CSV)",
        "local_filename":"06_online_retail.csv",
        "availability":"Public research dataset under CC BY 4.0; the UCI distribution is an Excel workbook inside the public archive.",
        "natural_issue":"The source data dictionary documents cancellation codes and transaction semantics; these are treated as legitimate domain values, not fabricated errors.",
        "controlled_issue":"Optional synthetic perturbations can test whether SelfHealDS avoids unsafe negative-quantity corrections.",
        "detect":"Missingness, date parsing, numeric parsing, duplicates and schema consistency while explicitly respecting documented cancellation/return semantics.",
        "repair":"Do not auto-flip legitimate negative quantities. Only repair a negative quantity when an independent rule proves it is erroneous.",
        "case_constraints":{"allow_negative_quantity":True},
    },
    {
        "id":"uci_adult_census",
        "name":"UCI Adult / Census Income",
        "organization":"UCI Machine Learning Repository / 1994 U.S. Census extraction",
        "domain":"Government / Census",
        "year":1994,
        "incident":"The dataset is an extraction from the 1994 U.S. Census and is a widely used public benchmark. UCI explicitly marks workclass, occupation and native-country as fields with missing values.",
        "issue":"Missing categorical values are present in the public data; HoloClean also used this dataset for integrity-constraint inconsistency analysis.",
        "impact":"Missing or inconsistent demographic fields can affect segmentation, statistical summaries and models trained from census records.",
        "source":"UCI Machine Learning Repository; HoloClean benchmark literature",
        "source_url":"https://uci-ics-mlr-prod.aws.uci.edu/dataset/2/adult",
        "dataset_url":"https://archive.ics.uci.edu/static/public/2/adult.zip",
        "dataset_name":"adult.data converted to CSV",
        "local_filename":"06_adult_census.csv",
        "availability":"Public CSV/text dataset under CC BY 4.0.",
        "natural_issue":"UCI documents missing values; any HoloClean-style integrity constraints are analytical evaluation rules, not claims of a historical incident.",
        "controlled_issue":"Optional corruption experiments are separately labelled synthetic.",
        "detect":"Placeholder/missing values, categorical consistency, numeric parsing and selected relationship constraints (e.g. relationship/sex contradictions where applicable).",
        "repair":"Observed-mode imputation only within the project's missingness policy; categorical contradictions are review-first unless a deterministic rule supplies the replacement.",
        "case_constraints":{},
    },
]


def get_case(case_id):
    return next((c for c in CASE_STUDIES if c["id"] == case_id), None)


def _read_cached(path):
    if path.exists():
        if path.suffix.lower()=='.xlsx': return pd.read_excel(path)
        return pd.read_csv(path, low_memory=False)
    return None


def case_dataset_filename(case):
    """Return the required local CSV filename for a real-world case."""
    return case.get("local_filename", f"{case['id']}.csv")


def case_ground_truth_filename(case):
    """Return the optional local ground-truth CSV filename, if configured."""
    return case.get("ground_truth_filename")


def load_case_dataset(case, uploaded_bytes=None, uploaded_name=None):
    """Load a case dataset from an uploaded/local CSV only.

    No network access is performed. This deliberately makes the case-study
    workflow deterministic and fast once the researcher has supplied the CSV.
    """
    if uploaded_bytes is not None:
        from io import BytesIO
        return pd.read_csv(BytesIO(uploaded_bytes), low_memory=False), {"status":"uploaded","filename":uploaded_name}
    path=CACHE / case_dataset_filename(case)
    if not path.exists():
        return None, {"status":"missing_local","reason":f"Upload or place {case_dataset_filename(case)} in data/real_world/.","filename":case_dataset_filename(case)}
    return pd.read_csv(path, low_memory=False), {"status":"local","path":str(path),"filename":path.name}


def load_case_ground_truth(case, uploaded_bytes=None):
    """Load optional ground truth from a local/uploaded CSV; never from the web."""
    if uploaded_bytes is not None:
        from io import BytesIO
        return pd.read_csv(BytesIO(uploaded_bytes), low_memory=False)
    name=case_ground_truth_filename(case)
    if not name: return None
    path=CACHE / name
    if not path.exists(): return None
    return pd.read_csv(path, low_memory=False)

def apply_natural_normalization(df, case):
    """Return a copy with documented sentinel handling only; no synthetic errors."""
    out=df.copy()
    if case["id"]=="uci_air_quality":
        for c in out.columns:
            if pd.api.types.is_numeric_dtype(out[c]):
                out.loc[out[c] == -200, c] = np.nan
    return out


def natural_issue_summary(df, case):
    """Count directly observable documented issue patterns without claiming ground truth beyond provenance."""
    counts={"missing_or_placeholders":0,"duplicates":int(df.duplicated().sum()),"sentinel_missing":0,"numeric_outliers":0}
    if case["id"]=="uci_air_quality":
        counts["sentinel_missing"]=int((df == -200).sum().sum())
    counts["missing_or_placeholders"]=int(df.isna().sum().sum())
    return counts


def _ground_truth_score(case, df, healing, truth_override=None):
    url=case.get("ground_truth_url")
    if not url:
        return {"available":False,"reason":"No independently curated cell-level ground truth URL is configured for this case."}
    try:
        truth=truth_override if truth_override is not None else load_case_ground_truth(case)
        if truth is None:
            return {"available":False,"reason":"Ground-truth CSV is not available locally. Upload the documented clean reference for this case to enable cell-level scoring."}
        if truth.shape != df.shape:
            return {"available":False,"reason":f"Ground-truth shape {truth.shape} does not match dirty dataset {df.shape}."}
        dirty=df.reset_index(drop=True); truth=truth.reset_index(drop=True)
        changed=[]
        for c in dirty.columns:
            mask=dirty[c].astype("string").fillna("<NA>").ne(truth[c].astype("string").fillna("<NA>"))
            for i in dirty.index[mask]: changed.append((int(i),str(c)))
        gt=set(changed)
        detected={(int(i.row),str(i.column)) for i in healing["before_issues"]}
        tp=len(detected & gt); fp=len(detected-gt); fn=len(gt-detected)
        precision=tp/(tp+fp)*100 if tp+fp else None; recall=tp/(tp+fn)*100 if tp+fn else None
        attempts=0; correct=0; unsafe=0
        for e in healing["repair_log"]:
            row=e.get("row"); col=str(e.get("column"))
            if not isinstance(row,int) or row<0 or row>=len(truth) or col not in truth.columns: continue
            if str(e.get("status","")).startswith("Automatically"):
                attempts+=1
                expected=truth.iloc[row][col]; actual=e.get("repaired_value")
                if (pd.isna(expected) and pd.isna(actual)) or str(expected)==str(actual) or (pd.notna(expected) and pd.notna(actual) and abs(float(expected)-float(actual))<=1e-9 if isinstance(expected,(int,float,np.integer,np.floating)) and isinstance(actual,(int,float,np.integer,np.floating)) else str(expected)==str(actual)):
                    correct+=1
                else: unsafe+=1
        return {"available":True,"ground_truth_cells":len(gt),"detected_cells":len(detected),"true_positives":tp,"false_positives":fp,"false_negatives":fn,"precision_pct":None if precision is None else round(precision,2),"recall_pct":None if recall is None else round(recall,2),"f1_pct":None if precision is None or recall is None or precision+recall==0 else round(2*precision*recall/(precision+recall),2),"repair_attempts":attempts,"repair_correct":correct,"repair_unsafe":unsafe,"repair_accuracy_pct":None if attempts==0 else round(correct/attempts*100,2)}
    except Exception as exc:
        return {"available":False,"reason":f"Ground-truth load/scoring failed: {exc}"}

def evaluate_case(case, df, ground_truth_df=None):
    """Run the normal SelfHealDS pipeline plus documented natural-issue metrics."""
    from .pipeline import assess, heal
    from .profiler import dataset_fingerprint
    from .rules import POLICY
    natural=natural_issue_summary(df,case)
    # Run detection on the RAW case-study data. In particular, documented
    # sentinels such as UCI Air Quality's -200 must appear as detected errors
    # before the repair stage; normalization belongs inside the repair pipeline.
    assessment=assess(df, context=case["id"])
    healing=heal(df, context=case["id"])
    before=int(len(assessment["issues"]))
    after=int(len(healing["after_issues"]))
    gt=_ground_truth_score(case, df, healing, ground_truth_df)
    return {
        "dataset_fingerprint":dataset_fingerprint(df),
        "rows":len(df),"columns":len(df.columns),"natural_issue_counts":natural,
        "detected_findings":before,"post_repair_findings":after,
        "repair_events":len(healing["repair_log"]),"quarantined_rows":healing["quarantined_rows"],
        "before_quality":healing["before_quality"],"after_quality":healing["after_quality"],
        "assessment":assessment,"healing":healing,
        "controlled_corruption":"Not included in these natural-issue counts. Use Research Evaluation for labelled synthetic corruption sweeps.",
    }
