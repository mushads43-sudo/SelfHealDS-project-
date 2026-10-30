import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
import pandas as pd
import streamlit as st

from backend.pipeline import assess, heal
from backend.research import run_research_suite, to_jsonable
from backend.models import ISSUE_TAXONOMY
from backend.rules import POLICY, POLICY_RATIONALE, missing_fallback, semantic_type, canonicalize_placeholders
from backend.research import RULESET_VERSION
from backend.evaluation import ground_truth_table
from backend.case_studies import CASE_STUDIES, get_case, load_case_dataset, natural_issue_summary, apply_natural_normalization, evaluate_case
from backend.text_processing import text_to_dataframe, dataframe_to_text
from backend import llm_baseline

SEVERITY_ORDER = {"Error": 0, "Warning": 1, "Info": 2}

st.set_page_config(page_title="SelfHealDS | Data Quality Intelligence", page_icon="", layout="wide", initial_sidebar_state="expanded")

st.markdown(r'''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root{--navy:#0B1220;--navy2:#12213A;--blue:#2563EB;--teal:#0F766E;--green:#15803D;--red:#B91C1C;--amber:#B45309;--ink:#172033;--muted:#667085;--line:#E4E8EF;--bg:#F4F6F9;--white:#fff}
html,body,[class*="css"]{font-family:"DM Sans",sans-serif;color:var(--ink);font-size:15px}.stApp{background:var(--bg)}.block-container{max-width:1560px;padding:30px 48px 64px}#MainMenu,footer{visibility:hidden}header{background:transparent!important}
[data-testid="stDataFrame"]{font-size:14px}
h1,h2,h3,h4{line-height:1.25}
[data-testid="stSidebar"]{background:var(--navy);border-right:1px solid #1D2A40}[data-testid="stSidebar"] *{color:#D8E1EE!important}[data-testid="stSidebar"] hr{border-color:#25334A}
[data-testid="stSidebar"] .stButton>button{background:transparent!important;color:#D8E1EE!important;border:0!important;text-align:left!important;box-shadow:none!important;padding:5px 8px!important;min-height:32px!important;font-weight:500!important;font-size:14px!important}
[data-testid="stSidebar"] .stButton>button:hover{background:transparent!important;color:#FFFFFF!important}
[data-testid="stSidebar"] .stButton>button p{font-size:14px!important;margin:0!important;color:#D8E1EE!important}
.side-brand{padding:4px 4px 25px}.side-mark{display:inline-flex;width:38px;height:38px;border-radius:9px;align-items:center;justify-content:center;background:#2563EB;color:white!important;font-family:Manrope;font-size:15px;font-weight:800;margin-right:10px}.side-name{color:white!important;font-family:Manrope;font-size:20px;font-weight:800;vertical-align:middle}.side-caption{color:#8292AA!important;font-size:11.5px;text-transform:uppercase;letter-spacing:.12em;margin-top:8px}
.side-workspace{color:#D8E1EE!important;font-family:Manrope;font-size:14px;font-weight:800;margin:0 0 8px 4px}
.side-page-active{margin:2px 0;padding:5px 8px;color:#FFFFFF!important;font-family:"DM Sans",sans-serif;font-size:14px;font-weight:700}.side-group{margin:0 0 4px}
.side-group .stButton>button{font-family:Manrope!important;font-size:14px!important;font-weight:700!important;letter-spacing:.01em!important;padding:8px 7px!important}
.side-chevron{display:inline-block;width:20px;color:#D8E1EE!important;font-size:18px;font-weight:500}
.side-child{margin-left:10px;padding-left:12px;border-left:1px solid #1B2A3F}
.side-child .stButton>button{font-family:"DM Sans",sans-serif!important;font-weight:500!important;padding:5px 8px!important}
.side-child-active{margin:2px 0;padding:7px 8px;border-radius:7px;background:#162640;color:#FFFFFF!important;font-weight:700;font-size:14px}
.side-bullet{display:inline-block;width:15px;color:#F45B55!important;font-size:19px;line-height:0;vertical-align:-1px}
.side-bullet-inactive{display:inline-block;width:15px;color:#F2F4F7!important;font-size:18px;line-height:0;vertical-align:-1px}
.side-divider{height:1px;background:#1B2A3F;margin:18px 4px 16px}
.eyebrow{color:#3B82F6;font-size:12.5px;font-weight:800;letter-spacing:.13em;text-transform:uppercase;margin-bottom:8px}.hero{background:linear-gradient(120deg,#0B1220,#12223B 65%,#123E4A);border-radius:18px;padding:36px 40px;color:#fff;box-shadow:0 14px 40px rgba(15,23,42,.12);margin-bottom:24px}.hero h1{margin:0 0 12px;color:#fff;font-family:Manrope;font-size:40px;font-weight:800;line-height:1.08;letter-spacing:-1.4px}.hero p{margin:0;max-width:960px;color:#C9D4E3;font-size:15px;line-height:1.7}.hero-line{height:1px;background:rgba(255,255,255,.13);margin:22px 0 16px}.hero-foot{color:#9FB0C7;font-size:13px}
.section-title{font-family:Manrope;font-size:22px;font-weight:800;margin:0}.section-copy{color:var(--muted);font-size:14.5px;line-height:1.6;margin:5px 0 17px}
.card,.rw-card,.why-card{background:#fff;border:1px solid var(--line);box-shadow:0 3px 15px rgba(15,23,42,.035)}
.card{border-radius:13px;padding:19px;height:100%}.card h4{font-family:Manrope;font-size:15px;margin:0 0 7px}.card p{color:var(--muted);font-size:14px;line-height:1.6;margin:0}
.metric-card{background:#fff;border:1px solid var(--line);border-radius:13px;padding:15px 17px;box-shadow:0 3px 15px rgba(15,23,42,.035)}.research-metric-card{background:#0B1220;border:1px solid #1E2C43;border-radius:13px;padding:18px 18px;box-shadow:0 10px 24px rgba(7,15,29,.12);min-height:145px}.research-metric-card .metric-label{color:#91A4BF}.research-metric-card .metric-value{color:#F5F8FC;font-size:29px}.research-metric-card .metric-sub{color:#8292AA}.research-metric-card .metric-accent{height:2px;width:28px;background:#2563EB;margin:9px 0 2px}.metric-label{color:#7A8698;font-size:11.5px;font-weight:800;text-transform:uppercase;letter-spacing:.1em}.metric-value{font-family:Manrope;color:var(--ink);font-size:25px;font-weight:800;margin-top:4px}.metric-sub{color:#98A2B3;font-size:11.5px;margin-top:2px}
.callout{padding:15px 17px;border-radius:10px;font-size:14px;line-height:1.6;border:1px solid #D7E4F5;background:#F6FAFF;color:#334E68}.callout-success{border-color:#CDE8D5;background:#F4FBF6;color:#28623C}.callout-warning{border-color:#F2D9A8;background:#FFFBF2;color:#6A4B18}.callout-danger{border-color:#F1CACA;background:#FFF7F7;color:#7A2424}.rule-box{background:#F9FAFB;border:1px solid #EAECF0;border-radius:10px;padding:14px 16px;margin:8px 0}.rule-box p{margin:5px 0 0;color:#667085;font-size:13.5px;line-height:1.6}.status{display:inline-block;padding:4px 8px;border-radius:999px;font-size:11.5px;font-weight:800;text-transform:uppercase;letter-spacing:.04em}.status-error{color:#991B1B;background:#FEF2F2}.status-warning{color:#92400E;background:#FFFBEB}.status-info{color:#1D4ED8;background:#EFF6FF}.status-success{color:#166534;background:#F0FDF4}.small-note{color:#7B8797;font-size:13px;line-height:1.55}.footer{color:#98A2B3;text-align:center;font-size:12px;padding-top:35px}.metric-validity{color:#7B8797;font-size:12px;line-height:1.5;padding:9px 12px;background:#F8FAFC;border-left:3px solid #CBD5E1;border-radius:5px;margin-top:10px}.research-badge{display:inline-block;font-size:11.5px;font-weight:800;letter-spacing:.08em;text-transform:uppercase;color:#155E75;background:#ECFEFF;border:1px solid #A5F3FC;border-radius:999px;padding:4px 8px}
.stButton>button,.stDownloadButton>button{min-height:42px;border-radius:9px!important;font-weight:700!important;border:1px solid #D7DEE8!important}.stButton>button[kind="primary"]{background:#2563EB!important;border-color:#2563EB!important}div[data-testid="stFileUploader"]{background:#F8FAFC;border:1px dashed #AEB9C9;border-radius:10px;padding:5px}div[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:9px;overflow:hidden}hr{border-color:var(--line)}
.rw-hero{background:linear-gradient(120deg,#0B1220,#12223B 65%,#123E4A);border-radius:18px;padding:32px 38px;color:#fff;box-shadow:0 14px 40px rgba(15,23,42,.12);margin:18px 0 24px}.rw-hero h1{font-family:Manrope;font-size:38px!important;line-height:1.08;margin:8px 0 12px;color:#fff}.rw-hero p{font-size:17px;line-height:1.75;color:#D7E1EE;max-width:1100px}.rw-eyebrow{font-size:13.5px;font-weight:800;letter-spacing:.13em;color:#60A5FA}.rw-meta{display:flex;gap:45px;border-top:1px solid rgba(255,255,255,.14);margin-top:22px;padding-top:18px;font-size:14px;color:#D7E1EE}.rw-card{border-radius:14px;padding:23px 25px;margin:0 0 18px}.rw-card h2{font-family:Manrope;font-size:21px;margin:0 0 8px;color:var(--ink)}.rw-card p{font-size:15px;line-height:1.75;color:var(--muted);margin:0 0 18px}.rw-section{font-family:Manrope;font-size:24px;font-weight:800;margin:28px 0 12px;color:var(--ink)}.why-hero{background:#0B1220;border-radius:18px;padding:34px 38px;color:#fff;margin:18px 0 24px}.why-hero h1{font-family:Manrope;font-size:36px;margin:0 0 10px}.why-hero p{font-size:16px;line-height:1.7;color:#D7E1EE;max-width:1050px}.flow{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:14px 0}.flow-step{background:#F8FAFC;border:1px solid #DCE3EC;border-radius:10px;padding:12px 15px;font-weight:800;font-size:14.5px}.flow-arrow{color:#64748B;font-weight:800}.why-card{border-radius:14px;padding:21px;height:100%}.why-card h3{font-family:Manrope;font-size:17px;margin:0 0 8px}.why-card p{font-size:14.5px;line-height:1.65;color:var(--muted);margin:0}

.research-ribbon{display:flex;gap:10px;flex-wrap:wrap;margin:0 0 20px}
.research-pill{background:#fff;border:1px solid #DCE3EC;border-radius:999px;padding:7px 12px;font-size:12.5px;font-weight:800;color:#344054}
.research-pill strong{color:#155E75}
.proof-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:16px 0 22px}
.proof-card{background:#fff;border:1px solid var(--line);border-radius:13px;padding:17px;min-height:118px}
.proof-card .proof-num{font-family:Manrope;font-size:27px;font-weight:800;color:var(--ink)}
.proof-card .proof-label{font-size:12.5px;font-weight:800;text-transform:uppercase;letter-spacing:.08em;color:#7A8698;margin-top:3px}
.proof-card .proof-copy{font-size:13px;line-height:1.45;color:#667085;margin-top:6px}
.heal-banner{background:linear-gradient(120deg,#0B1220,#17304A);border-radius:15px;padding:21px 24px;color:#fff;margin:18px 0}
.heal-banner h3{font-family:Manrope;font-size:20px;margin:0 0 5px;color:#fff}
.heal-banner p{font-size:14.5px;line-height:1.6;color:#D7E1EE;margin:0}
.before-after{display:grid;grid-template-columns:1fr auto 1fr;gap:12px;align-items:center;margin:15px 0}
.ba-card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:15px}
.ba-card h4{font-family:Manrope;margin:0 0 6px;font-size:14.5px}
.ba-card p{margin:0;color:#667085;font-size:13.5px;line-height:1.5}
.ba-arrow{font-size:20px;font-weight:800;color:#2563EB}
.repair-focus{border:1px solid #CDE8D5;background:#F5FCF7;border-radius:13px;padding:17px 19px;margin:15px 0}
.repair-focus h4{font-family:Manrope;font-size:15px;margin:0 0 7px;color:#166534}
.repair-focus p{font-size:14px;line-height:1.6;color:#28623C;margin:0}
@media(max-width:900px){.proof-grid{grid-template-columns:repeat(2,1fr)}.before-after{grid-template-columns:1fr}.ba-arrow{display:none}}

.proof-flow-card{background:#0B1220;border-radius:14px;padding:17px 20px;margin:0 0 22px;border:1px solid #1E2C43}.proof-flow-title{font-size:11.5px;font-weight:800;letter-spacing:.12em;color:#7DD3FC;margin-bottom:13px}.proof-flow{display:flex;align-items:center;gap:9px;overflow-x:auto}.proof-flow>div{min-width:120px;background:#12213A;border:1px solid #263B58;border-radius:10px;padding:10px 12px}.proof-flow b{display:inline-flex;width:22px;height:22px;border-radius:50%;background:#2563EB;color:white;align-items:center;justify-content:center;font-size:12.5px;margin-right:6px}.proof-flow span{color:#F8FAFC;font-weight:800;font-size:13.5px}.proof-flow small{display:block;color:#9FB0C7;font-size:11.5px;margin-top:6px}.proof-flow i{color:#64748B;font-style:normal;font-weight:800}.proof-answer{background:#F0FDF4;border:1px solid #BBE3C7;border-radius:14px;padding:20px 22px;margin:20px 0 24px}.proof-answer-head{font-family:Manrope;font-size:15px;font-weight:800;color:#14532D;margin-bottom:14px}.proof-answer-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}.proof-answer-grid>div{background:white;border:1px solid #D9EBDD;border-radius:10px;padding:13px}.proof-answer-grid b{display:block;font-size:13.5px;color:#14532D;margin-bottom:5px}.proof-answer-grid span{display:block;color:#667085;font-size:13px;line-height:1.5}.proof-answer-foot{font-size:12.5px;color:#4B6351;margin-top:13px;padding-top:11px;border-top:1px solid #D9EBDD}.validation-proof{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:14px 0 20px}.validation-proof>div{background:#F8FAFC;border:1px solid #DCE3EC;border-radius:10px;padding:12px}.validation-proof .vp-label{display:block;font-size:10.5px;font-weight:800;letter-spacing:.08em;color:#7A8698;margin-bottom:4px}.validation-proof b{font-family:Manrope;font-size:16px;color:#172033}.proof-grid{margin-top:12px}@media(max-width:900px){.proof-answer-grid{grid-template-columns:1fr 1fr}.validation-proof{grid-template-columns:1fr 1fr}}@media(max-width:650px){.proof-flow{align-items:stretch}.proof-answer-grid,.validation-proof{grid-template-columns:1fr}}
</style>''', unsafe_allow_html=True)

for k,v in {"df":None,"assessment":None,"healing":None,"research":None,"filename":None,"uploaded_at":None,"pending_upload_bytes":None,"pending_upload_name":None,"case_df":None,"case_result":None,"case_id":None,"case_gt":None,"case_upload_name":None,"data_context":None}.items():
    if k not in st.session_state: st.session_state[k]=v

with st.sidebar:
    st.markdown("""<div class="side-brand"><span class="side-mark">SH</span><span class="side-name">SelfHealDS</span><div class="side-caption">Explainable Data Quality Intelligence</div></div>""",unsafe_allow_html=True)

    if "active_page" not in st.session_state:
        st.session_state.active_page = "Upload Your Own Dataset"

    page = st.session_state.active_page
    nav_groups = [
        ("START", [("Sample Datasets",""), ("Real-World Case Studies",""), ("Upload Your Own Dataset","")]),
        ("UNDERSTAND", [("Overview",""), ("Data Profile",""), ("Data Assessment","")]),
        ("HEAL & PROVE", [("Explainability",""), ("Auto-Heal",""), ("Audit Trail","")]),
        ("GOVERN", [("Rule Catalogue",""), ("Data Contract","")]),
        ("RESEARCH", [("Research Evaluation",""), ("Reports & Export",""), ("Why SelfHealDS?","")]),
    ]

    st.markdown('<div class="side-workspace">Workspace</div>',unsafe_allow_html=True)

    nav_index=0
    for group, items in nav_groups:
        st.markdown(f'<div class="side-caption" style="margin:14px 4px 5px">{group}</div>', unsafe_allow_html=True)
        for item, _ in items:
            active = page == item
            if active:
                st.markdown(
                    f'<div class="side-page-active"><span class="side-bullet">●</span>&nbsp;&nbsp;{item}</div>',
                    unsafe_allow_html=True
                )
            else:
                if st.button(item, key=f"nav_group_{nav_index}", use_container_width=True):
                    st.session_state.active_page = item
                    st.rerun()
            nav_index += 1

    st.markdown('<div class="side-divider"></div>',unsafe_allow_html=True)
    st.markdown(
        '<div class="small-note">Local-first processing<br>Deterministic validation<br>Evidence-backed repairs<br>Core pipeline needs no paid AI API</div>',
        unsafe_allow_html=True
    )

df=st.session_state.df

# Dedicated dataset intake page. Uploading is intentionally separate from analysis:
# the file is parsed and assessed only after the explicit Analyze Dataset action.
if page=="Upload Your Own Dataset":
    st.markdown('<div class="section-title">Upload Your Own Dataset</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Upload a structured <b>CSV</b> or plain-text <b>TXT</b> file. CSVs use the structured data-quality pipeline; TXT files use a semantic text-cleaning pipeline that checks cross-sentence contradictions, geographic consistency, spelling, duplicate statements, placeholders and recoverable format errors.</div>',unsafe_allow_html=True)
    st.markdown('<div class="card">',unsafe_allow_html=True)
    uploaded = st.file_uploader("Drag & drop CSV or TXT", type=["csv","txt"], key="dataset_uploader", help="CSV and UTF-8 plain-text files. Maximum supported file size: 50 MB.")
    st.markdown('</div>',unsafe_allow_html=True)
    st.markdown('<div class="small-note" style="margin-top:8px"><b>Accepted:</b> CSV (.csv) and plain text (.txt) · Maximum size: 50 MB · The original upload remains unchanged.</div>',unsafe_allow_html=True)
    if uploaded is not None:
        file_bytes=uploaded.getvalue(); file_size_mb=len(file_bytes)/(1024*1024); suffix=Path(uploaded.name).suffix.lower()
        if file_size_mb>50:
            st.error("This file exceeds the 50 MB upload limit.")
        else:
            try:
                if suffix==".txt":
                    text=file_bytes.decode("utf-8-sig")
                    full_df=text_to_dataframe(text); preview_df=full_df.head(100); context="text_document"
                    c1,c2,c3=st.columns(3); c1.metric("Document",uploaded.name); c2.metric("Lines",f"{len(full_df):,}"); c3.metric("Characters",f"{len(text):,}")
                    st.markdown('<div class="callout"><b>Semantic text mode:</b> SelfHealDS reasons across repeated entity statements and explicit city/state/country relationships. Automatic semantic changes require deterministic reference evidence or repeated-document majority evidence; ambiguous prose is not guessed.</div>',unsafe_allow_html=True)
                else:
                    full_df=pd.read_csv(BytesIO(file_bytes)); preview_df=full_df.head(100); context=None
                    c1,c2,c3=st.columns(3); c1.metric("Dataset",uploaded.name); c2.metric("Rows",f"{len(full_df):,}"); c3.metric("Columns",f"{len(full_df.columns):,}")
                st.markdown('<div class="section-title" style="font-size:17px">Preview</div>',unsafe_allow_html=True)
                st.dataframe(preview_df,use_container_width=True,hide_index=True)
                if st.button("Analyze Dataset",type="primary",use_container_width=True):
                    st.session_state.df=full_df; st.session_state.filename=uploaded.name; st.session_state.uploaded_at=datetime.now(timezone.utc).isoformat(); st.session_state.data_context=context
                    st.session_state.assessment=assess(full_df,context=context); st.session_state.healing=None; st.session_state.research=None; st.session_state.active_page="Overview"; st.rerun()
            except UnicodeDecodeError:
                st.error("Unable to decode this TXT file as UTF-8 text. Save it as UTF-8 and upload again.")
            except Exception as exc:
                st.error(f"Unable to read this file: {exc}")
    if st.session_state.df is not None and st.session_state.filename:
        st.markdown(f'<div class="callout"><b>Analyzed file:</b> {st.session_state.filename}. Continue from Overview.</div>',unsafe_allow_html=True)
    st.stop()


if page=="Sample Datasets":
    st.markdown('<div class="section-title">Sample Datasets</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Controlled demonstration files supplied with SelfHealDS. CSV samples test structured rules; TXT samples emphasize semantic reasoning and context-aware text repair.</div>',unsafe_allow_html=True)
    sample_dir=Path(__file__).resolve().parent / "sample messy datasets"
    csv_paths=sorted(sample_dir.glob("*.csv")); txt_paths=sorted((sample_dir/"text_samples").glob("*.txt"))
    st.markdown('<div class="section-title" style="font-size:18px">Structured CSV samples</div>',unsafe_allow_html=True)
    cols=st.columns(2)
    for i,path in enumerate(csv_paths):
        try:
            sdf=pd.read_csv(path)
            with cols[i%2]:
                st.markdown(f'<div class="card"><h4>{path.stem}</h4><p>{len(sdf):,} rows × {len(sdf.columns):,} columns · Controlled structured-data demo</p></div>',unsafe_allow_html=True)
                if st.button(f"Use {path.name}",key=f"sample_csv_{i}",use_container_width=True):
                    st.session_state.df=sdf; st.session_state.filename=path.name; st.session_state.data_context=None; st.session_state.uploaded_at=datetime.now(timezone.utc).isoformat(); st.session_state.assessment=assess(sdf); st.session_state.healing=None; st.session_state.research=None; st.session_state.active_page="Overview"; st.rerun()
        except Exception as exc: st.error(f"Could not read {path.name}: {exc}")
    st.markdown('<div class="section-title" style="font-size:18px;margin-top:24px">Semantic TXT samples</div>',unsafe_allow_html=True)
    st.markdown('<div class="callout"><b>Text research focus:</b> these examples contain contradictions that require reasoning across statements—not just syntax fixes. The backend uses deterministic geographic knowledge and repeated-entity majority evidence to repair only semantically supported conflicts.</div>',unsafe_allow_html=True)
    cols=st.columns(2)
    for i,path in enumerate(txt_paths):
        try:
            text=path.read_text(encoding="utf-8"); sdf=text_to_dataframe(text); pre=assess(sdf,context="text_document"); errs=sum(1 for x in pre["issues"] if x.severity=="Error")
            with cols[i%2]:
                st.markdown(f'<div class="card"><h4>{path.stem}</h4><p>{len(sdf):,} lines · {len(pre["issues"]):,} detected findings · {errs} semantic/error-level findings</p></div>',unsafe_allow_html=True)
                if st.button(f"Use {path.name}",key=f"sample_txt_{i}",use_container_width=True):
                    st.session_state.df=sdf; st.session_state.filename=path.name; st.session_state.data_context="text_document"; st.session_state.uploaded_at=datetime.now(timezone.utc).isoformat(); st.session_state.assessment=pre; st.session_state.healing=None; st.session_state.research=None; st.session_state.active_page="Overview"; st.rerun()
        except Exception as exc: st.error(f"Could not read {path.name}: {exc}")
    st.markdown('<div class="callout callout-warning"><b>Research distinction:</b> all sample files are intentionally messy controlled demonstrations. They should not be presented as independent real-world evidence.</div>',unsafe_allow_html=True)
    st.stop()


if page=="Real-World Case Studies":
    st.markdown('<div class="section-title">Real-World Case Studies</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Seven documented real-world data-quality cases. Datasets are processed locally from researcher-supplied CSVs; SelfHealDS never downloads a dataset at runtime.</div>',unsafe_allow_html=True)
    case_names=[c["name"] for c in CASE_STUDIES]
    selected=st.selectbox("Select a case study",case_names,key="case_selector")
    case=next(c for c in CASE_STUDIES if c["name"]==selected)
    if st.session_state.case_id != case["id"]:
        st.session_state.case_df=None; st.session_state.case_result=None; st.session_state.case_gt=None; st.session_state.case_upload_name=None
    st.session_state.case_id=case["id"]
    st.markdown(f'<div class="rw-hero"><div class="rw-eyebrow">{case["domain"]} · {case["year"]} · DOCUMENTED CASE</div><h1>{case["name"]}</h1><p>{case["incident"]}</p><div class="rw-meta"><span><b>Organization</b><br>{case["organization"]}</span><span><b>Year</b><br>{case["year"]}</span><span><b>Dataset</b><br>{case["dataset_name"]}</span></div></div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1: st.markdown(f'<div class="rw-card"><h2>What Happened</h2><p>{case["incident"]}</p><h2>Data-Quality Issue</h2><p>{case["issue"]}</p><h2>Real-World Impact</h2><p>{case["impact"]}</p></div>',unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="rw-card"><h2>Dataset & Provenance</h2><p><b>Source:</b> {case["source"]}</p><p><b>Required local file:</b> <code>{case.get("local_filename",case["id"]+".csv")}</code></p><p>{case["availability"]}</p><h2>Research Integrity</h2><p><b>Documented/natural:</b> {case["natural_issue"]}</p><p><b>Controlled/synthetic:</b> {case["controlled_issue"]}</p></div>',unsafe_allow_html=True)
    st.markdown(f'<div class="rw-card"><h2>What SelfHealDS Detects</h2><p>{case["detect"]}</p><h2>What SelfHealDS Repairs</h2><p>{case["repair"]}</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="rw-section">LOCAL DATASET INPUT</div>',unsafe_allow_html=True)
    st.info(f'Upload the required local CSV: {case.get("local_filename",case["id"]+".csv")}. No external download is performed.')
    uploaded=st.file_uploader("Upload case-study CSV",type=["csv"],key=f"rw_upload_{case['id']}")
    gt_uploaded=None; gt_name=case.get("ground_truth_filename")
    if gt_name: gt_uploaded=st.file_uploader(f"Optional documented ground-truth CSV ({gt_name})",type=["csv"],key=f"rw_gt_{case['id']}")
    if uploaded is not None:
        try:
            cdf,_=load_case_dataset(case,uploaded_bytes=uploaded.getvalue(),uploaded_name=uploaded.name)
            st.session_state.case_df=cdf; st.session_state.case_result=None; st.session_state.case_upload_name=uploaded.name
        except Exception as exc: st.error(f"Could not read the uploaded CSV: {exc}")
    elif st.session_state.case_df is None:
        cdf,_=load_case_dataset(case)
        if cdf is not None: st.session_state.case_df=cdf
    if gt_uploaded is not None:
        try: st.session_state.case_gt=pd.read_csv(BytesIO(gt_uploaded.getvalue()),low_memory=False)
        except Exception as exc: st.warning(f"Ground-truth CSV could not be read: {exc}")
    cdf=st.session_state.case_df
    if cdf is not None:
        st.markdown('<div class="rw-section">BEFORE CLEANING</div>',unsafe_allow_html=True)
        m=st.columns(5); facts=[("Rows",f"{len(cdf):,}"),("Columns",f"{len(cdf.columns):,}"),("Cells",f"{cdf.shape[0]*cdf.shape[1]:,}"),("Duplicates",f"{cdf.duplicated().sum():,}"),("Missing",f"{cdf.isna().sum().sum():,}")]
        for x,(lab,val) in zip(m,facts): x.markdown(f'<div class="metric-card"><div class="metric-label">{lab}</div><div class="metric-value">{val}</div></div>',unsafe_allow_html=True)
        st.dataframe(cdf.head(25),use_container_width=True,hide_index=True)
        if st.button("Run SelfHealDS Case Study",type="primary",use_container_width=True):
            try:
                with st.spinner("Running detection, repair and post-repair validation..."): st.session_state.case_result=evaluate_case(case,cdf,st.session_state.case_gt)
            except Exception as exc: st.error(f"Case-study evaluation failed: {exc}")
        result=st.session_state.case_result
        if result:
            st.markdown('<div class="rw-section">SELFHEALDS ANALYSIS</div>',unsafe_allow_html=True)
            issue_df=pd.DataFrame(result["assessment"]["issue_table"])
            st.dataframe(issue_df.head(250),use_container_width=True,hide_index=True)
            # Explicitly surface the rows where the documented case-study problem
            # was repaired. This is the research evidence view, not just a log dump.
            log_df=pd.DataFrame(result["healing"].get("repair_log",[]))
            main_terms={
                "nasa_mars_climate_orbiter":["Unit normalization","Unrecognized measurement value"],
                "holoclean_hospital":["Provider identifier normalization","Invalid phone length"],
                "holoclean_flights":["Invalid flight time","Non-numeric value in numeric field","Case-study numeric repair","Boolean representation inconsistency"],
                "chicago_food_inspections":["Invalid ZIP code","Invalid date","Unknown risk category","Categorical representation inconsistency","Duplicate inspection record"],
                "uci_air_quality":["Documented missing sentinel","Non-numeric sensor value"],
                "uci_online_retail":["Negative unit price","Non-numeric value in quantity field","Invalid date"],
                "uci_adult_census":["Age domain repair","Invalid work-hours value","Missing value"]
            }.get(case["id"],[])
            if not log_df.empty and main_terms:
                key=log_df[log_df["issue_type"].isin(main_terms)].copy()
                if not key.empty:
                    st.markdown('<div class="rw-section">KEY ISSUE ROWS — MAIN CASE PROBLEM RESOLVED</div>',unsafe_allow_html=True)
                    st.markdown('<div class="callout callout-success"><b>Research evidence:</b> these are the specific rows/cells where SelfHealDS applied a repair associated with the documented case-study issue. Original and repaired values are shown side-by-side.</div>',unsafe_allow_html=True)
                    cols=[c for c in ["row","column","issue_type","original_value","repaired_value","rule_id","confidence"] if c in key.columns]
                    st.dataframe(key[cols].head(100),use_container_width=True,hide_index=True)
            st.markdown('<div class="rw-section">AFTER CLEANING</div>',unsafe_allow_html=True); h=result["healing"]
            r1,r2,r3,r4=st.columns(4)
            for col,label,val in [(r1,"Detected findings",result["detected_findings"]),(r2,"Repair events",result["repair_events"]),(r3,"Remaining findings",result["post_repair_findings"]),(r4,"Review rows",result["quarantined_rows"])]: col.metric(label,val)
            q1,q2=st.columns(2); q1.markdown(f'<div class="rw-card"><h2>Before Quality</h2><p>Score <b>{result["before_quality"]["score"]:.2f}</b> · Errors {result["before_quality"]["errors"]} · Warnings {result["before_quality"]["warnings"]}</p></div>',unsafe_allow_html=True); q2.markdown(f'<div class="rw-card"><h2>After Quality</h2><p>Score <b>{result["after_quality"]["score"]:.2f}</b> · Errors {result["after_quality"]["errors"]} · Warnings {result["after_quality"]["warnings"]}</p></div>',unsafe_allow_html=True)
            st.markdown(f'<div class="callout callout-success"><b>Row preservation:</b> input {len(cdf):,} rows → cleaned {len(h["cleaned"]):,} rows. Unresolved rows are retained for case-study transparency and separately listed for review; the general upload workflow keeps its strict quarantine behavior.</div>',unsafe_allow_html=True)
            st.dataframe(h["cleaned"].head(25),use_container_width=True,hide_index=True)
            _dl_filename = case['local_filename'].replace('.csv', '_cleaned.csv')
            st.download_button(
                "Download cleaned case-study CSV",
                data=h["cleaned"].to_csv(index=False).encode("utf-8"),
                file_name=_dl_filename,
                mime="text/csv",
                use_container_width=True,
                key=f"download_cleaned_{case['id']}"
            )
            if len(h["review_required"]):
                st.markdown('<h3 style="font-size:21px">Rows Requiring Review</h3>',unsafe_allow_html=True); st.dataframe(h["review_required"].head(100),use_container_width=True,hide_index=True)
            if h["repair_log"]:
                st.markdown('<h3 style="font-size:21px">Repair Audit</h3>',unsafe_allow_html=True); st.dataframe(pd.DataFrame(h["repair_log"]).head(250),use_container_width=True,hide_index=True)
            gt=result.get("ground_truth",{})
            if gt.get("available"): st.markdown(f'<div class="callout callout-success"><b>Independent ground truth:</b> Precision {gt.get("precision_pct")} · Recall {gt.get("recall_pct")} · F1 {gt.get("f1_pct")} · Repair accuracy {gt.get("repair_accuracy_pct")} over {gt.get("ground_truth_cells")} reference changes.</div>',unsafe_allow_html=True)
            else: st.markdown(f'<div class="callout callout-warning"><b>Ground truth not scored:</b> {gt.get("reason","No independent cell-level reference supplied.")}</div>',unsafe_allow_html=True)
            if st.button("Open cleaned case in main workspace",use_container_width=True):
                st.session_state.df=h["cleaned"]; st.session_state.filename=f"real_world_{case['id']}_cleaned.csv"; st.session_state.uploaded_at=datetime.now(timezone.utc).isoformat(); st.session_state.assessment=assess(cdf,context=case["id"]); st.session_state.healing=h; st.session_state.active_page="Overview"; st.rerun()
    else: st.markdown(f'<div class="callout callout-warning"><b>CSV required:</b> upload <code>{case.get("local_filename",case["id"]+".csv")}</code> above to run this case.</div>',unsafe_allow_html=True)
    st.stop()

if df is None:
    st.markdown('''<div class="hero"><div class="eyebrow">Data quality engineering platform</div><h1>From raw data to<br>trusted analytical data.</h1><p>SelfHealDS is an explainable preprocessing and self-healing framework for structured datasets. It profiles the data, detects structural, syntactic, semantic, unit and statistical defects, explains every finding, performs only evidence-backed repairs, re-validates the result, quarantines unresolved records and preserves an audit trail.</p><div class="hero-line"></div><div class="hero-foot">Built as a research-oriented preprocessing layer before analytics, reporting and machine learning.</div></div>''',unsafe_allow_html=True)
    st.markdown('<div class="section-title">End-to-end methodology</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-copy">The system is designed around a safety principle: a value is never changed merely to make a dataset appear clean.</div>',unsafe_allow_html=True)
    cards=[("01","Profile","Infer structure, semantic field types, missingness, uniqueness and numeric compatibility."),("02","Detect","Identify quality defects across multiple dimensions using explicit validation rules."),("03","Explain","Expose the rule, observed evidence, impact, proposed treatment and confidence."),("04","Heal","Apply deterministic transformations only where the intended correction is sufficiently supported."),("05","Validate","Run the detector again and enforce a strict post-repair validation gate."),("06","Audit","Preserve original value, new value, reason, confidence, rule identifier and final status.")]
    cols=st.columns(3)
    for col,(n,t,dsc) in zip(cols*2,cards): col.markdown(f'<div class="card"><div style="color:#2563EB;font-weight:800;font-size:11px">{n}</div><h4>{t}</h4><p>{dsc}</p></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True); st.markdown('<div class="callout"><b>Start:</b> Upload a CSV from the <b>Upload Your Own Dataset</b> page. The first operation is always a non-destructive pre-repair assessment.</div>',unsafe_allow_html=True); st.markdown('<div class="footer">SelfHealDS · Explainable Data Quality Intelligence</div>',unsafe_allow_html=True); st.stop()

assessment=st.session_state.assessment; q=assessment["quality"]

def metric_note(definition, scope, timestamp):
    st.markdown(f'<div class="metric-validity"><b>Definition:</b> {definition}<br><b>Validity:</b> {scope}<br><b>Generated:</b> {timestamp}</div>', unsafe_allow_html=True)

st.markdown(f'<div class="eyebrow">SelfHealDS / {page}</div>',unsafe_allow_html=True)

# The differentiator ribbon and workflow banner are the "why this is more than
# a cleaned file" pitch. Showing them at full size on every single page (Rule
# Catalogue, Data Contract, Reports & Export, ...) repeated the same pitch
# eleven times per session, which reads as padding rather than proof to a
# reviewer clicking through the app. They now appear at full size only on the
# pages where they orient the reader (Overview, Auto-Heal, Why SelfHealDS?);
# every other page gets the slim breadcrumb above and nothing else here.
_BANNER_PAGES = {"Overview", "Auto-Heal", "Why SelfHealDS?"}
if page in _BANNER_PAGES:
    st.markdown(
        '<div class="research-ribbon">'
        '<div class="research-pill"><strong>Evidence</strong> every repair is justified</div>'
        '<div class="research-pill"><strong>Safety</strong> repairs are re-validated</div>'
        '<div class="research-pill"><strong>Traceability</strong> every change is auditable</div>'
        '<div class="research-pill"><strong>Reproducibility</strong> rules + fingerprint are retained</div>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="proof-flow-card">'
        '<div class="proof-flow-title">THE SELFHEALDS GUARDED HEALING LOOP</div>'
        '<div class="proof-flow">'
        '<div><b>1</b><span>Detect</span><small>Find the problem</small></div><i>→</i>'
        '<div><b>2</b><span>Explain</span><small>Show the evidence</small></div><i>→</i>'
        '<div><b>3</b><span>Root cause</span><small>Separate cause from symptom</small></div><i>→</i>'
        '<div><b>4</b><span>Repair</span><small>Change only when supported</small></div><i>→</i>'
        '<div><b>5</b><span>Re-check</span><small>Re-run the detector</small></div><i>→</i>'
        '<div><b>6</b><span>Validate</span><small>Enforce the safety gate</small></div><i>→</i>'
        '<div><b>7</b><span>Prove</span><small>Keep the audit trail</small></div>'
        '</div></div>', unsafe_allow_html=True
    )

if page=="Overview":
    st.markdown(f'''<div class="hero" style="padding:28px 34px"><h1 style="font-size:30px">Data Quality Control Center</h1><p>Current assessment of <b>{st.session_state.filename}</b>. All figures below are calculated from this dataset and this run; they are not hardcoded benchmark claims.</p><div class="hero-line"></div><div class="hero-foot">Assessment timestamp: {assessment["run_at"]} · Scope: {assessment["scope"]} · Dataset fingerprint: {assessment["summary"]["fingerprint"]}</div></div>''',unsafe_allow_html=True)
    metrics=[("Rows",f'{len(df):,}',"dataset scope"),("Columns",f'{len(df.columns):,}',"dataset scope"),("Quality score",f'{q["score"]:.2f}%',"severity-weighted indicator"),("Errors",f'{q["errors"]:,}',"high-impact findings"),("Warnings",f'{q["warnings"]:,}',"review-level findings"),("Auto-repairable",f'{q["repairable"]:,}',"eligible findings")]
    cols=st.columns(6)
    for c,(a,b,dsc) in zip(cols,metrics): c.markdown(f'<div class="metric-card"><div class="metric-label">{a}</div><div class="metric-value">{b}</div><div class="metric-sub">{dsc}</div></div>',unsafe_allow_html=True)
    metric_note(
        "Quality score = max(0, 100 × (1 − weighted finding penalty / cell count)). "
        f"Active weights are Error={POLICY['quality_error_weight']}, Warning={POLICY['quality_warning_weight']}, Info={POLICY['quality_info_weight']}. "
        "These weights preserve severity ordering and are engineering choices, not learned parameters or an external quality standard.",
        f"Current dataset fingerprint {assessment['summary']['fingerprint']} and assessment run only.",
        assessment["run_at"]
    )
    st.markdown(
        '<div class="proof-answer">'
        '<div class="proof-answer-head">WHY THIS IS MORE THAN “AI CLEANING A FILE”</div>'
        '<div class="proof-answer-grid">'
        '<div><b>AI may suggest a fix.</b><span>SelfHealDS records the evidence that justified the change.</span></div>'
        '<div><b>A cleaner may apply a rule.</b><span>SelfHealDS re-checks the repaired dataset before accepting it.</span></div>'
        '<div><b>An anomaly detector may flag a value.</b><span>SelfHealDS attempts to identify the underlying root cause.</span></div>'
        '<div><b>An automated script may overwrite data.</b><span>SelfHealDS preserves original values and repair provenance.</span></div>'
        '</div>'
        '<div class="proof-answer-foot">Research claim: the contribution is the integrated, evidence-driven closed loop—not the claim that individual cleaning operations have never existed before.</div>'
        '</div>', unsafe_allow_html=True
    )

    st.markdown(
        '<div class="heal-banner"><h3>What SelfHealDS proves</h3>'
        '<p>This run is not just a cleaned file. SelfHealDS records why a value was considered wrong, '
        'what evidence supported the repair, whether the repair improved the data, and whether the change passed validation.</p></div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="proof-grid">'
        f'<div class="proof-card"><div class="proof-num">{len(assessment["issues"]):,}</div><div class="proof-label">Findings inspected</div><div class="proof-copy">Every finding is tied to a rule and evidence.</div></div>'
        f'<div class="proof-card"><div class="proof-num">{q["repairable"]:,}</div><div class="proof-label">Eligible to heal</div><div class="proof-copy">Only findings covered by the active repair policy.</div></div>'
        f'<div class="proof-card"><div class="proof-num">{len(assessment["dimensions"]):,}</div><div class="proof-label">Quality dimensions</div><div class="proof-copy">Structure, validity, semantics, units and statistics.</div></div>'
        f'<div class="proof-card"><div class="proof-num">100%</div><div class="proof-label">Traceable</div><div class="proof-copy">Automatic changes carry an auditable repair record.</div></div>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown("<br>",unsafe_allow_html=True)
    c1,c2=st.columns([1.3,1])
    with c1:
        st.markdown('<div class="section-title">Quality dimensions</div>',unsafe_allow_html=True); st.markdown('<div class="section-copy">Issue counts grouped by the dimension of data quality they affect.</div>',unsafe_allow_html=True)
        dims=[]
        for cat,val in assessment["dimensions"].items(): dims.append({"Dimension":cat,"Findings":val["findings"],"Errors":val["errors"],"Warnings":val["warnings"]})
        st.dataframe(pd.DataFrame(dims),use_container_width=True,hide_index=True)
    with c2:
        st.markdown('<div class="card"><h4>Quality score interpretation</h4><p>The score is a transparent severity-weighted indicator over the current dataset. It is useful for before/after comparison within a run, but it is not an industry-standard quality percentage and must not be presented as one.</p><br><h4>Safety boundary</h4><p>High-confidence normalization can be automated. Ambiguous semantic, malformed or statistically unusual values are retained for review instead of being fabricated.</p></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True); st.markdown('<div class="section-title">Highest-priority findings</div>',unsafe_allow_html=True)
    rows=assessment["issue_table"]
    if rows:
        rows_df=pd.DataFrame(rows)
        # Explicit severity ranking (Error > Warning > Info): a plain
        # alphabetical sort on the severity string would put "Info" ahead of
        # "Warning" (E < I < W), which is not a meaningful priority order.
        rows_df["_severity_rank"]=rows_df["severity"].map(SEVERITY_ORDER).fillna(99)
        rows_df=rows_df.sort_values(["_severity_rank","confidence"],ascending=[True,False]).drop(columns="_severity_rank")
        st.dataframe(rows_df.head(50),use_container_width=True,hide_index=True)
    else: st.markdown('<div class="callout callout-success"><b>No active findings.</b> The uploaded dataset passed the current rule catalogue.</div>',unsafe_allow_html=True)

elif page=="Data Assessment":
    st.markdown('<div class="section-title">Data Assessment</div>',unsafe_allow_html=True); st.markdown('<div class="section-copy">Every finding is tied to a row, column, rule, severity, confidence and repair decision. Select a severity to focus the table.</div>',unsafe_allow_html=True)
    sev=st.selectbox("Severity",["All","Error","Warning","Info"])
    data=assessment["issue_table"]
    if sev!="All": data=[x for x in data if x["severity"]==sev]
    st.dataframe(pd.DataFrame(data),use_container_width=True,hide_index=True)

elif page=="Explainability":
    st.markdown('<div class="section-title">Explainability & Evidence</div>',unsafe_allow_html=True); st.markdown('<div class="section-copy">The framework explains not only what is wrong, but what evidence triggered the rule and why the proposed treatment is or is not safe.</div>',unsafe_allow_html=True)
    issues=assessment["issues"]
    if not issues: st.markdown('<div class="callout callout-success">No findings require explanation.</div>',unsafe_allow_html=True)
    else:
        # Labels are indexed rather than matched back by text: two different
        # findings can render identical label text (e.g. a duplicated column
        # inflating the same row/column/issue_type combination), and
        # labels.index(selected) would then always resolve to the first one.
        options=list(range(len(issues)))
        selected_idx=st.selectbox(
            "Finding", options,
            format_func=lambda idx: f"Row {issues[idx].row} · {issues[idx].column} · {issues[idx].issue_type}"
        )
        i=issues[selected_idx]
        status="Automatic repair permitted" if i.repairable else "Manual review required"
        st.markdown(f'<div class="rule-box"><span class="status {"status-error" if i.severity=="Error" else "status-warning" if i.severity=="Warning" else "status-info"}">{i.severity}</span><span class="status status-info" style="margin-left:6px">{i.category}</span><h3 style="margin:10px 0 3px;font-family:Manrope">{i.issue_type}</h3><p><b>Location:</b> row {i.row}, column {i.column}<br><b>Observed value:</b> {i.value!r}<br><b>Rule:</b> {i.rule_id}<br><b>Evidence score:</b> {i.confidence:.2f} <span style="color:#667085">(not a probability)</span><br><b>Decision:</b> {status}</p></div>',unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            st.markdown(f'<div class="card"><h4>Why is this an error or warning?</h4><p>{i.reason}</p><br><h4>Evidence</h4><p>{i.evidence}</p><br><h4>Why this evidence score?</h4><p>{i.confidence_basis}</p></div>',unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="card"><h4>Proposed treatment</h4><p>{i.proposed_repair}</p><br><h4>Why the system stops</h4><p>{"This repair has sufficient deterministic evidence under the active policy." if i.repairable else "The correct replacement cannot be inferred safely from the available evidence. The record remains eligible for review rather than receiving a fabricated value."}</p></div>',unsafe_allow_html=True)
        if i.issue_type=="Missing value":
            # Canonicalize placeholder tokens first so this illustrative
            # derivation matches what repair_dataset() actually computes
            # (repair.py converts placeholders to NA before deriving the
            # fallback; without this, a column containing e.g. "n/a" strings
            # could show a different, incorrect candidate here).
            fallback,method,fb=missing_fallback(canonicalize_placeholders(df[i.column]),semantic_type(i.column))
            st.markdown(
                f'<div class="callout"><b>Missing-value derivation</b><br>{fb}<br><br>'
                f'<b>Candidate:</b> {fallback!r} &nbsp; <b>Method:</b> {method or "none"}<br>'
                f'<b>Policy:</b> automatic imputation requires at least {POLICY["missing_min_observed"]} usable observations and no more than {POLICY["missing_max_rate"]*100:.0f}% missingness. '
                f'These are declared engineering thresholds, not universal statistical guarantees.</div>',
                unsafe_allow_html=True
            )

elif page=="Auto-Heal":
    st.markdown('<div class="section-title">Automatic Healing & Verification</div>',unsafe_allow_html=True); st.markdown('<div class="section-copy">SelfHealDS does not stop at “fixing” values. It repairs, re-runs the detector, checks for new problems, and keeps an auditable record of every accepted or rejected change.</div>',unsafe_allow_html=True)
    st.markdown('<div class="callout"><b>Repair policy:</b> deterministic transformations are preferred; ambiguous values are not guessed. The validation gate has authority over the final clean dataset.</div>',unsafe_allow_html=True); st.markdown("<br>",unsafe_allow_html=True)
    if st.button("Run automatic healing and post-repair validation",type="primary",use_container_width=True):
        with st.spinner("Applying safe repairs and validating the result..."): st.session_state.healing=heal(df,context=st.session_state.data_context)
    h=st.session_state.healing
    if h:
        m=st.columns(6); vals=[("Before",h["before_quality"]["score"],"quality indicator"),("After",h["after_quality"]["score"],"validated clean data"),("Repairs",len(h["repair_log"]),"audit events"),("Review rows",h["quarantined_rows"],"unresolved errors"),("Remaining",len(h["after_issues"]),"findings after gate"),("Run",h["run_at"][:19].replace("T"," "),"UTC timestamp")]
        for c,(a,b,dsc) in zip(m,vals): c.markdown(f'<div class="metric-card"><div class="metric-label">{a}</div><div class="metric-value" style="font-size:{"17px" if a=="Run" else "25px"}">{b}</div><div class="metric-sub">{dsc}</div></div>',unsafe_allow_html=True)
        metric_note("Before/after quality values are severity-weighted indicators; repairs are audit-event counts; review rows are records excluded by the post-repair Error-level gate.", f"Valid only for dataset fingerprint {assessment['summary']['fingerprint']} and healing run {h['run_at']}.", h["run_at"])
        v=h.get("validation",{})
        vc1,vc2,vc3,vc4=st.columns(4)
        vc1.metric("Errors removed",max(0,v.get("error_reduction",0)))
        vc2.metric("Validated repairs",sum(1 for e in h.get("repair_log",[]) if e.get("validation_status")=="PASSED"))
        vc3.metric("Repair checks failed",v.get("unsafe_repairs",0))
        vc4.metric("Rows preserved",f"{v.get('rows_after',0):,}/{v.get('rows_before',0):,}")
        if v.get("validation_passed"):
            st.success("Self-healing validation passed: Error-level findings did not increase and automatic repairs have no cell-level validation failure.")
        else:
            st.warning("Validation requires attention: review unresolved errors or failed repair checks before treating the output as fully clean.")
        st.markdown(
            f'<div class="before-after">'
            f'<div class="ba-card"><h4>BEFORE</h4><p>{h["before_quality"]["errors"]:,} Error-level findings · quality {h["before_quality"]["score"]:.2f}</p></div>'
            f'<div class="ba-arrow">→</div>'
            f'<div class="ba-card"><h4>AFTER VALIDATION</h4><p>{h["after_quality"]["errors"]:,} Error-level findings · quality {h["after_quality"]["score"]:.2f}</p></div>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown("<br>",unsafe_allow_html=True)
        if h["quarantined_rows"]: st.markdown('<div class="callout callout-warning"><b>Validation gate triggered.</b> Unresolved Error-level records were quarantined and are not part of the validated clean dataset.</div>',unsafe_allow_html=True)
        else: st.markdown('<div class="callout callout-success"><b>Validation gate passed.</b> No unresolved Error-level records remain in the clean output.</div>',unsafe_allow_html=True)
        st.markdown(
            f'<div class="validation-proof">'
            f'<div><span class="vp-label">REPAIR DECISION</span><b>{"ACCEPTED & VERIFIED" if v.get("validation_passed") else "REQUIRES ATTENTION"}</b></div>'
            f'<div><span class="vp-label">ERRORS</span><b>{h["before_quality"]["errors"]:,} → {h["after_quality"]["errors"]:,}</b></div>'
            f'<div><span class="vp-label">VALIDATED CHANGES</span><b>{sum(1 for e in h.get("repair_log",[]) if e.get("validation_status")=="PASSED"):,}</b></div>'
            f'<div><span class="vp-label">AUDIT COVERAGE</span><b>{("100%" if h.get("repair_log") and all(e.get("reason") and e.get("rule_id") for e in h["repair_log"]) else "CHECK")}</b></div>'
            f'</div>', unsafe_allow_html=True
        )
        st.markdown("#### Repair reasoning")
        if h["repair_log"]:
            log_df=pd.DataFrame(h["repair_log"])
            st.dataframe(log_df[["row","column","issue_type","original_value","repaired_value","reason","confidence","confidence_basis","status","rule_id"]],use_container_width=True,hide_index=True)
            st.markdown(
                '<div class="callout"><b>Interpretation rule:</b> a replacement is not justified merely because it makes a value look plausible. '
                'The audit reason must identify the evidence used to derive the replacement. For example, a missing numeric age is filled with the observed median only when the declared minimum-observation and missingness thresholds are satisfied; '
                'the exact observed count, missingness rate and resulting median are recorded in the audit event.</div>',
                unsafe_allow_html=True
            )
        if h.get("root_causes"):
            st.markdown("#### Root-cause view")
            st.markdown('<div class="small-note">Related findings are grouped to show likely upstream causes versus downstream symptoms. This is an explainable grouping heuristic, not causal discovery.</div>',unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(h["root_causes"]),use_container_width=True,hide_index=True)
        st.markdown("#### Validated clean dataset"); st.dataframe(h["cleaned"].head(200),use_container_width=True,hide_index=True)
        if st.session_state.data_context=="text_document":
            cleaned_text=dataframe_to_text(h["cleaned"])
            st.download_button("Download cleaned TXT",cleaned_text.encode("utf-8"),file_name=f"{Path(st.session_state.filename).stem}_cleaned.txt",mime="text/plain",use_container_width=True)
        else:
            st.download_button("Download cleaned CSV",h["cleaned"].to_csv(index=False).encode("utf-8"),file_name=f"{Path(st.session_state.filename).stem}_cleaned.csv",mime="text/csv",use_container_width=True)
        if h["quarantined_rows"]: st.markdown("#### Review-required dataset"); st.dataframe(h["review_required"],use_container_width=True,hide_index=True)

elif page=="Audit Trail":
    st.markdown('<div class="section-title">Audit Trail</div>',unsafe_allow_html=True); st.markdown('<div class="section-copy">Every automatic transformation is recorded with its rule identifier, evidence-based reason, confidence and status.</div>',unsafe_allow_html=True)
    h=st.session_state.healing
    if not h: st.markdown('<div class="callout">Run automatic healing to generate the audit trail.</div>',unsafe_allow_html=True)
    else:
        log=pd.DataFrame(h["repair_log"])
        st.dataframe(log,use_container_width=True,hide_index=True)

elif page=="Data Profile":
    st.markdown('<div class="section-title">Dataset Profile</div>',unsafe_allow_html=True); st.markdown('<div class="section-copy">The profile is generated before repair and forms the structural baseline for this run.</div>',unsafe_allow_html=True)
    st.dataframe(assessment["profile"],use_container_width=True,hide_index=True)
    st.markdown('<div class="callout"><b>Profile metric definitions:</b> missing rate = missing cells / rows; unique rate = distinct non-null values / rows; numeric compatibility = values parseable as numeric / rows. These are descriptive statistics for the current dataset, not external quality standards.</div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True); st.markdown('<div class="section-title">Raw data preview</div>',unsafe_allow_html=True); st.markdown('<div class="section-copy">Original uploaded values. This view is read-only.</div>',unsafe_allow_html=True); st.dataframe(df.head(200),use_container_width=True,hide_index=True)

elif page=="Rule Catalogue":
    st.markdown('<div class="section-title">Rule Catalogue</div>',unsafe_allow_html=True); st.markdown('<div class="section-copy">The active rule catalogue makes the system auditable and gives the research paper a reproducible definition of what constitutes a finding.</div>',unsafe_allow_html=True)
    active_ids={i.issue_type:i.rule_id for i in assessment["issues"]}
    rows=[{"Category":cat,"Issue type":issue,"Rule identifier":active_ids.get(issue,"Catalogue entry; activated when applicable"),"Default action":"Detect; repair only when evidence supports it"} for issue,cat in ISSUE_TAXONOMY.items()]
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    st.markdown("<br>",unsafe_allow_html=True)
    policy_rows=[
        {"Policy parameter":"Age upper boundary","Value":POLICY["age_upper_bound"],"Why it is used":POLICY_RATIONALE["age_upper_bound"]},
        {"Policy parameter":"Phone digit range", "Value":f"{POLICY['phone_min_digits']}–{POLICY['phone_max_digits']}", "Why it is used":POLICY_RATIONALE["phone_range"]},
        {"Policy parameter":"Automatic imputation ceiling","Value":f"{POLICY['missing_max_rate']*100:.0f}% missingness", "Why it is used":POLICY_RATIONALE["missing_rate"]},
        {"Policy parameter":"Minimum observed values for fallback","Value":POLICY["missing_min_observed"],"Why it is used":POLICY_RATIONALE["missing_min_observed"]},
        {"Policy parameter":"Outlier screening fence","Value":f"{POLICY['outlier_iqr_multiplier']} × IQR","Why it is used":POLICY_RATIONALE["iqr_multiplier"]},
        {"Policy parameter":"Quality score weights","Value":f"E={POLICY['quality_error_weight']}, W={POLICY['quality_warning_weight']}, I={POLICY['quality_info_weight']}","Why it is used":POLICY_RATIONALE["quality_weights"]},
    ]
    st.markdown('<div class="section-title" style="font-size:17px">Policy parameters and rationale</div>',unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(policy_rows),use_container_width=True,hide_index=True)
    st.markdown('<div class="callout"><b>Scope limitation:</b> no finite rule set can detect every possible data-quality problem. The system reports only findings covered by its declared rules and reference vocabularies. Every threshold is a documented policy decision rather than an unstated claim of universal truth.</div>',unsafe_allow_html=True)

elif page=="Data Contract":
    st.markdown('<div class="section-title">Inferred Data Contract</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-copy">A machine-readable contract inferred from the uploaded schema and active rule catalogue. It documents what SelfHealDS currently believes each field represents and which constraints are enforceable.</div>',unsafe_allow_html=True)
    rows=[]
    for c in df.columns:
        stype=assessment["profile"].loc[assessment["profile"]["column"]==c,"semantic_type"].iloc[0]
        constraints={
            "age":"non-negative; upper plausibility boundary 120",
            "phone":"7–15 digits after harmless formatting normalization",
            "email":"minimum local-part@domain structure",
            "date":"parseable calendar date; ISO output",
            "quantity":"non-negative numeric quantity",
            "weight":"recognized unit; canonical kg",
            "length":"recognized unit; canonical m",
            "temperature":"recognized C/F unit; canonical C",
            "city":"reference vocabulary when available",
            "country":"reference vocabulary / aliases",
            "currency_numeric":"deterministically parseable numeric value",
            "generic":"generic missingness, duplication and whitespace checks",
        }.get(stype,"generic validation")
        rows.append({"Column":c,"Inferred type":stype,"Enforced constraints":constraints,"Nulls":int(df[c].isna().sum()),"Distinct":int(df[c].nunique(dropna=True))})
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    st.markdown('<div class="callout"><b>Research boundary:</b> inferred semantics are evidence, not ground truth. The system exposes its inferred contract so researchers can inspect, reproduce and challenge the assumptions used during validation.</div>',unsafe_allow_html=True)
    metric_note("Field constraints are derived from column names, observed values and the declared rule catalogue.", f"Valid for dataset fingerprint {assessment['summary']['fingerprint']} and ruleset loaded by this application.", assessment["run_at"])

elif page=="Research Evaluation":
    st.markdown('<div class="section-title">Research Evaluation</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-copy">This module is the experimental layer of SelfHealDS. It creates deterministic labelled corruptions, measures detection against ground truth, evaluates repair safety, compares against a minimal baseline, and tests multiple corruption levels. Every metric is tied to a dataset fingerprint, evaluation version, seed and timestamp.</div>',unsafe_allow_html=True)
    if st.button("Run research benchmark",type="primary",use_container_width=True):
        with st.spinner("Running controlled corruption sweep, baseline comparison and repair safety analysis..."): st.session_state.research=run_research_suite(df)
    r=st.session_state.research
    if r:
        ev=r["evaluation"]
        st.markdown(f'<span class="research-badge">Reproducible run</span> &nbsp; {r["generated_at"]} &nbsp; · &nbsp; fingerprint {r["fingerprint"]} &nbsp; · &nbsp; ruleset {r["ruleset_version"]}',unsafe_allow_html=True)
        readiness=r["evaluation"].get("research_readiness",{})
        if readiness.get("status")=="recommended":
            st.markdown(
                f'<div class="callout callout-success"><b>Benchmark readiness:</b> {readiness.get("reference_rows")} reference rows meet the '
                f'{readiness.get("minimum_recommended_rows")}-row minimum recommended for this local experimental protocol.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="callout callout-warning"><b>Small-sample warning:</b> this run has {readiness.get("reference_rows",0)} reference rows. '
                f'{readiness.get("reason","Use a larger dataset for publication-grade results.")}</div>',
                unsafe_allow_html=True
            )
        st.markdown(
            '<div class="callout"><b>Hybrid detection architecture</b><br>'
            'SelfHealDS combines deterministic domain, format, unit and semantic rules with '
            '<b>Isolation Forest</b> for statistical anomaly detection. The ML layer flags unusual '
            'observations only; it never changes values or overrides deterministic validation. '
            'This separation preserves explainability and prevents statistical unusualness from being treated as proof of corruption.</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="small-note"><b>ML configuration:</b> {r["ml_model"]["name"]}; '
            f'{r["ml_model"]["version"]}; random seed {r["ml_model"]["random_state"]}. '
            'The model is a statistical anomaly flagger only. Its decision score is not a probability, and ML flags are never automatically repaired. '
            'The fixed configuration exists for reproducibility; it is not presented as an empirically optimized hyperparameter setting.</div>',
            unsafe_allow_html=True
        )
        st.markdown('<br>',unsafe_allow_html=True)
        m=st.columns(8)
        vals=[("Precision",ev["precision"],"TP / all positive detections"),("Recall",ev["recall"],"injected corruptions detected"),("F1",ev["f1"],"harmonic mean of precision and recall"),("Repair accuracy",ev["repair_accuracy_pct"],"correct among attempted repairs"),("Safe repair coverage",ev.get("safe_repair_coverage_pct"),"correct repairs across cases with a known safe target"),("False-alarm rate",ev.get("clean_false_positive_rate_pct"),"false alarms on data that was never corrupted"),("Detection ms",r["detection_latency_ms"],"local execution time"),("Healing ms",r["healing_latency_ms"],"local execution time")]
        for c,(name,val,desc) in zip(m,vals):
            shown="N/A" if val is None else f"{val}{'%' if name in ['Precision','Recall','F1','Repair accuracy','Safe repair coverage','False-alarm rate'] else ''}"
            c.markdown(f'<div class="research-metric-card"><div class="metric-label">{name}</div><div class="metric-accent"></div><div class="metric-value">{shown}</div><div class="metric-sub">{desc}</div></div>',unsafe_allow_html=True)
        metric_note("Precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2PR/(P+R). A metric is N/A when its denominator is zero. Repair accuracy is correct automatic repairs divided by all automatic repair attempts. Safe repair coverage is correct automatic repairs divided by benchmark cases with an explicitly known safe target; non-inferable cases are never counted as successful repairs. False-alarm rate (also called clean-input false-positive rate) is false positives divided by cells that were never deliberately corrupted.", f"Valid only for fingerprint {r['fingerprint']}, evaluation version {ev['evaluation_version']}, seed {ev['seed']} and benchmark method shown below.", r["generated_at"])
        st.markdown(f'<div class="small-note"><b>Detection counts:</b> TP {ev["true_positives"]} &nbsp;|&nbsp; FP {ev["false_positives"]} &nbsp;|&nbsp; FN {ev["false_negatives"]} &nbsp;|&nbsp; labelled cases {ev["actual_corruption_cases"]} &nbsp;·&nbsp; <b>Repair counts:</b> {ev.get("repair_correct",0)} correct / {ev.get("repair_attempts",0)} attempted &nbsp;·&nbsp; {ev.get("repairable_cases",0)} cases had a known safe target. A high precision value with low recall is therefore not a perfect detector; it means the system is selective and misses some labelled corruptions.</div>',unsafe_allow_html=True)
        st.markdown('<div class="callout"><b>How to read safe repair coverage:</b> 0% means the benchmark contained cases for which an exact safe target was known, but SelfHealDS made zero correct automatic repairs among those cases in this run. It does <b>not</b> mean 0% of the entire dataset was cleaned. Non-inferable cases, such as a three-digit phone where the missing digits are unknown, are never treated as successful repairs.</div>',unsafe_allow_html=True)
        st.markdown('<br>',unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            st.markdown('<div class="card"><h4>Experimental design</h4><p>The uploaded reference is copied without first running SelfHealDS on it. Controlled corruptions are injected only where relevant semantic fields exist, and each injection records its original value, corrupted value, issue type and expected repair where known. Natural pre-existing findings are removed from the prediction set before scoring to avoid treating existing defects as benchmark false positives. This prevents natural defects in the uploaded file from being mistaken for benchmark errors.</p><br><h4>Threats to validity</h4><p>Synthetic corruption does not perfectly reproduce production errors. Reference vocabularies are finite. Therefore benchmark results support reproducibility for the tested scenarios but do not establish universal performance.</p></div>',unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card"><h4>Safety interpretation</h4><p>The system separates detection from authorization to repair. A detected anomaly may be repairable, review-only, or blocked when evidence is insufficient. The post-repair detector is the final validation gate.</p><br><h4>Publication rule</h4><p>Only report metrics generated from documented datasets, seeds, corruption levels and baselines. Never copy illustrative percentages into the paper.</p></div>',unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            ob=ev.get("obvious_case_metrics",{})
            st.markdown(f'<div class="card"><h4>Obvious corruption cases</h4><p>{ev.get("obvious_cases",0)} labelled cases using explicit domain, format or semantic violations. Recall: <b>{"N/A" if ob.get("recall_pct") is None else str(ob.get("recall_pct"))+"%"}</b>.</p></div>',unsafe_allow_html=True)
        with c2:
            pl=ev.get("plausible_case_metrics",{})
            st.markdown(f'<div class="card"><h4>Plausible corruption cases</h4><p>{ev.get("plausible_cases",0)} labelled cases where the corrupted value remains structurally valid. Recall: <b>{"N/A" if pl.get("recall_pct") is None else str(pl.get("recall_pct"))+"%"}</b>. These cases test whether the system avoids pretending to know what it cannot infer.</p></div>',unsafe_allow_html=True)
        st.markdown('#### Baseline comparison')
        st.markdown('<div class="small-note" style="margin-bottom:8px">In plain terms: how much better is SelfHealDS than the simplest possible checker (just missing values and exact duplicates)?</div>',unsafe_allow_html=True)
        base=r["baseline"]
        bt=pd.DataFrame([{"Method":k,"Precision (%)":v["precision_pct"],"Recall (%)":v["recall_pct"],"F1 (%)":v["f1_pct"],"Predictions":v["predictions"]} for k,v in base.items() if isinstance(v,dict)])
        st.dataframe(bt,use_container_width=True,hide_index=True)
        st.markdown('<div class="small-note">The baseline intentionally checks only missing values and exact duplicate rows. It is a transparent internal baseline for incremental coverage, not a reproduction of a named commercial system.</div>',unsafe_allow_html=True)

        st.markdown('#### Component ablation')
        st.markdown('<div class="small-note" style="margin-bottom:8px">In plain terms: if we switched off the machine-learning layer, would the deterministic rules alone still catch most problems? ("Ablation" means testing a system with one part removed.)</div>',unsafe_allow_html=True)
        st.dataframe(r["ablation"],use_container_width=True,hide_index=True)
        st.markdown('<div class="small-note">Ablation asks what the deterministic rule layer contributes relative to the full hybrid architecture. Isolation Forest is reported separately when its anomaly taxonomy cannot be mapped one-to-one onto deterministic corruption labels; forcing incompatible labels into one F1 score would be methodologically misleading.</div>',unsafe_allow_html=True)

        st.markdown('#### Corruption-level sensitivity')
        st.markdown('<div class="small-note" style="margin-bottom:8px">In plain terms: does accuracy hold up as we make the test data progressively messier (5% -> 30% corrupted)?</div>',unsafe_allow_html=True)
        st.dataframe(r["sweep_summary"],use_container_width=True,hide_index=True)
        st.markdown(
            f'<div class="callout"><b>Experimental design:</b> the sweep uses {len(r["experimental_protocol"]["seeds"])} fixed seeds per corruption level. '
            f'The levels ({", ".join(str(int(x))+"%" for x in r["experimental_protocol"]["corruption_rates_pct"])}) are controlled stress-test scenarios, not claims about real-world corruption prevalence. '
            'For each trial, row selection is seeded and the injected issue, original value and expected repair are retained as ground truth. '
            'Mean and sample standard deviation expose variation across trials; they are not population confidence intervals.</div>',
            unsafe_allow_html=True
        )
        sweep_csv=r["sweep"].to_csv(index=False).encode()
        st.download_button("Download sensitivity results (CSV)",sweep_csv,"selfhealds_sensitivity.csv","text/csv",use_container_width=True)
        st.download_button("Download research run manifest (JSON)",json.dumps(to_jsonable(r),indent=2,default=str).encode(),"selfhealds_research_run.json","application/json",use_container_width=True)

        st.markdown('#### Error-category performance')
        st.markdown('<div class="small-note" style="margin-bottom:8px">In plain terms: which kinds of mistakes (missing values, bad dates, wrong units...) is SelfHealDS best and worst at catching?</div>',unsafe_allow_html=True)
        st.dataframe(r["category_breakdown"],use_container_width=True,hide_index=True)
        st.markdown('<div class="small-note">Category metrics are calculated only for injected issue labels with known ground truth. They should not be generalized beyond the tested corruption operators.</div>',unsafe_allow_html=True)

        st.markdown('#### Heterogeneous sample-suite comparison')
        st.markdown('<div class="small-note" style="margin-bottom:8px">In plain terms: does the same detector hold up across very different kinds of files (customer data, e-commerce, HR, health)?</div>',unsafe_allow_html=True)
        st.dataframe(r["bundled_suite"],use_container_width=True,hide_index=True)
        st.markdown(
            '<div class="small-note">The bundled suite keeps datasets separate rather than pooling their scores. '
            'These files are controlled project examples, so they demonstrate cross-domain behavior but are not equivalent to independently curated public benchmark datasets.</div>',
            unsafe_allow_html=True
        )
        gt=ground_truth_table(ev["ground_truth"])
        if not gt.empty:
            st.markdown('#### Ground-truth cases for this run')
            st.dataframe(gt,use_container_width=True,hide_index=True)
        st.markdown('#### Detector output on benchmark data')
        st.dataframe(pd.DataFrame([{"Row":i.row,"Column":i.column,"Issue":i.issue_type,"Category":i.category,"Severity":i.severity,"Rule confidence":i.confidence,"Rule":i.rule_id} for i in ev["detected"]]),use_container_width=True,hide_index=True)

        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown('#### Optional: compare against a general-purpose LLM')
        st.markdown(
            '<div class="callout"><b>In plain terms:</b> this runs the exact same corrupted-data test used above, '
            'but hands the messy file to a general-purpose AI model with a single cleaning prompt instead of SelfHealDS, '
            'then scores it with the identical precision/recall/repair-accuracy metrics. This is the most direct, apples-to-apples '
            'version of "why not just ask an AI to clean it?" — a live, same-conditions number instead of an assertion. '
            'It never runs automatically: it requires your own API key, is not stored anywhere, and costs you API usage.</div>',
            unsafe_allow_html=True
        )
        llm_key = st.text_input("Anthropic API key (used only for this run, not saved)", type="password", key="llm_baseline_key")
        if st.button("Run LLM baseline comparison", use_container_width=True, key="run_llm_baseline"):
            if not llm_baseline.is_configured(llm_key):
                st.warning("No API key supplied — enter one above, or set the ANTHROPIC_API_KEY environment variable, to run this comparison.")
            else:
                with st.spinner("Sending the benchmark file to the model and scoring its response..."):
                    st.session_state.llm_baseline_result = llm_baseline.evaluate_llm_baseline(
                        df, corruption_rate=0.10, seed=42, api_key=llm_key
                    )
        lr = st.session_state.get("llm_baseline_result")
        if lr:
            if not lr.get("available"):
                st.markdown(f'<div class="callout callout-warning"><b>Comparison not available:</b> {lr.get("reason")}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="research-badge">LLM baseline run</span> &nbsp; {lr["generated_at"]} &nbsp; · &nbsp; model {lr["model"]}', unsafe_allow_html=True)
                comparison_rows = [
                    {"Metric": "Precision (%)", "SelfHealDS": ev["precision"], f'{lr["model"]} (single prompt)': lr["precision_pct"]},
                    {"Metric": "Recall (%)", "SelfHealDS": ev["recall"], f'{lr["model"]} (single prompt)': lr["recall_pct"]},
                    {"Metric": "F1 (%)", "SelfHealDS": ev["f1"], f'{lr["model"]} (single prompt)': lr["f1_pct"]},
                    {"Metric": "Repair accuracy (%)", "SelfHealDS": ev["repair_accuracy_pct"], f'{lr["model"]} (single prompt)': lr["repair_accuracy_pct"]},
                ]
                st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
                st.markdown(
                    f'<div class="small-note"><b>Method:</b> {lr["method"]}<br><b>Caveat:</b> {lr["caveats"]}</div>',
                    unsafe_allow_html=True
                )
                with st.expander("Raw model response (preview)"):
                    st.code(lr.get("raw_response_preview", ""), language="json")

elif page=="Why SelfHealDS?":
    st.markdown("<div class=\"why-hero\"><div class=\"eyebrow\" style=\"color:#60A5FA\">RESEARCH POSITIONING</div><h1>Why SelfHealDS?</h1><p>SelfHealDS is designed as a reproducible, evidence-driven data-quality system: detect the problem, explain the evidence, identify a likely root cause, repair only when policy permits, verify the result, and keep the repair history.</p></div>",unsafe_allow_html=True)
    st.markdown('<div class="section-title">The difference</div><div class="section-copy">A general AI can suggest a cleaned answer. SelfHealDS is built as a governed quality-control protocol: every automatic change is evidence-backed, validated after repair, traceable, and reproducible.</div>',unsafe_allow_html=True)
    st.markdown('<div class="callout callout-warning"><b>Do not use the claim “AI cannot do this.”</b> A general AI can perform many individual cleaning tasks. The defensible research question is whether a dedicated system can make those operations <b>repeatable, evidence-backed, safe, root-cause-aware and auditable at dataset scale</b>.</div>',unsafe_allow_html=True)
    st.markdown('<div class="flow"><div class="flow-step">Detect</div><div class="flow-arrow">→</div><div class="flow-step">Explain</div><div class="flow-arrow">→</div><div class="flow-step">Root cause</div><div class="flow-arrow">→</div><div class="flow-step">Repair</div><div class="flow-arrow">→</div><div class="flow-step">Re-check</div><div class="flow-arrow">→</div><div class="flow-step">Validate</div><div class="flow-arrow">→</div><div class="flow-step">Prove</div></div>',unsafe_allow_html=True)
    cards=[("Evidence-backed repair","Every finding carries a rule, evidence and an evidence-strength score. The score is not presented as a probability."),("Root-cause view","Related findings are grouped so downstream symptoms such as outliers can be linked to upstream causes such as unit or semantic inconsistencies."),("Do-no-harm boundary","High-confidence repairs can be automated; uncertain cases remain reviewable instead of receiving invented values."),("Closed-loop validation","The repaired data is scanned again. Repairs that still leave an Error at the same cell are marked failed."),("Full provenance","Each automatic change records original value, new value, rule, reason, confidence and validation status."),("Reproducible research","Dataset fingerprints, ruleset versions, seeds and benchmark protocols are retained for experiments and reports.")]
    cols=st.columns(3)
    for j,(title,body) in enumerate(cards):
        with cols[j%3]: st.markdown(f'<div class="why-card"><h3>{title}</h3><p>{body}</p></div>',unsafe_allow_html=True)
    st.markdown('<div class="rw-section">What makes the claim testable</div>',unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([{"Metric":"Detection precision / recall / F1","Purpose":"Find labelled errors without flagging clean cells."},{"Metric":"Repair accuracy","Purpose":"Match known ground-truth repairs when a target exists."},{"Metric":"False-repair rate","Purpose":"Measure unnecessary changes to clean values."},{"Metric":"Post-repair Error count","Purpose":"Verify that healing actually reduces unresolved errors."},{"Metric":"Validation failure / rollback rate","Purpose":"Measure the safety boundary of automatic repair."},{"Metric":"Audit completeness","Purpose":"Verify that every change has traceable evidence."}]),use_container_width=True,hide_index=True)
    st.markdown(
        '<div class="callout callout-success"><b>The research question:</b> Can automated data cleaning be made '
        'reproducible and safe enough to prove not only that a value changed, but why it changed and whether the change helped?</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="proof-grid">'
        '<div class="proof-card"><div class="proof-num">1</div><div class="proof-label">General AI</div><div class="proof-copy">Can propose a correction from a prompt, but the interaction itself is not a reproducible quality-control protocol.</div></div>'
        '<div class="proof-card"><div class="proof-num">2</div><div class="proof-label">SelfHealDS</div><div class="proof-copy">Uses declared rules, evidence, policies and dataset fingerprints so the same run can be inspected and repeated.</div></div>'
        '<div class="proof-card"><div class="proof-num">3</div><div class="proof-label">Safety</div><div class="proof-copy">Repairs are re-checked and uncertain cases can remain for review rather than being invented.</div></div>'
        '<div class="proof-card"><div class="proof-num">4</div><div class="proof-label">Proof</div><div class="proof-copy">The research evaluation measures detection, repair accuracy, false repairs and validation failures.</div></div>'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="callout"><b>Important research discipline:</b> SelfHealDS does not claim that no previous system or AI has ever performed any individual cleaning operation. Novelty must be established through your literature review. The research contribution is the integrated, evidence-driven closed loop and its measurable safety, provenance and evaluation protocol.</div>',unsafe_allow_html=True)

elif page=="Reports & Export":
    st.markdown('<div class="section-title">Reports & Export</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Generate a reproducible quality passport, research manifest and validated data exports from the current dataset and healing run. Every report carries the dataset fingerprint and execution context needed to interpret its numbers.</div>',unsafe_allow_html=True)

    h=st.session_state.healing
    passport={
        "system":"SelfHealDS",
        "ruleset_version":RULESET_VERSION,
        "generated_at":datetime.now(timezone.utc).isoformat(),
        "dataset":st.session_state.filename,
        "fingerprint":assessment["summary"]["fingerprint"],
        "assessment_run":assessment["run_at"],
        "assessment_scope":assessment["scope"],
        "pre_repair_quality":assessment["quality"],
        "quality_dimensions":assessment["dimensions"],
        "rule_count":len(ISSUE_TAXONOMY),
        "confidence_definition":"Rule-level evidence score; not a calibrated probability.",
        "policy_parameters":POLICY,
        "policy_rationale":POLICY_RATIONALE,
        "ml_configuration":assessment.get("ml_model",{})
    }
    if h:
        passport["healing"]={
            "run_at":h["run_at"],"before_quality":h["before_quality"],
            "after_quality":h["after_quality"],"repair_events":len(h["repair_log"]),
            "review_rows":h["quarantined_rows"],"remaining_findings":len(h["after_issues"])
        }
    if st.session_state.research:
        passport["research_run"]={
            "generated_at":st.session_state.research["generated_at"],
            "evaluation_version":st.session_state.research["evaluation"]["evaluation_version"],
            "seed":st.session_state.research["evaluation"]["seed"],
            "metric_validity":st.session_state.research["metric_validity"]
        }
    passport["metric_validity"]="All values are snapshots valid for the displayed dataset fingerprint and execution timestamp; changing the dataset or rules requires a new run."

    c1,c2=st.columns(2)
    with c1:
        st.markdown('<div class="card"><h4>Quality Passport</h4><p>A machine-readable snapshot of schema assumptions, quality indicators, policy parameters, model configuration and healing status.</p></div>',unsafe_allow_html=True)
        st.download_button("Download quality passport (JSON)",json.dumps(passport,indent=2,default=str).encode(),"selfhealds_quality_passport.json","application/json",use_container_width=True)
    with c2:
        st.markdown('<div class="card"><h4>Research Manifest</h4><p>Contains dataset identity, experiment configuration and metric validity metadata. Run the Research Evaluation module first to populate research results.</p></div>',unsafe_allow_html=True)
        if st.session_state.research:
            st.download_button("Download research manifest (JSON)",json.dumps(to_jsonable(st.session_state.research),indent=2,default=str).encode(),"selfhealds_research_manifest.json","application/json",use_container_width=True)
        else:
            st.button("Research run not available",disabled=True,use_container_width=True)

    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="font-size:17px">Validated data exports</div>',unsafe_allow_html=True)
    if not h:
        st.markdown('<div class="callout callout-warning">Run automatic healing before exporting. This prevents the application from presenting unvalidated data as clean.</div>',unsafe_allow_html=True)
    else:
        c1,c2=st.columns(2)
        with c1:
            b=BytesIO(); h["cleaned"].to_csv(b,index=False)
            st.download_button("Download validated clean CSV",b.getvalue(),"selfhealds_cleaned.csv","text/csv",type="primary",use_container_width=True)
            st.markdown('<div class="small-note">Contains only records that pass the post-repair Error-level validation gate.</div>',unsafe_allow_html=True)
        with c2:
            b=BytesIO(); h["review_required"].to_csv(b,index=False)
            st.download_button("Download review-required CSV",b.getvalue(),"selfhealds_review_required.csv","text/csv",use_container_width=True)
            st.markdown('<div class="small-note">Preserves unresolved records with machine-readable reasons instead of silently discarding them.</div>',unsafe_allow_html=True)

        st.markdown("<br>",unsafe_allow_html=True)
        audit_payload={
            "system":"SelfHealDS","dataset":st.session_state.filename,
            "fingerprint":assessment["summary"]["fingerprint"],
            "assessment_run":assessment["run_at"],"healing_run":h["run_at"],
            "repair_events":h["repair_log"],"post_repair_findings":len(h["after_issues"]),
            "policy_parameters":POLICY,"policy_rationale":POLICY_RATIONALE,
            "metric_validity":"Snapshot valid only for this dataset fingerprint and run."
        }
        st.download_button("Download audit trail (JSON)",json.dumps(audit_payload,indent=2,default=str).encode(),"selfhealds_audit.json","application/json",use_container_width=True)

st.markdown('<div class="footer">SelfHealDS · Explainable Data Quality Intelligence · Every metric is run-specific and timestamped</div>',unsafe_allow_html=True)
