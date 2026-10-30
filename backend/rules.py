import re
import pandas as pd
import numpy as np

# Single source of truth for the placeholder-token vocabulary. Previously this
# set was independently duplicated in detector.py and repair.py; keeping one
# definition here guarantees detection and repair (and any illustrative UI
# derivation) always agree on what counts as a placeholder.
PLACEHOLDERS = {"n/a","na","none","null","nan","missing","unknown","-","--","?","not available","not_applicable"}

def is_placeholder_token(v):
    """True when v is one of the declared placeholder tokens (e.g. 'n/a', '?').

    Used by the semantic-type-specific validators (phone/email/date/country/...)
    so a placeholder cell is reported once, as a Missingness finding, instead of
    also tripping a second, misleading "invalid format" finding on the same
    cell (e.g. '?' in a country column previously produced both a
    'Placeholder value' finding AND an unrelated 'Unknown country' finding).
    """
    return isinstance(v,str) and v.strip().lower() in PLACEHOLDERS

def canonicalize_placeholders(series):
    """Return a copy of series with placeholder tokens replaced by pandas NA.

    This mirrors the canonicalization repair.py performs before computing a
    missing-value fallback, so any other caller (e.g. an illustrative UI
    derivation) that wants to reproduce the same fallback candidate must run
    values through this function first rather than re-implementing the token
    match.
    """
    def _clean(v):
        if isinstance(v, str) and v.strip().lower() in PLACEHOLDERS:
            return pd.NA
        return v
    return series.map(_clean)

CITY_COUNTRY = {
    "mumbai":"India","delhi":"India","new delhi":"India","pune":"India","bangalore":"India",
    "bengaluru":"India","hyderabad":"India","chennai":"India","kolkata":"India","ahmedabad":"India",
    "jaipur":"India","surat":"India","lucknow":"India","indore":"India","nagpur":"India",
    "paris":"France","london":"United Kingdom","manchester":"United Kingdom","berlin":"Germany",
    "tokyo":"Japan","osaka":"Japan","singapore":"Singapore","dubai":"United Arab Emirates",
    "new york":"United States","los angeles":"United States","chicago":"United States",
    "san francisco":"United States","toronto":"Canada","sydney":"Australia","melbourne":"Australia",
    "moscow":"Russia","rome":"Italy","madrid":"Spain","beijing":"China","shanghai":"China",
}
COUNTRY_ALIASES = {
    "usa":"United States","us":"United States","u.s.":"United States","united states of america":"United States",
    "uk":"United Kingdom","u.k.":"United Kingdom","uae":"United Arab Emirates","ind":"India",
}
KNOWN_COUNTRIES = set(CITY_COUNTRY.values()) | set(COUNTRY_ALIASES.values())

# A broad, general-purpose reference vocabulary used ONLY to judge whether a
# value in a "country" field is a recognizable country name at all. This is
# intentionally separate from CITY_COUNTRY (a small, curated demo mapping used
# for city<->country cross-field consistency). Using the tiny demo mapping as
# the "is this a real country" check was a correctness bug: it caused any real
# dataset containing common countries such as "Mexico", "Netherlands" or
# "Philippines" to be flagged as an "Unknown country" on essentially every row,
# drowning genuine findings in noise. This list covers the UN member states
# under their common English short names, plus a few common CSV-export
# variants (e.g. hyphenated "United-States").
WORLD_COUNTRIES = {
    "afghanistan","albania","algeria","andorra","angola","antigua and barbuda","argentina","armenia",
    "australia","austria","azerbaijan","bahamas","bahrain","bangladesh","barbados","belarus","belgium",
    "belize","benin","bhutan","bolivia","bosnia and herzegovina","botswana","brazil","brunei","bulgaria",
    "burkina faso","burundi","cabo verde","cambodia","cameroon","canada","central african republic","chad",
    "chile","china","colombia","comoros","congo","costa rica","croatia","cuba","cyprus","czechia",
    "czech republic","denmark","djibouti","dominica","dominican republic","ecuador","egypt","el salvador",
    "equatorial guinea","eritrea","estonia","eswatini","ethiopia","fiji","finland","france","gabon",
    "gambia","georgia","germany","ghana","greece","grenada","guatemala","guinea","guinea-bissau","guyana",
    "haiti","honduras","hungary","iceland","india","indonesia","iran","iraq","ireland","israel","italy",
    "jamaica","japan","jordan","kazakhstan","kenya","kiribati","kosovo","kuwait","kyrgyzstan","laos",
    "latvia","lebanon","lesotho","liberia","libya","liechtenstein","lithuania","luxembourg","madagascar",
    "malawi","malaysia","maldives","mali","malta","marshall islands","mauritania","mauritius","mexico",
    "micronesia","moldova","monaco","mongolia","montenegro","morocco","mozambique","myanmar","namibia",
    "nauru","nepal","netherlands","new zealand","nicaragua","niger","nigeria","north korea",
    "north macedonia","norway","oman","pakistan","palau","palestine","panama","papua new guinea",
    "paraguay","peru","philippines","poland","portugal","qatar","romania","russia","rwanda",
    "saint kitts and nevis","saint lucia","saint vincent and the grenadines","samoa","san marino",
    "sao tome and principe","saudi arabia","senegal","serbia","seychelles","sierra leone","singapore",
    "slovakia","slovenia","solomon islands","somalia","south africa","south korea","south sudan","spain",
    "sri lanka","sudan","suriname","sweden","switzerland","syria","taiwan","tajikistan","tanzania",
    "thailand","timor-leste","togo","tonga","trinidad and tobago","tunisia","turkey","turkmenistan",
    "tuvalu","uganda","ukraine","united arab emirates","united kingdom","united states","uruguay",
    "uzbekistan","vanuatu","vatican city","venezuela","vietnam","yemen","zambia","zimbabwe",
    "hong kong","macau","puerto rico",
}


def normalize_country_token(v):
    """Normalize a raw country string for lookup: collapse hyphens/underscores
    to spaces and lowercase. This alone fixes common CSV-export spellings such
    as 'United-States' (UCI Adult) not matching 'United States'."""
    return re.sub(r"[\s_\-]+", " ", str(v).strip()).strip().lower()


def is_known_country(v):
    norm = normalize_country_token(v)
    resolved = COUNTRY_ALIASES.get(norm, norm)
    return resolved in WORLD_COUNTRIES or resolved.title() in KNOWN_COUNTRIES or norm in {c.lower() for c in KNOWN_COUNTRIES}
UNIT_MAP = {
    "weight": {"mg": .000001, "g": .001, "gram": .001, "grams": .001, "kg": 1.0, "kgs": 1.0,
               "kilogram":1.0, "kilograms":1.0, "lb":.45359237, "lbs":.45359237},
    "length": {"mm": .001, "cm": .01, "m": 1.0, "meter":1.0, "meters":1.0, "km":1000.0,
               "kilometer":1000.0, "kilometers":1000.0, "in":.0254, "inch":.0254, "ft":.3048, "feet":.3048},
    "temperature": {"c":("c",lambda x:x), "°c":("c",lambda x:x), "celsius":("c",lambda x:x),
                    "f":("c",lambda x:(x-32)*5/9), "°f":("c",lambda x:(x-32)*5/9),
                    "fahrenheit":("c",lambda x:(x-32)*5/9)},
}

def norm_col(c):
    s=str(c).strip()
    # Split camelCase/PascalCase boundaries (e.g. "ProviderNumber" ->
    # "Provider Number", "DepTime" -> "Dep Time") before lowercasing, so
    # token-based semantic matching below sees separate words instead of one
    # fused lowercase string.
    s=re.sub(r"(?<=[a-z0-9])(?=[A-Z])"," ",s)
    return re.sub(r"[^a-z0-9]+","_",s.lower()).strip("_")

def semantic_type(col):
    """Infer a semantic field type from a column name.

    Matching is done against whole underscore-delimited tokens of the
    normalized column name (e.g. "customer_age" -> {"customer","age"}),
    never raw substring containment. Raw substring containment previously
    caused false matches such as "Destination" (contains the substring
    "nation") being classified as a country field, and would similarly
    misfire on ordinary words like "average", "package" or "message" that
    happen to contain "age". A handful of genuinely multi-word keys (e.g.
    "years_old", "e_mail", "contact_number") are matched against the full
    normalized string, since they are phrases rather than single tokens.
    """
    c=norm_col(col)
    tokens=set(c.split("_"))

    def has(*keys):
        return any(k in tokens for k in keys)

    if has("age") or "years_old" in c: return "age"
    if has("phone","mobile","telephone") or "contact_number" in c: return "phone"
    if has("email") or "e_mail" in c: return "email"
    if has("city","town"): return "city"
    if has("country","nation"): return "country"
    # Distinguish true calendar dates/timestamps from time-of-day fields.
    # Flight benchmarks use HHMM numeric fields and UCI Air Quality uses HH.MM.SS;
    # treating these as dates creates false errors such as 500 -> 1970-01-01.
    if c in {"deptime","arrtime","crsdeptime","crsarrtime","time"} or c.endswith("_time") and c not in {"timestamp"}:
        return "time"
    if has("date","dob","birth","timestamp"): return "date"
    if has("weight","mass"): return "weight"
    if has("height","length","distance","width","depth"): return "length"
    if has("temperature") or c in {"temp","temperature_c","temperature_f"}: return "temperature"
    if has("price","salary","amount","revenue","cost","income","balance"): return "currency_numeric"
    if has("quantity","qty","units") or c in {"count","counts"} or c.endswith("_count") or c.endswith("_counts"): return "quantity"
    return "generic"

def parse_numeric(v):
    if pd.isna(v): return np.nan
    if isinstance(v,(int,float,np.integer,np.floating)): return float(v)
    s=str(v).strip().replace(",","")
    s=re.sub(r"^[₹$€£]\s*", "", s)
    try: return float(s)
    except: return np.nan

def unit_parse(v, kind):
    if pd.isna(v): return None
    s=str(v).strip().lower().replace(" ","")
    m=re.fullmatch(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))([a-z°]+)?",s)
    if not m: return None
    n=float(m.group(1)); u=m.group(2)
    if not u: return (n,None)
    if kind=="temperature" and u in UNIT_MAP["temperature"]: return (n,u)
    if kind!="temperature" and u in UNIT_MAP[kind]: return (n,u)
    return None


# Research-visible policy parameters. These are engineering decisions, not
# universal statistical truths; the UI and audit trail expose their rationale.
POLICY = {
    "age_upper_bound": 120,
    "phone_min_digits": 7,
    "phone_max_digits": 15,
    "missing_max_rate": 0.20,
    "missing_min_observed": 5,
    "outlier_iqr_multiplier": 1.5,
    "quality_error_weight": 1.0,
    "quality_warning_weight": 0.35,
    "quality_info_weight": 0.10,
}

POLICY_RATIONALE = {
    "age_upper_bound": "A conservative domain boundary for ordinary human-age fields. It is a validation policy, not a claim about every possible age-related dataset.",
    "phone_range": "A broad generic international identifier range. Because country-specific numbering plans differ, this is a structural plausibility check rather than proof that a number is assigned.",
    "missing_rate": "Automatic imputation is limited to at most one-fifth missingness. Above this level, the observed sample may be too incomplete for a simple single-value fallback to be defensible without domain context.",
    "missing_min_observed": "At least five observed values are required before a single-value fallback is used. This is a conservative minimum for a small-sample engineering rule, not a statistical guarantee.",
    "iqr_multiplier": "The conventional 1.5×IQR exploratory fence is used to flag unusual values. It identifies potential outliers; it does not establish that a value is erroneous.",
    "quality_weights": "The quality score is an internal comparison indicator. Errors receive full penalty, warnings 0.35 and informational findings 0.10 to preserve severity ordering. These weights are not learned and are not an industry standard.",
}

def missing_fallback(series, st):
    """Return (candidate, method, evidence) for a conservative missing-value repair."""
    usable=series[~series.isna()]
    vals=usable.astype("string").str.strip()
    vals=vals[vals.ne("")]
    n=len(vals)
    if n < POLICY["missing_min_observed"]:
        return None,None,f"Only {n} usable observations are available; at least {POLICY['missing_min_observed']} are required by the active policy."
    if st in {"age","currency_numeric","quantity"}:
        numeric=vals.map(parse_numeric).dropna()
        if len(numeric) < POLICY["missing_min_observed"]:
            return None,None,f"Only {len(numeric)} observations are numerically parseable; at least {POLICY['missing_min_observed']} are required."
        candidate=float(numeric.median())
        ordered=sorted(float(x) for x in numeric)
        return candidate,"median",(
            f"The fallback is the observed median of {len(ordered)} parseable values. "
            f"After sorting the observed values, the median is {candidate:g}; "
            f"the median is used because it is resistant to extreme values and does not require assuming a normal distribution."
        )
    mode=vals.mode()
    if len(mode)==0:
        return None,None,"No repeated non-empty categorical value exists from which to derive a mode."
    candidate=str(mode.iloc[0])
    freq=int((vals==candidate).sum())
    return candidate,"mode",(
        f"The fallback is the observed mode '{candidate}', appearing {freq} time(s) among {n} usable values. "
        f"Mode is used for categorical/string fields because it preserves an actually observed category rather than inventing a new one."
    )
