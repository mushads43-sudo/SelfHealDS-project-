from dataclasses import dataclass
from typing import Any

@dataclass
class Issue:
    row: Any
    column: str
    issue_type: str
    category: str
    severity: str
    value: Any
    reason: str
    evidence: str
    proposed_repair: str
    repairable: bool
    confidence: float  # evidence score; not a calibrated probability
    rule_id: str
    confidence_basis: str = ""
    repaired: bool = False
    new_value: Any = None

ISSUE_TAXONOMY = {
    "Missing value":"Missingness", "Placeholder value":"Missingness",
    "Duplicate record":"Uniqueness", "Duplicate column name":"Structural integrity",
    "Constant column":"Structural integrity", "Inconsistent column type":"Type consistency",
    "Non-numeric value in numeric field":"Type consistency", "Negative age":"Domain validity",
    "Impossible age":"Domain validity", "Negative phone number":"Domain validity",
    "Invalid phone length":"Domain validity", "Phone formatting inconsistency":"Format consistency",
    "Malformed email":"Format validity", "Invalid date":"Format validity",
    "Inconsistent date format":"Format consistency", "Invalid time":"Format validity", "Boolean representation inconsistency":"Format consistency", "Invalid ZIP code":"Format validity", "Unknown risk category":"Domain validity", "Categorical representation inconsistency":"Format consistency", "Negative unit price":"Domain validity", "Non-numeric age":"Type consistency", "Invalid work-hours value":"Domain validity", "Age domain repair":"Domain validity", "Case-study numeric repair":"Type consistency", "Unrecognized measurement value":"Unit consistency", "Whitespace inconsistency":"Format consistency",
    "Negative quantity":"Domain validity", "Statistical outlier":"Statistical anomaly",
    "Mixed measurement units":"Unit consistency", "Unrecognized measurement value":"Unit consistency",
    "City-country semantic inconsistency":"Semantic consistency", "Unknown city":"Semantic consistency",
    "Unknown country":"Semantic consistency", "Near-match categorical value":"Semantic consistency",
    "ML statistical anomaly":"Statistical anomaly",
    "Invalid provider identifier":"Domain validity", "Cross-record consistency conflict":"Semantic consistency",
    "Invalid flight time":"Domain validity", "Duplicate inspection record":"Uniqueness",
    "Documented missing sentinel":"Missingness",
    "Duplicate text line":"Uniqueness", "Text whitespace inconsistency":"Format consistency",
    "Contextual spelling error":"Semantic consistency", "Text placeholder value":"Missingness",
    "Obfuscated email format":"Format validity", "Text geographic contradiction":"Semantic consistency",
    "Cross-sentence semantic contradiction":"Semantic consistency",
}

def issue_category(issue_type: str) -> str:
    return ISSUE_TAXONOMY.get(issue_type, "General data quality")
