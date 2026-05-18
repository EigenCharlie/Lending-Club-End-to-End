"""Audit Lending Club variables against bounded Paper 4 research lanes.

This script is intentionally unversioned: it is a standing audit utility, not
another Paper 4 wave. It profiles the cleaned parquet, checks the raw CSV header
and selected raw servicing fields, links variables to the local Lending Club
dictionary, and writes a compact claim/evidence decision matrix.
"""

from __future__ import annotations

import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import duckdb
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
INTERIM_PARQUET = ROOT / "data/interim/lending_club_cleaned.parquet"
RAW_CSV = ROOT / "data/raw/Loan_status_2007-2020Q3.csv"
LC_DICTIONARY = ROOT / "docs/LCDataDictionary.xlsx"
TABLE_DIR = ROOT / "reports/paper_material/paper4/tables"
NOTES_DIR = ROOT / "reports/paper_material/paper4/notes"
DOCS_DIR = ROOT / "docs/research"
RUN_DATE = "2026-05-18"

VARIABLE_INVENTORY = TABLE_DIR / f"paper4_data_frontier_variable_inventory_{RUN_DATE}.csv"
LANE_DECISIONS = TABLE_DIR / f"paper4_data_frontier_lane_decisions_{RUN_DATE}.csv"
RAW_PROFILE = TABLE_DIR / f"paper4_data_frontier_raw_servicing_profile_{RUN_DATE}.csv"
MEMO = DOCS_DIR / f"paper4_data_frontier_research_proposal_{RUN_DATE}.md"
NOTE_MEMO = NOTES_DIR / f"paper4_data_frontier_research_proposal_{RUN_DATE}.md"


LANES: dict[str, dict[str, Any]] = {
    "ifrs9_sicr": {
        "title": "IFRS9/SICR and ECL",
        "target_claim": "Upgrade from IFRS9-inspired proxy to bounded lifetime-ECL/SICR stress diagnostic.",
        "cleaned_need": [
            "id",
            "loan_amnt",
            "term",
            "int_rate",
            "installment",
            "issue_d",
            "loan_status",
            "default_flag",
            "lgd",
            "lgd_months_since_issue",
            "lgd_is_mature_24m",
            "fico_range_low",
            "fico_range_high",
            "last_fico_range_low",
            "last_fico_range_high",
        ],
        "raw_need": [
            "funded_amnt",
            "out_prncp",
            "total_pymnt",
            "total_rec_prncp",
            "total_rec_int",
            "recoveries",
            "collection_recovery_fee",
            "last_pymnt_d",
            "last_pymnt_amnt",
            "last_credit_pull_d",
            "hardship_flag",
            "hardship_status",
            "hardship_dpd",
            "hardship_loan_status",
            "debt_settlement_flag",
        ],
        "hard_blockers": [
            "No monthly account performance panel",
            "No contractual days-past-due history before default",
            "No borrower-level macro scenario path",
        ],
        "decision": "bounded_experiment",
    },
    "fair_lending_proxy": {
        "title": "Fair lending and protected-attribute governance",
        "target_claim": "Keep legal fair-lending claim false; improve proxy/source governance.",
        "cleaned_need": [
            "addr_state",
            "zip_code",
            "annual_inc",
            "dti",
            "home_ownership",
            "purpose",
            "emp_title",
            "grade",
            "sub_grade",
            "verification_status",
        ],
        "raw_need": ["zip_code", "addr_state"],
        "hard_blockers": [
            "No race, ethnicity, sex or age",
            "No surname for BISG",
            "No full address or Census tract",
        ],
        "decision": "proxy_only",
    },
    "cate_policy_value": {
        "title": "CATE and causal policy value",
        "target_claim": "Reopen only as observational price/tightness sensitivity, not policy value.",
        "cleaned_need": [
            "int_rate",
            "grade",
            "sub_grade",
            "term",
            "verification_status",
            "fico_range_low",
            "fico_range_high",
            "annual_inc",
            "dti",
            "loan_status",
            "default_flag",
            "lgd",
        ],
        "raw_need": ["funded_amnt", "total_pymnt", "recoveries"],
        "hard_blockers": [
            "Only accepted/funded loans are visible",
            "No randomized pricing or approval instrument",
            "No rejected-applicant counterfactuals in the retained project data",
        ],
        "decision": "diagnostic_only",
    },
    "online_conformal": {
        "title": "Deployable online conformal and source holdouts",
        "target_claim": "Improve source-family coverage diagnostics; keep deployable claim false.",
        "cleaned_need": [
            "issue_d",
            "grade",
            "sub_grade",
            "addr_state",
            "zip_code",
            "annual_inc",
            "dti",
            "purpose",
            "default_flag",
            "lgd",
        ],
        "raw_need": ["last_credit_pull_d", "hardship_flag"],
        "hard_blockers": [
            "No true external source distribution",
            "No production feedback loop",
            "Only historical retrospective issue-month evaluation",
        ],
        "decision": "bounded_experiment",
    },
    "spo_dfl": {
        "title": "Differentiable SPO/DFL",
        "target_claim": "Prototype in an isolated environment only; do not integrate into main champion.",
        "cleaned_need": [
            "loan_amnt",
            "int_rate",
            "installment",
            "grade",
            "sub_grade",
            "default_flag",
            "lgd",
            "annual_inc",
            "dti",
        ],
        "raw_need": ["total_pymnt", "recoveries"],
        "hard_blockers": [
            "Differentiable optimization dependency and scaling risk",
            "No reason to disturb the main CRPTO pipeline",
        ],
        "decision": "isolated_prototype",
    },
    "dla_adp": {
        "title": "Exact DLA/ADP and Bellman optimality",
        "target_claim": "Keep exact optimality false; improve rollout simulator features.",
        "cleaned_need": [
            "issue_d",
            "term",
            "loan_amnt",
            "grade",
            "sub_grade",
            "default_flag",
            "lgd",
            "lgd_months_since_issue",
            "addr_state",
        ],
        "raw_need": ["last_pymnt_d", "last_credit_pull_d", "out_prncp", "hardship_flag"],
        "hard_blockers": [
            "No monthly borrower state trajectory",
            "No actual sequential decision logs",
            "No realized action policy history beyond accepted loans",
        ],
        "decision": "rollout_only",
    },
    "cvar_oce": {
        "title": "CVaR/OCE as champion replacement",
        "target_claim": "Use CVaR/OCE as tail-risk challenger, not champion replacement.",
        "cleaned_need": [
            "loan_amnt",
            "int_rate",
            "installment",
            "grade",
            "sub_grade",
            "default_flag",
            "lgd",
            "lgd_months_since_issue",
        ],
        "raw_need": ["total_pymnt", "total_rec_prncp", "total_rec_int", "recoveries"],
        "hard_blockers": [
            "Existing paired replay did not beat economic champion",
            "No new cap or return floor has changed the objective",
        ],
        "decision": "tail_challenger_only",
    },
}

RAW_PROFILE_COLUMNS = sorted(
    {
        col
        for lane in LANES.values()
        for col in lane["raw_need"]
        if col not in {"zip_code", "addr_state"}
    }
    | {
        "loan_status",
        "issue_d",
        "last_pymnt_d",
        "next_pymnt_d",
        "hardship_type",
        "hardship_reason",
        "hardship_amount",
        "hardship_start_date",
        "hardship_end_date",
        "payment_plan_start_date",
        "orig_projected_additional_accrued_interest",
        "hardship_payoff_balance_amount",
        "hardship_last_payment_amount",
    }
)


def _xlsx_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    strings: list[str] = []
    for si in root.findall("x:si", ns):
        parts = [node.text or "" for node in si.findall(".//x:t", ns)]
        strings.append("".join(parts))
    return strings


def _read_xlsx_first_sheet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path, sheet_name="LoanStats")
    except ImportError:
        pass

    with zipfile.ZipFile(path) as zf:
        shared = _xlsx_shared_strings(zf)
        sheet_name = "xl/worksheets/sheet1.xml"
        root = ET.fromstring(zf.read(sheet_name))
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows: list[list[str]] = []
    for row in root.findall(".//x:sheetData/x:row", ns):
        values: list[str] = []
        last_col = 0
        for cell in row.findall("x:c", ns):
            ref = cell.attrib.get("r", "")
            match = re.match(r"([A-Z]+)", ref)
            col_idx = 1
            if match:
                col_idx = 0
                for char in match.group(1):
                    col_idx = col_idx * 26 + (ord(char) - ord("A") + 1)
            while last_col + 1 < col_idx:
                values.append("")
                last_col += 1
            last_col = col_idx
            raw_value = cell.findtext("x:v", default="", namespaces=ns)
            if cell.attrib.get("t") == "s" and raw_value:
                values.append(shared[int(raw_value)])
            else:
                values.append(raw_value)
        rows.append(values)
    header, *body = rows
    width = len(header)
    normalized_body = []
    for body_row in body:
        if len(body_row) < width:
            body_row = body_row + [""] * (width - len(body_row))
        normalized_body.append(body_row[:width])
    return pd.DataFrame(normalized_body, columns=header)


def load_dictionary() -> dict[str, str]:
    df = _read_xlsx_first_sheet(LC_DICTIONARY)
    cols = {str(c).lower(): c for c in df.columns}
    var_col = cols.get("loanstatsnew") or cols.get("loanstatnew") or df.columns[0]
    desc_col = cols.get("description") or df.columns[1]
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        variable = str(row.get(var_col, "")).strip()
        desc = str(row.get(desc_col, "")).strip()
        if variable and variable.lower() != "nan":
            out[variable] = desc if desc.lower() != "nan" else ""
    return out


def read_raw_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def sql_literal(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def profile_cleaned_columns(columns: list[str]) -> dict[str, dict[str, Any]]:
    cols = sorted(set(columns))
    if not cols:
        return {}
    query_parts = []
    for col in cols:
        quoted = sql_literal(col)
        query_parts.append(
            "SELECT "
            f"'{col}' AS variable, "
            "count(*) AS row_count, "
            f"count({quoted}) AS non_null_count, "
            f"count(DISTINCT {quoted}) AS distinct_count "
            f"FROM read_parquet('{INTERIM_PARQUET.as_posix()}')"
        )
    con = duckdb.connect(database=":memory:")
    df = con.execute(" UNION ALL ".join(query_parts)).fetchdf()
    con.close()
    profiles: dict[str, dict[str, Any]] = {}
    for row in df.to_dict("records"):
        row_count = int(row["row_count"])
        non_null = int(row["non_null_count"])
        profiles[row["variable"]] = {
            "row_count": row_count,
            "non_null_count": non_null,
            "missing_count": row_count - non_null,
            "missing_rate": (row_count - non_null) / row_count if row_count else math.nan,
            "distinct_count": int(row["distinct_count"]),
        }
    return profiles


def dataset_summary() -> dict[str, Any]:
    con = duckdb.connect(database=":memory:")
    source = f"read_parquet('{INTERIM_PARQUET.as_posix()}')"
    summary = con.execute(
        f"""
        SELECT
            count(*) AS row_count,
            count(DISTINCT id) AS distinct_id,
            avg(default_flag) AS default_rate,
            min(try_strptime(issue_d, '%b-%Y')) AS min_issue_month,
            max(try_strptime(issue_d, '%b-%Y')) AS max_issue_month,
            count(DISTINCT issue_d) AS issue_months,
            count(DISTINCT addr_state) AS states,
            count(DISTINCT zip_code) AS zip3,
            count(DISTINCT grade) AS grades,
            count(DISTINCT sub_grade) AS sub_grades,
            avg(lgd) AS mean_lgd,
            avg(CASE WHEN lgd_is_mature_24m = 1 THEN lgd END) AS mean_lgd_mature_24m
        FROM {source}
        """
    ).fetchone()
    status = con.execute(
        f"""
        SELECT loan_status, count(*) AS n
        FROM {source}
        GROUP BY loan_status
        ORDER BY n DESC
        LIMIT 20
        """
    ).fetchdf()
    application = con.execute(
        f"""
        SELECT application_type, count(*) AS n
        FROM {source}
        GROUP BY application_type
        ORDER BY n DESC
        """
    ).fetchdf()
    con.close()
    keys = [
        "row_count",
        "distinct_id",
        "default_rate",
        "min_issue_month",
        "max_issue_month",
        "issue_months",
        "states",
        "zip3",
        "grades",
        "sub_grades",
        "mean_lgd",
        "mean_lgd_mature_24m",
    ]
    return {
        "summary": dict(zip(keys, summary, strict=True)),
        "loan_status": status.to_dict("records"),
        "application_type": application.to_dict("records"),
    }


def profile_raw_columns(raw_header: list[str]) -> list[dict[str, Any]]:
    usecols = [c for c in RAW_PROFILE_COLUMNS if c in raw_header]
    counters: dict[str, dict[str, Any]] = {
        col: {
            "non_null_count": 0,
            "distinct_sample": set(),
            "top_values": Counter(),
        }
        for col in usecols
    }
    row_count = 0
    for chunk in pd.read_csv(
        RAW_CSV,
        usecols=usecols,
        chunksize=200_000,
        low_memory=False,
    ):
        row_count += len(chunk)
        for col in usecols:
            series = chunk[col]
            non_null = series.dropna()
            counters[col]["non_null_count"] += int(non_null.shape[0])
            if non_null.dtype == object:
                vals = non_null.astype(str)
                counters[col]["top_values"].update(vals[vals != ""].head(20).tolist())
                if len(counters[col]["distinct_sample"]) < 50:
                    counters[col]["distinct_sample"].update(
                        vals.drop_duplicates().head(50).tolist()
                    )
            else:
                if len(counters[col]["distinct_sample"]) < 50:
                    counters[col]["distinct_sample"].update(
                        non_null.drop_duplicates().head(50).astype(str).tolist()
                    )
    rows: list[dict[str, Any]] = []
    for col in usecols:
        top = "; ".join(f"{k}:{v}" for k, v in counters[col]["top_values"].most_common(5))
        rows.append(
            {
                "variable": col,
                "source": "raw_csv",
                "row_count": row_count,
                "non_null_count": counters[col]["non_null_count"],
                "missing_count": row_count - counters[col]["non_null_count"],
                "missing_rate": (row_count - counters[col]["non_null_count"]) / row_count
                if row_count
                else "",
                "distinct_sample_n": len(counters[col]["distinct_sample"]),
                "top_values_sample": top,
            }
        )
    return rows


def write_variable_inventory(
    parquet_cols: list[tuple[str, str]],
    raw_header: list[str],
    dictionary: dict[str, str],
    cleaned_profiles: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    lane_by_var: dict[str, list[str]] = defaultdict(list)
    for lane_name, lane in LANES.items():
        for col in lane["cleaned_need"]:
            lane_by_var[col].append(f"{lane_name}:cleaned")
        for col in lane["raw_need"]:
            lane_by_var[col].append(f"{lane_name}:raw")

    raw_set = set(raw_header)
    rows: list[dict[str, Any]] = []
    for col, dtype in parquet_cols:
        profile = cleaned_profiles.get(col, {})
        rows.append(
            {
                "variable": col,
                "source": "cleaned_parquet",
                "dtype": dtype,
                "present_in_cleaned": True,
                "present_in_raw": col in raw_set,
                "row_count": profile.get("row_count", ""),
                "non_null_count": profile.get("non_null_count", ""),
                "missing_rate": profile.get("missing_rate", ""),
                "distinct_count": profile.get("distinct_count", ""),
                "lane_flags": ";".join(lane_by_var.get(col, [])),
                "dictionary_description": dictionary.get(col, ""),
            }
        )
    cleaned_set = {c for c, _ in parquet_cols}
    for col in raw_header:
        if col not in cleaned_set and col in lane_by_var:
            rows.append(
                {
                    "variable": col,
                    "source": "raw_csv_only",
                    "dtype": "",
                    "present_in_cleaned": False,
                    "present_in_raw": True,
                    "row_count": "",
                    "non_null_count": "",
                    "missing_rate": "",
                    "distinct_count": "",
                    "lane_flags": ";".join(lane_by_var.get(col, [])),
                    "dictionary_description": dictionary.get(col, ""),
                }
            )
    pd.DataFrame(rows).to_csv(VARIABLE_INVENTORY, index=False)
    return rows


def write_lane_decisions(
    parquet_columns: set[str],
    raw_columns: set[str],
    cleaned_profiles: dict[str, dict[str, Any]],
    raw_profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw_profile_by_col = {row["variable"]: row for row in raw_profiles}
    rows: list[dict[str, Any]] = []
    for lane_name, lane in LANES.items():
        cleaned_present = [c for c in lane["cleaned_need"] if c in parquet_columns]
        cleaned_missing = [c for c in lane["cleaned_need"] if c not in parquet_columns]
        raw_present = [c for c in lane["raw_need"] if c in raw_columns]
        raw_missing = [c for c in lane["raw_need"] if c not in raw_columns]
        raw_non_null = []
        for c in raw_present:
            profile = raw_profile_by_col.get(c)
            if profile:
                raw_non_null.append(f"{c}={profile['non_null_count']}")
        min_cleaned_availability = ""
        if cleaned_present:
            rates = [
                1 - float(cleaned_profiles[c]["missing_rate"])
                for c in cleaned_present
                if c in cleaned_profiles and cleaned_profiles[c]["missing_rate"] != ""
            ]
            min_cleaned_availability = min(rates) if rates else ""
        rows.append(
            {
                "lane": lane_name,
                "title": lane["title"],
                "decision_after_data_audit": lane["decision"],
                "target_claim": lane["target_claim"],
                "cleaned_present": ";".join(cleaned_present),
                "cleaned_missing": ";".join(cleaned_missing),
                "raw_present": ";".join(raw_present),
                "raw_missing": ";".join(raw_missing),
                "min_cleaned_availability": min_cleaned_availability,
                "raw_profile_non_null": ";".join(raw_non_null),
                "hard_blockers": "; ".join(lane["hard_blockers"]),
            }
        )
    pd.DataFrame(rows).to_csv(LANE_DECISIONS, index=False)
    return rows


def fmt_pct(value: Any) -> str:
    if value in ("", None):
        return ""
    try:
        return f"{100 * float(value):.2f}%"
    except (TypeError, ValueError):
        return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            text = str(val).replace("\n", " ").replace("|", "/")
            vals.append(text)
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_memo(
    summary: dict[str, Any], lane_rows: list[dict[str, Any]], raw_profiles: list[dict[str, Any]]
) -> None:
    s = summary["summary"]
    status_rows = summary["loan_status"][:8]
    app_rows = summary["application_type"]
    compact_lane_rows = []
    for row in lane_rows:
        compact_lane_rows.append(
            {
                "lane": row["lane"],
                "decision": row["decision_after_data_audit"],
                "cleaned_missing": row["cleaned_missing"] or "none",
                "raw_adds": row["raw_present"] or "none",
                "hard_blockers": row["hard_blockers"],
            }
        )
    raw_compact = [
        {
            "variable": row["variable"],
            "non_null": row["non_null_count"],
            "missing_rate": fmt_pct(row["missing_rate"]),
            "top_values_sample": row["top_values_sample"],
        }
        for row in raw_profiles
        if row["variable"]
        in {
            "recoveries",
            "total_pymnt",
            "last_pymnt_d",
            "hardship_flag",
            "hardship_status",
            "hardship_dpd",
            "debt_settlement_flag",
        }
    ]
    text = f"""# Paper 4 Data Frontier Research Proposal - {RUN_DATE}

## Purpose

This memo re-audits the seven Paper 4 lanes that were parked or blocked after
loop closure. The audit uses `data/interim/lending_club_cleaned.parquet`, the
raw Lending Club CSV header and selected raw servicing fields, and
`docs/LCDataDictionary.xlsx`.

The goal is not to reopen the old wave loop. The goal is to decide which lanes
can support one bounded experiment and which remain blocked with the current
data.

## Dataset Surface

| metric | value |
|---|---:|
| cleaned rows | {s["row_count"]} |
| cleaned columns | {pq.ParquetFile(INTERIM_PARQUET).metadata.num_columns} |
| distinct loans | {s["distinct_id"]} |
| issue months | {s["issue_months"]} |
| issue range | {s["min_issue_month"]} to {s["max_issue_month"]} |
| states | {s["states"]} |
| zip3 prefixes | {s["zip3"]} |
| default rate | {fmt_pct(s["default_rate"])} |
| mean LGD | {fmt_pct(s["mean_lgd"])} |
| mean LGD mature 24m | {fmt_pct(s["mean_lgd_mature_24m"])} |

Top loan statuses:

{markdown_table(status_rows, ["loan_status", "n"])}

Application types:

{markdown_table(app_rows, ["application_type", "n"])}

## Lane Decisions After Data Audit

{markdown_table(compact_lane_rows, ["lane", "decision", "cleaned_missing", "raw_adds", "hard_blockers"])}

## Raw Servicing Fields Worth Knowing

{markdown_table(raw_compact, ["variable", "non_null", "missing_rate", "top_values_sample"])}

## External Research Triangulation

| Lane | Source | Implication For Paper 4 |
|---|---|---|
| IFRS9/SICR | IFRS Foundation IFRS 9 official page: https://www.ifrs.org/issued-standards/list-of-standards/ifrs-9-financial-instruments/ | IFRS9 is the right accounting reference for expected credit losses, but Paper 4 still lacks contractual servicing and macro scenario infrastructure. |
| IFRS9/SICR | Competing-risks survival for lifetime ECL: https://www.sciencedirect.com/science/article/pii/S095741742400472X | The best bounded upgrade is a lifetime/default-timing diagnostic, not a full accounting-compliance claim. |
| Fair lending | CFPB BISG proxy methodology: https://github.com/cfpb/proxy-methodology | BISG needs surname plus geocoding. Lending Club exposes zip3/state but no surname or tract, so legal fair-lending stays false. |
| Fair lending | Zhang, "Assessing Fair Lending Risks Using Race/Ethnicity Proxies": https://pubsonline.informs.org/doi/10.1287/mnsc.2016.2579 | Proxy-based disparity estimation is a real research lane, but Paper 4 has insufficient protected-attribute proxy inputs. |
| CATE | DoWhy assumptions paper: https://www.microsoft.com/en-us/research/publication/dowhy-addressing-challenges-in-expressing-and-validating-causal-assumptions/ | Causal estimates need explicit assumptions and refutations; prediction-style validation is not enough. |
| CATE | EconML CausalForestDML docs: https://www.pywhy.org/EconML/_autosummary/econml.dml.CausalForestDML.html | A high-rate-within-grade CATE screen is technically possible, but only as sensitivity/diagnostic evidence. |
| Online conformal | Adaptive conformal inference: https://arxiv.org/abs/2106.00170 | ACI is relevant for distribution shift and online coverage, but Paper 4 still has retrospective history rather than production feedback. |
| Online conformal | Multi-Distribution Robust Conformal Prediction: https://arxiv.org/abs/2601.02998 | MDCP-style uniform source coverage is the right direction for source-family holdouts. |
| SPO/DFL | Smart "Predict, then Optimize": https://arxiv.org/abs/1710.08005 | SPO+ is relevant because CRPTO has an optimization decision downstream of predictions. |
| SPO/DFL | PyEPO SPOPlus docs: https://khalil-research.github.io/PyEPO/build/html/content/examples/function.html | Use PyEPO only in an isolated prototype to avoid disturbing the main pipeline. |
| SPO/DFL | CVXPYlayers: https://github.com/cvxpy/cvxpylayers | Differentiable convex layers are feasible, but dependency/scaling risk keeps this out of the official champion. |
| CVaR/OCE | Rockafellar and Uryasev CVaR optimization: https://sites.math.washington.edu/~rtr/papers/rtr179-CVaR1.pdf | CVaR can be optimized with scenario/LP methods; this supports challenger analysis, not champion replacement by itself. |
| CVaR/OCE | Riskfolio-Lib: https://github.com/dcajasn/Riskfolio-Lib | Mature CVaR tooling exists, so future work should reuse tooling or a compact CVXPY LP rather than more generated waves. |

## Bounded Implementation Proposal

1. `ifrs9_sicr`: run one raw-enriched lifetime-ECL/SICR diagnostic using
   `total_pymnt`, `recoveries`, `last_pymnt_d`, hardship flags, and the existing
   LGD fields. Claim remains IFRS9-inspired, not contractual IFRS9.
2. `online_conformal`: run one source-family holdout redesign using
   issue-month, grade, state, income/DTI bins, and zip3. Evaluate MDCP-style
   max-p/union or defended-source pooling. Claim remains retrospective source
   governance.
3. `cate_policy_value`: run only a high-rate-within-grade observational
   sensitivity screen. The output is a causal-identification memo and placebo
   diagnostics, not a policy-value claim.
4. `cvar_oce`: run at most one raw-cashflow repricing check to see whether
   recovery-aware losses materially change the tail challenger. The champion
   cannot change unless paired wealth beats the current economic champion.
5. `fair_lending_proxy`: keep legal fair-lending false. Optionally add a
   geography/source governance appendix. BISG is not feasible without surnames
   and finer geocoding.
6. `spo_dfl`: keep isolated. A toy PyEPO or cvxpylayers prototype may be useful,
   but it should not enter the main CRPTO pipeline.
7. `dla_adp`: keep exact Bellman optimality false. Improve only the rollout
   simulator language and feature list.

## Files Written

- `{VARIABLE_INVENTORY.relative_to(ROOT)}`
- `{LANE_DECISIONS.relative_to(ROOT)}`
- `{RAW_PROFILE.relative_to(ROOT)}`
"""
    MEMO.write_text(text, encoding="utf-8")
    NOTE_MEMO.write_text(text, encoding="utf-8")


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    dictionary = load_dictionary()
    raw_header = read_raw_header(RAW_CSV)
    parquet_file = pq.ParquetFile(INTERIM_PARQUET)
    parquet_cols = [(field.name, str(field.type)) for field in parquet_file.schema_arrow]
    needed_cleaned = sorted(
        {
            c
            for lane in LANES.values()
            for c in lane["cleaned_need"]
            if c in {name for name, _ in parquet_cols}
        }
    )
    cleaned_profiles = profile_cleaned_columns(needed_cleaned)
    summary = dataset_summary()
    inventory_rows = write_variable_inventory(
        parquet_cols, raw_header, dictionary, cleaned_profiles
    )
    raw_profiles = profile_raw_columns(raw_header)
    pd.DataFrame(raw_profiles).to_csv(RAW_PROFILE, index=False)
    lane_rows = write_lane_decisions(
        {name for name, _ in parquet_cols}, set(raw_header), cleaned_profiles, raw_profiles
    )
    write_memo(summary, lane_rows, raw_profiles)

    print(
        json.dumps(
            {
                "variable_inventory_rows": len(inventory_rows),
                "lane_decision_rows": len(lane_rows),
                "raw_profile_rows": len(raw_profiles),
                "memo": str(MEMO.relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
