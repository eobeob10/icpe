import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from config import PROJECT_ROOT, SUMMARY_ID, BUG_ID_COL


def find_first(pattern: str, root: Path):
    hits = list(root.rglob(pattern))
    return hits[0] if hits else None


def load_raw_data():
    alerts_csv = find_first("alerts_data.csv", PROJECT_ROOT)
    bugs_csv = find_first("bugs_data.csv", PROJECT_ROOT)

    if not alerts_csv:
        raise FileNotFoundError("alerts_data.csv not found")

    print(f"Loading alerts from {alerts_csv}")
    alerts = pd.read_csv(alerts_csv, low_memory=False)

    bugs = None
    if bugs_csv:
        try:
            bugs = pd.read_csv(bugs_csv, low_memory=False)
        except Exception as e:
            print(f"Warning: Could not load bugs_data.csv: {e}")

    return alerts, bugs


def first_non_null(s: pd.Series):
    s2 = s.dropna()
    return s2.iloc[0] if len(s2) else np.nan


def mode_or_nan(s: pd.Series):
    s2 = s.dropna()
    if len(s2) == 0:
        return np.nan
    return s2.value_counts().idxmax()


def p90_non_null(x: pd.Series):
    x2 = pd.to_numeric(x, errors="coerce").dropna()
    if len(x2) == 0:
        return np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanpercentile(x2.values.astype(float), 90))


def aggregate_alerts(alerts: pd.DataFrame) -> pd.DataFrame:
    print("Aggregating alerts...")

    dirty_cols = [
        "single_alert_backfill_record_total_backfills_failed",
        "single_alert_backfill_record_total_backfills_successful"
    ]

    for c in dirty_cols:
        if c in alerts.columns:
            alerts[c] = pd.to_numeric(
                alerts[c].astype(str).str.replace(r'[(),]', '', regex=True),
                errors='coerce'
            ).fillna(0.0)

    single_num_candidates = [
        "single_alert_amount_abs",
        "single_alert_amount_pct",
        "single_alert_prev_value",
        "single_alert_new_value",
        # "single_alert_t_value",
    ]

    single_cat_candidates = [
        "single_alert_is_regression",
        "single_alert_manually_created",
        "single_alert_noise_profile",
        "single_alert_backfill_record_status",
        "single_alert_backfill_record_context",
        "single_alert_series_signature_suite",
        "single_alert_series_signature_test",
        "single_alert_series_signature_machine_platform",
        "single_alert_series_signature_measurement_unit",
        "single_alert_series_signature_lower_is_better",
        "single_alert_series_signature_extra_options",
        "single_alert_classifier",
        "single_alert_series_signature_has_subtests",
    ]

    summary_cols_candidates = [
        "push_timestamp",
        "alert_summary_creation_timestamp",
        "alert_summary_repository",
        "alert_summary_revision",
        "alert_summary_push_id",
        "alert_summary_prev_push_id",
        "alert_summary_prev_push_revision",
        "alert_summary_framework",
        #"alert_summary_issue_tracker",
        "alert_summary_related_alerts",
        "alert_summary_triage_due_date",
        "alert_summary_notes",
        "alert_summary_performance_tags",
    ]

    available_summary_cols = [c for c in summary_cols_candidates if c in alerts.columns]
    available_single_num = [c for c in single_num_candidates if c in alerts.columns]
    available_single_cat = [c for c in single_cat_candidates if c in alerts.columns]

    agg_spec = {}

    for c in available_summary_cols:
        agg_spec[c] = pd.NamedAgg(column=c, aggfunc=first_non_null)

    if "signature_id" in alerts.columns:
        agg_spec["signature_id__nunique"] = pd.NamedAgg(column="signature_id", aggfunc=lambda x: x.dropna().nunique())

    for c in available_single_num:
        agg_spec[f"{c}__mean"] = pd.NamedAgg(column=c, aggfunc="mean")
        agg_spec[f"{c}__max"] = pd.NamedAgg(column=c, aggfunc="max")
        agg_spec[f"{c}__min"] = pd.NamedAgg(column=c, aggfunc="min")
        agg_spec[f"{c}__std"] = pd.NamedAgg(column=c, aggfunc="std")
        agg_spec[f"{c}__p90"] = pd.NamedAgg(column=c, aggfunc=p90_non_null)

    for c in available_single_cat:
        agg_spec[f"{c}__mode"] = pd.NamedAgg(column=c, aggfunc=mode_or_nan)
        agg_spec[f"{c}__nunique"] = pd.NamedAgg(column=c, aggfunc=lambda x: x.dropna().nunique()) # Manquait

    agg_spec["n_single_alerts"] = pd.NamedAgg(column=SUMMARY_ID, aggfunc="size")

    g = alerts.groupby(SUMMARY_ID, dropna=False)
    summary_df = g.agg(**agg_spec).reset_index()

    if BUG_ID_COL in alerts.columns:
        bug_created = g[BUG_ID_COL].apply(lambda s: s.notna().any()).reset_index(name="bug_created")
        summary_df = summary_df.merge(bug_created, on=SUMMARY_ID, how="left")

        bug_id = g[BUG_ID_COL].apply(first_non_null).reset_index(name="bug_id")
        summary_df = summary_df.merge(bug_id, on=SUMMARY_ID, how="left")
    else:
        raise KeyError(f"{BUG_ID_COL} not found. Cannot build label bug_created.")

    if "push_timestamp" in summary_df.columns:
        ts = pd.to_datetime(summary_df["push_timestamp"], errors="coerce", utc=True)
        summary_df["push_dow"] = ts.dt.dayofweek
        summary_df["push_hour"] = ts.dt.hour
        summary_df["push_is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype("int8")

    if "alert_summary_notes" in summary_df.columns:
        notes = summary_df["alert_summary_notes"].fillna("").astype(str)
        summary_df["notes_len_chars"] = notes.str.len()
        summary_df["notes_len_words"] = notes.str.split().str.len()

    print(f"Aggregation ok. Shape: {summary_df.shape}")
    return summary_df