
import faulthandler
faulthandler.enable()
import os, sys
import numpy as np
import pandas as pd

print("Python exe:", sys.executable)
print("CWD:", os.getcwd())
print("numpy:", np.__version__)
print("pandas:", pd.__version__)

import torch
import transformers

print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("transformers:", transformers.__version__)

from transformers import Trainer, TrainingArguments

print("Trainer import OK")


from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path.cwd() / "../../icpe_data"


def find_first(pattern: str, root: Path):
    hits = list(root.rglob(pattern))
    return hits[0] if hits else None


# Core CSVs
ALERTS_CSV = find_first("alerts_data.csv", PROJECT_ROOT)
BUGS_CSV = find_first("bugs_data.csv", PROJECT_ROOT)

print("PROJECT_ROOT:", PROJECT_ROOT)
print("ALERTS_CSV:", ALERTS_CSV)
print("BUGS_CSV  :", BUGS_CSV)

# Timeseries root folder (must be the folder that contains autoland1, autoland2, ...)
ts_root_candidates = [p for p in PROJECT_ROOT.rglob("timeseries-data") if p.is_dir()]
assert len(ts_root_candidates) > 0, "Could not find a folder named 'timeseries-data' under PROJECT_ROOT."
TS_ROOT = ts_root_candidates[0]

# Collect all TS files under TS_ROOT (all subfolders)
ts_files = list(TS_ROOT.rglob("*_timeseries_data.csv"))

print("TS_ROOT:", TS_ROOT)
print("Timeseries files found:", len(ts_files))

# Sanity: how many files per subfolder
folder_counts = pd.Series([p.parent.name for p in ts_files]).value_counts()
print("\nTimeseries files per subfolder:")
print(folder_counts)


import pandas as pd

assert ALERTS_CSV is not None, "alerts_data.csv not found. Set ALERTS_CSV manually."
alerts = pd.read_csv(ALERTS_CSV, low_memory=False)

bugs = None
if BUGS_CSV is not None:
    try:
        bugs = pd.read_csv(BUGS_CSV, low_memory=False)
    except Exception as e:
        print("Could not load bugs_data.csv, continuing without it:", repr(e))

print("alerts shape:", alerts.shape)
print("bugs loaded:", bugs is not None)

SUMMARY_ID = "alert_summary_id"
BUG_ID_COL = "alert_summary_bug_number"

assert SUMMARY_ID in alerts.columns, f"Missing {SUMMARY_ID} in alerts"
print("Has bug id col:", BUG_ID_COL in alerts.columns)

# quick peek
print(alerts.head(3))


import numpy as np
import pandas as pd


def first_non_null(s: pd.Series):
    s2 = s.dropna()
    return s2.iloc[0] if len(s2) else np.nan


def mode_or_nan(s: pd.Series):
    s2 = s.dropna()
    if len(s2) == 0:
        return np.nan
    return s2.value_counts().idxmax()


def nunique_non_null(s: pd.Series) -> int:
    return int(s.dropna().nunique())


def p90_non_null(x: pd.Series):
    x2 = pd.to_numeric(x, errors="coerce").dropna()
    return float(np.nanpercentile(x2, 90)) if len(x2) else np.nan


summary_cols_candidates = [
    "push_timestamp",
    "alert_summary_creation_timestamp",
    "alert_summary_repository",
    "alert_summary_revision",
    "alert_summary_push_id",
    "alert_summary_prev_push_id",
    "alert_summary_prev_push_revision",
    "alert_summary_framework",
    "alert_summary_issue_tracker",
    "alert_summary_related_alerts",
    "alert_summary_triage_due_date",
    "alert_summary_notes",
]

single_num_candidates = [
    "single_alert_amount_abs",
    "single_alert_amount_pct",
    "single_alert_prev_value",
    "single_alert_new_value",
    "single_alert_t_value",
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
]

available_summary_cols = [c for c in summary_cols_candidates if c in alerts.columns]
available_single_num = [c for c in single_num_candidates if c in alerts.columns]
available_single_cat = [c for c in single_cat_candidates if c in alerts.columns]

print("Using summary cols:", available_summary_cols)
print("Using numeric cols:", available_single_num)
print("Using categorical cols:", available_single_cat)

agg_spec = {}
for c in available_summary_cols:
    agg_spec[c] = pd.NamedAgg(column=c, aggfunc=first_non_null)

if "signature_id" in alerts.columns:
    agg_spec["signature_id__nunique"] = pd.NamedAgg(column="signature_id", aggfunc=nunique_non_null)

for c in available_single_num:
    agg_spec[f"{c}__mean"] = pd.NamedAgg(column=c, aggfunc="mean")
    agg_spec[f"{c}__max"] = pd.NamedAgg(column=c, aggfunc="max")
    agg_spec[f"{c}__min"] = pd.NamedAgg(column=c, aggfunc="min")
    agg_spec[f"{c}__std"] = pd.NamedAgg(column=c, aggfunc="std")
    agg_spec[f"{c}__p90"] = pd.NamedAgg(column=c, aggfunc=p90_non_null)

for c in available_single_cat:
    agg_spec[f"{c}__mode"] = pd.NamedAgg(column=c, aggfunc=mode_or_nan)
    agg_spec[f"{c}__nunique"] = pd.NamedAgg(column=c, aggfunc=nunique_non_null)

agg_spec["n_single_alerts"] = pd.NamedAgg(column=SUMMARY_ID, aggfunc="size")

g = alerts.groupby(SUMMARY_ID, dropna=False)
summary_df = g.agg(**agg_spec).reset_index()

# Label: bug_created and optional bug_id
if BUG_ID_COL in alerts.columns:
    bug_created = g[BUG_ID_COL].apply(lambda s: s.notna().any()).reset_index(name="bug_created")
    summary_df = summary_df.merge(bug_created, on=SUMMARY_ID, how="left")

    bug_id = g[BUG_ID_COL].apply(first_non_null).reset_index(name="bug_id")
    summary_df = summary_df.merge(bug_id, on=SUMMARY_ID, how="left")
else:
    raise KeyError(f"{BUG_ID_COL} not found. Cannot build label bug_created.")

# Derived time features
if "push_timestamp" in summary_df.columns:
    ts = pd.to_datetime(summary_df["push_timestamp"], errors="coerce", utc=True)
    summary_df["push_dow"] = ts.dt.dayofweek
    summary_df["push_hour"] = ts.dt.hour
    summary_df["push_is_weekend"] = ts.dt.dayofweek.isin([5, 6]).astype("int8")

# Notes length features
if "alert_summary_notes" in summary_df.columns:
    notes = summary_df["alert_summary_notes"].fillna("").astype(str)
    summary_df["notes_len_chars"] = notes.str.len()
    summary_df["notes_len_words"] = notes.str.split().str.len()

print("summary_df:", summary_df.shape, "unique summaries:", summary_df[SUMMARY_ID].nunique())
print("bug_created rate:", float(summary_df["bug_created"].mean()))
print(summary_df.head(3))


from collections import defaultdict
from pathlib import Path
import pandas as pd

# ts_files and TS_ROOT come from Cell 1
assert "TS_ROOT" in globals(), "TS_ROOT not found. Run Cell 1."
assert "ts_files" in globals(), "ts_files not found. Run Cell 1."
assert len(ts_files) > 0, "No *_timeseries_data.csv files found under TS_ROOT."

# Build: folder -> {sig_id: path}
ts_index = defaultdict(dict)

# Build reverse map: sig_id -> [folders...]
sig_to_folders = defaultdict(list)

for p in ts_files:
    name = p.name
    if not name.endswith("_timeseries_data.csv"):
        continue
    prefix = name.replace("_timeseries_data.csv", "")
    try:
        sig_id = int(prefix)
    except ValueError:
        continue

    folder = p.parent.name  # autoland1, mozilla-central, etc.
    ts_index[folder][sig_id] = p
    sig_to_folders[sig_id].append(folder)

folders = sorted(ts_index.keys())

print("TS_ROOT:", TS_ROOT)
print("Folders indexed:", folders)
print("Files per folder:")
print(pd.Series({k: len(v) for k, v in ts_index.items()}).sort_values(ascending=False))

n_unique_sigs = len(sig_to_folders)
n_multi = sum(1 for sig, fl in sig_to_folders.items() if len(set(fl)) > 1)
print("Unique signature_ids indexed:", n_unique_sigs)
print("signature_ids appearing in >1 folder:", n_multi)


def folder_candidates_from_row(row: pd.Series) -> list[str]:
    repo = str(row.get("alert_summary_repository", "")).lower()
    fw = str(row.get("alert_summary_framework", "")).lower()
    hay = repo + " " + fw

    # Priority buckets
    candidates = []

    # Direct folder name match
    for f in folders:
        if f.lower() in hay:
            candidates.append(f)

    # Heuristics for common cases
    if "autoland" in hay:
        candidates.extend([f for f in folders if f.startswith("autoland")])
    if "android" in hay:
        candidates.extend([f for f in folders if "android" in f])

    # Deduplicate while preserving order
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def resolve_ts_path(sig_id: int, row: pd.Series | None = None) -> Path | None:
    sig_id = int(sig_id)

    # 1) Try folders suggested by this row
    if row is not None:
        for f in folder_candidates_from_row(row):
            p = ts_index.get(f, {}).get(sig_id)
            if p is not None:
                return p

    # 2) Fallback: any folder that contains the sig_id
    fl = sig_to_folders.get(sig_id, [])
    for f in fl:
        p = ts_index.get(f, {}).get(sig_id)
        if p is not None:
            return p

    return None



from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

# -----------------------------
# TS window parameters + cache
# -----------------------------
N_PRE = 20
N_POST = 10

CACHE_DIR = Path("./derived_features")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / f"ts_features_ALLFOLDERS_pre{N_PRE}_post{N_POST}.parquet"

# -----------------------------
# Anchor columns (what we align TS on)
# -----------------------------
ANCHOR_PUSH_COL = "alert_summary_push_id" if "alert_summary_push_id" in summary_df.columns else None
ANCHOR_REV_COL = "alert_summary_revision" if "alert_summary_revision" in summary_df.columns else None

if ANCHOR_PUSH_COL is None and ANCHOR_REV_COL is None:
    raise KeyError("Need alert_summary_push_id or alert_summary_revision in summary_df to anchor TS windows.")

print("ANCHOR_PUSH_COL:", ANCHOR_PUSH_COL)
print("ANCHOR_REV_COL :", ANCHOR_REV_COL)

# -----------------------------
# Build summary -> signature list + join needed row context for path resolution
# -----------------------------
summary_sigs = (
    alerts.groupby(SUMMARY_ID)["signature_id"]
    .apply(lambda s: sorted(set(pd.to_numeric(s, errors="coerce").dropna().astype(int).tolist())))
    .reset_index(name="signature_ids")
)

needed_cols = [SUMMARY_ID, "alert_summary_repository", "alert_summary_framework"]
if ANCHOR_PUSH_COL: needed_cols.append(ANCHOR_PUSH_COL)
if ANCHOR_REV_COL: needed_cols.append(ANCHOR_REV_COL)

needed_cols = [c for c in needed_cols if c in summary_df.columns]
summary_sigs = summary_sigs.merge(summary_df[needed_cols], on=SUMMARY_ID, how="left")

print("summary_sigs shape:", summary_sigs.shape)
print(summary_sigs.head(3))


# -----------------------------
# Cached loader by file path (fast)
# -----------------------------
@lru_cache(maxsize=4096)
def load_timeseries_by_path(path_str: str) -> pd.DataFrame:
    print("Loading timeseries by_path:", path_str)
    p = Path(path_str)
    df = pd.read_csv(p, engine="pyarrow")

    keep = [c for c in ["push_id", "revision", "value"] if c in df.columns]
    df = df[keep].copy()

    if "push_id" in df.columns:
        df["push_id"] = pd.to_numeric(df["push_id"], errors="coerce")
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["value"])
    if "push_id" in df.columns:
        df = df.dropna(subset=["push_id"]).sort_values("push_id")

    return df


def safe_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return np.nan


def compute_window_features(ts: pd.DataFrame, anchor_push_id=None, anchor_revision=None, n_pre=20, n_post=10):
    out = {
        "ts_n_total": int(len(ts)) if ts is not None else 0,
        "ts_n_pre": np.nan,
        "ts_n_post": np.nan,
        "ts_delta_mean_rel": np.nan,
        "ts_delta_mean": np.nan,
        "ts_z_pre_post": np.nan,
        "ts_delta_std": np.nan,
        "ts_slope_change": np.nan,
    }
    if ts is None or ts.empty or "value" not in ts.columns:
        return out

    anchor_idx = None

    # Prefer push_id alignment
    if anchor_push_id is not None and "push_id" in ts.columns and pd.notna(anchor_push_id):
        ap = pd.to_numeric(anchor_push_id, errors="coerce")
        if pd.notna(ap):
            push_ids = ts["push_id"].to_numpy()
            exact = np.where(push_ids == ap)[0]
            if len(exact):
                anchor_idx = int(exact[0])
            else:
                le = np.where(push_ids <= ap)[0]
                if len(le):
                    anchor_idx = int(le[-1])

    # Fallback to revision alignment
    if anchor_idx is None and anchor_revision is not None and "revision" in ts.columns and pd.notna(anchor_revision):
        rev = str(anchor_revision)
        matches = np.where(ts["revision"].astype(str).to_numpy() == rev)[0]
        if len(matches):
            anchor_idx = int(matches[0])

    if anchor_idx is None:
        return out

    values = ts["value"].to_numpy()
    pre_start = max(0, anchor_idx - n_pre)
    pre = values[pre_start:anchor_idx]
    post_end = min(len(values), anchor_idx + 1 + n_post)
    post = values[anchor_idx + 1:post_end]

    out["ts_n_pre"] = int(len(pre))
    out["ts_n_post"] = int(len(post))
    if len(pre) == 0 or len(post) == 0:
        return out

    pre_mean = float(np.nanmean(pre))
    post_mean = float(np.nanmean(post))

    pre_std = float(np.nanstd(pre, ddof=1)) if len(pre) > 1 else np.nan
    post_std = float(np.nanstd(post, ddof=1)) if len(post) > 1 else np.nan

    denom = max(abs(pre_mean), 1e-9)
    out["ts_delta_mean_rel"] = (post_mean - pre_mean) / denom
    out["ts_delta_mean"] = post_mean - pre_mean
    out["ts_delta_std"] = (post_std - pre_std) if (pd.notna(pre_std) and pd.notna(post_std)) else np.nan

    x_pre = np.arange(len(pre), dtype=float)
    x_post = np.arange(len(post), dtype=float)
    pre_slope = safe_slope(x_pre, pre.astype(float))
    post_slope = safe_slope(x_post, post.astype(float))
    if pd.notna(pre_slope) and pd.notna(post_slope):
        out["ts_slope_change"] = post_slope - pre_slope

    if pd.notna(pre_std) and pre_std > 0:
        out["ts_z_pre_post"] = (post_mean - pre_mean) / pre_std

    return out


# -----------------------------
# Compute TS features per summary (cached)
# -----------------------------
if CACHE_FILE.exists():
    ts_feat_df = pd.read_parquet(CACHE_FILE)
    print("Loaded cached TS features:", ts_feat_df.shape)
else:
    rows = []

    for row in summary_sigs.itertuples(index=False):
        sid = int(getattr(row, SUMMARY_ID))
        sigs = getattr(row, "signature_ids")

        anchor_push = getattr(row, ANCHOR_PUSH_COL) if ANCHOR_PUSH_COL else None
        anchor_rev = getattr(row, ANCHOR_REV_COL) if ANCHOR_REV_COL else None

        # Reconstruct row as Series for resolve_ts_path
        row_series = pd.Series(row._asdict())

        per_sig_feats = []
        missing = 0

        for sig in sigs:
            p = resolve_ts_path(int(sig), row=row_series)
            if p is None:
                missing += 1
                continue

            ts = load_timeseries_by_path(str(p))
            if ts.empty:
                missing += 1
                continue

            f = compute_window_features(
                ts,
                anchor_push_id=anchor_push,
                anchor_revision=anchor_rev,
                n_pre=N_PRE,
                n_post=N_POST
            )
            per_sig_feats.append(f)

        agg_row = {SUMMARY_ID: sid}
        agg_row["ts_sig_used"] = int(len(per_sig_feats))
        agg_row["ts_sig_missing"] = int(missing)

        if len(per_sig_feats) == 0:
            rows.append(agg_row)
            continue

        feats_df = pd.DataFrame(per_sig_feats)

        for base in ["ts_delta_mean_rel", "ts_delta_mean", "ts_z_pre_post", "ts_delta_std", "ts_slope_change",
                     "ts_n_pre", "ts_n_post"]:
            v = pd.to_numeric(feats_df.get(base, np.nan), errors="coerce")
            agg_row[f"{base}__mean_over_sigs"] = float(v.mean())
            agg_row[f"{base}__max_over_sigs"] = float(v.max())

        rows.append(agg_row)

    ts_feat_df = pd.DataFrame(rows)
    ts_feat_df.to_parquet(CACHE_FILE, index=False)
    print("Computed and cached TS features:", ts_feat_df.shape)

# Merge into summary_df_full_ts
summary_df_full_ts = summary_df.merge(ts_feat_df, on=SUMMARY_ID, how="left")
print("summary_df_full_ts shape:", summary_df_full_ts.shape)

# Quick coverage sanity
print("\nTS coverage summary:")
print(summary_df_full_ts[["ts_sig_used", "ts_sig_missing"]].fillna(0).describe())
print("Share with ts_sig_used > 0:", float((summary_df_full_ts["ts_sig_used"].fillna(0) > 0).mean()))


import re
import numpy as np
import pandas as pd

df = summary_df_full_ts.copy()

# -----------------------------
# 1) Time-based split (train/val/test)
#    - Train: oldest 70%
#    - Val  : next   10%
#    - Test : newest 20%   (held out; never used for model selection)
# -----------------------------
TEST_FRAC = 0.20
VAL_FRAC = 0.10

time_col = None
for c in ["push_timestamp", "alert_summary_creation_timestamp"]:
    if c in df.columns and pd.to_datetime(df[c], errors="coerce", utc=True).notna().any():
        time_col = c
        break
if time_col is None:
    raise ValueError("No usable time column found for time-based split.")

t = pd.to_datetime(df[time_col], errors="coerce", utc=True)
df = df.loc[t.notna()].copy()
t = t.loc[t.notna()].copy()

df["bug_created"] = df["bug_created"].astype(int)

order = np.argsort(t.values)
df_sorted = df.iloc[order].reset_index(drop=True)
t_sorted = t.iloc[order].reset_index(drop=True)

n = len(df_sorted)
n_test = max(1, int(TEST_FRAC * n))
n_val = max(1, int(VAL_FRAC * n))
n_train = n - n_val - n_test

# Guardrails for very small datasets
if n_train < 1:
    n_train = 1
    # rebalance val/test if needed
    remaining = n - n_train
    n_val = max(1, int(0.33 * remaining)) if remaining > 1 else 1
    n_test = max(1, remaining - n_val)

cut_train = n_train
cut_val = n_train + n_val

train_idx = np.arange(0, cut_train)
val_idx = np.arange(cut_train, cut_val)
test_idx = np.arange(cut_val, n)

print("Time col:", time_col)
print("Train:", t_sorted.iloc[0], "to", t_sorted.iloc[cut_train - 1], "n=", len(train_idx))
print("Val  :", t_sorted.iloc[cut_train], "to", t_sorted.iloc[cut_val - 1], "n=", len(val_idx))
print("Test :", t_sorted.iloc[cut_val], "to", t_sorted.iloc[-1], "n=", len(test_idx))

print("Train pos rate:", float(df_sorted.loc[train_idx, "bug_created"].mean()))
print("Val   pos rate:", float(df_sorted.loc[val_idx, "bug_created"].mean()))
print("Test  pos rate:", float(df_sorted.loc[test_idx, "bug_created"].mean()))

# -----------------------------
# 2) Notes sanitization
# -----------------------------
_bug_url_pat = re.compile(r"https?://\S*bugzilla\S*", flags=re.IGNORECASE)
_bug_num_pat = re.compile(r"\bbug\s*#?\s*\d+\b", flags=re.IGNORECASE)
_num_pat = re.compile(r"\b\d{4,}\b")


def sanitize_notes(s: str) -> str:
    s = "" if s is None else str(s)
    s = _bug_url_pat.sub(" BUGZILLA_URL ", s)
    s = _bug_num_pat.sub(" BUG_ID ", s)
    s = _num_pat.sub(" LONG_NUM ", s)
    return s.strip()


def fill_empty(s: str) -> str:
    s = "" if s is None else str(s).strip()
    return s if s else "NO_NOTES"


if "alert_summary_notes" not in df_sorted.columns:
    df_sorted["alert_summary_notes"] = ""

df_sorted["notes_raw"] = df_sorted["alert_summary_notes"].map(fill_empty)
df_sorted["notes_sanitized"] = df_sorted["notes_raw"].map(sanitize_notes).map(fill_empty)

# -----------------------------
# 3) Fused text (notes + safe metadata + TS aggregates)
# -----------------------------
SAFE_TEXT_FIELDS = [
    "alert_summary_repository",
    "alert_summary_framework",
    "single_alert_series_signature_suite__mode",
    "single_alert_series_signature_test__mode",
    "single_alert_series_signature_machine_platform__mode",
    "single_alert_noise_profile__mode",
    "n_single_alerts",
    "single_alert_amount_pct__mean",
    "single_alert_amount_pct__max",
    "single_alert_t_value__max",
    "ts_sig_used",
    "ts_sig_missing",
    "ts_delta_mean_rel__mean_over_sigs",
    "ts_delta_mean_rel__max_over_sigs",
    "ts_z_pre_post__max_over_sigs",
    "ts_slope_change__max_over_sigs",
]

SAFE_TEXT_FIELDS = [c for c in SAFE_TEXT_FIELDS if c in df_sorted.columns]


def to_token(k: str, v) -> str:
    if pd.isna(v):
        return ""
    if isinstance(v, (float, np.floating)):
        return f"{k}={float(v):.4g}"
    return f"{k}={str(v)}"


def build_fused_text(row: pd.Series, base_col: str) -> str:
    parts = [row[base_col]]
    for c in SAFE_TEXT_FIELDS:
        tok = to_token(c, row[c])
        if tok:
            parts.append(tok)
    return " | ".join(parts)


df_sorted["fused_text_raw"] = df_sorted.apply(lambda r: build_fused_text(r, "notes_raw"), axis=1)
df_sorted["fused_text_sanitized"] = df_sorted.apply(lambda r: build_fused_text(r, "notes_sanitized"), axis=1)

print(df_sorted[["bug_created", "notes_sanitized", "fused_text_sanitized"]].head(3))


import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from datasets import Dataset


# -----------------------------
# 1) Metrics helpers
# -----------------------------
def precision_recall_at_k(y_true, y_score, k: int):
    k = int(min(k, len(y_true)))
    idx = np.argsort(-y_score)[:k]
    y_top = np.array(y_true)[idx]
    precision = float(y_top.mean())
    recall = float(y_top.sum() / max(1, np.sum(y_true)))
    return precision, recall


def report_metrics(y_true: np.ndarray, y_score: np.ndarray, ks=(50, 100, 200)):
    out = {}
    out["AUPRC"] = float(average_precision_score(y_true, y_score))
    try:
        out["AUROC"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        out["AUROC"] = float("nan")
    for k in ks:
        p, r = precision_recall_at_k(y_true, y_score, k)
        out[f"P@{min(k, len(y_true))}"] = p
        out[f"R@{min(k, len(y_true))}"] = r
    return out


# -----------------------------
# 2) Build HF datasets from df_sorted
# -----------------------------
def make_hf_datasets(text_col: str):
    """Create Hugging Face datasets using the global time-based indices.

    Returns:
        (train_ds, val_ds, test_ds)
    """
    assert text_col in df_sorted.columns, f"{text_col} not found in df_sorted"

    train_df = df_sorted.loc[train_idx, [text_col, "bug_created"]].copy()
    val_df = df_sorted.loc[val_idx, [text_col, "bug_created"]].copy()
    test_df = df_sorted.loc[test_idx, [text_col, "bug_created"]].copy()

    train_df = train_df.rename(columns={text_col: "text", "bug_created": "label"})
    val_df = val_df.rename(columns={text_col: "text", "bug_created": "label"})
    test_df = test_df.rename(columns={text_col: "text", "bug_created": "label"})

    # HF expects int labels
    train_df["label"] = train_df["label"].astype(int)
    val_df["label"] = val_df["label"].astype(int)
    test_df["label"] = test_df["label"].astype(int)

    train_ds = Dataset.from_pandas(train_df, preserve_index=False)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False)
    test_ds = Dataset.from_pandas(test_df, preserve_index=False)
    return train_ds, val_ds, test_ds


# Choose which input text you want for BERT first:
TEXT_COL = "notes_sanitized"  # alternative: "fused_text_sanitized"

train_ds, val_ds, test_ds = make_hf_datasets(TEXT_COL)
print("Using TEXT_COL:", TEXT_COL)
print("Train:", train_ds)
print("Val  :", val_ds)
print("Test :", test_ds)

# Quick label balance check
print("Train pos rate:", float(np.mean(train_ds["label"])))
print("Val   pos rate:", float(np.mean(val_ds["label"])))
print("Test  pos rate:", float(np.mean(test_ds["label"])))



# --- Helper Functions for Serialization ---

def bin_val(col: str, value):
    """
    Calculates the bin index for a numeric value based on pre-calculated bin_edges.
    Returns -1 if missing or no edges found.
    """
    if pd.isna(value):
        return -1

    # Ensure bin_edges exists (computed in previous cells of your notebook)
    if 'bin_edges' not in globals() or col not in bin_edges:
        return -1

    edges = bin_edges[col]
    # Find insertion point
    idx = np.searchsorted(edges, float(value))
    # Clip to valid bin range (0 to N_BINS-1)
    # We subtract 1 because searchsorted returns the index where it *would* go
    bin_idx = max(0, min(len(edges) - 2, int(idx) - 1))
    return bin_idx


def num_to_token(col: str, q: int) -> str:
    """Creates a token string like 'feature=q5'."""
    if q < 0:
        return ""  # Skip missing values
    return f"{col}=q{q}"


def cat_to_token(col: str, val) -> str:
    """Creates a token string for categorical values."""
    if pd.isna(val):
        return ""
    # Simple sanitization: lowercase and replace spaces
    clean_val = str(val).lower().strip().replace(" ", "_")
    # Remove non-alphanumeric characters if necessary, or keep simple
    clean_val = re.sub(r"[^a-z0-9_\-\.]", "", clean_val)
    return f"{col}={clean_val}"



import numpy as np
import pandas as pd
import re

assert "df_sorted" in globals(), "Run Cell 6 first (df_sorted)."
assert "train_idx" in globals() and "val_idx" in globals() and "test_idx" in globals(), "Run Cell 6 first (train_idx/val_idx/test_idx)."
assert "SUMMARY_ID" in globals(), "Run earlier cells (SUMMARY_ID)."

# -----------------------------
# 1) Choose which columns become tokens
# -----------------------------
EXCLUDE_COLS = {
    "bug_created", "bug_id",
    SUMMARY_ID,
    "alert_summary_notes",  # we use notes_sanitized instead
    "notes_raw", "notes_sanitized",
    "fused_text_raw", "fused_text_sanitized",
    "fused_text_all", "all_text",
}

# Drop likely high-cardinality IDs/hashes/timestamps that explode vocab
DROP_IF_NAME_CONTAINS = [
    "revision",
    "push_id",  # raw push ids
    "prev_push",  # previous push ids/revs
    "timestamp",  # raw timestamps
    "triage_due_date",  # raw date strings
]

candidate_cols = []
for c in df_sorted.columns:
    if c in EXCLUDE_COLS:
        continue
    cl = c.lower()
    if any(k in cl for k in DROP_IF_NAME_CONTAINS):
        continue
    if cl.endswith("_id"):
        continue
    if cl.endswith("bug_number"):
        continue
    candidate_cols.append(c)

# Remove very high-cardinality object columns (heuristic)
MAX_CAT_UNIQUE = 500
final_cols = []
for c in candidate_cols:
    if df_sorted[c].dtype == "object":
        nunq = df_sorted.loc[train_idx, c].nunique(dropna=True)
        if nunq > MAX_CAT_UNIQUE:
            continue
    final_cols.append(c)

print("Tokenized feature columns:", len(final_cols))
print(final_cols[:30], "..." if len(final_cols) > 30 else "")

# -----------------------------
# 2) Fit quantile bins for numeric columns on TRAIN only (no leakage)
# -----------------------------
N_BINS = 10  # Q0..Q9

num_cols = df_sorted[final_cols].select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in final_cols if c not in num_cols]

bin_edges = {}
for c in num_cols:
    x = pd.to_numeric(df_sorted.loc[train_idx, c], errors="coerce").dropna().to_numpy()
    if x.size < 5:
        continue
    qs = np.linspace(0, 1, N_BINS + 1)
    edges = np.unique(np.quantile(x, qs))
    if edges.size >= 3:  # needs at least 2 intervals
        bin_edges[c] = edges

print("Numeric columns:", len(num_cols), "| with bins:", len(bin_edges))
print("Categorical columns:", len(cat_cols))

# -----------------------------
# 3) Token formatting helpers
# -----------------------------
_space_pat = re.compile(r"\s+")
_bad_pat = re.compile(r"[^A-Za-z0-9_\-\.]+")  # keep it tokenizer-friendly


def clean_value(v: str, max_len: int = 40) -> str:
    v = str(v)
    v = v.strip().lower()
    v = _space_pat.sub("_", v)
    v = _bad_pat.sub("", v)
    if not v:
        return "empty"
    return v[:max_len]


def num_to_bin_token(col: str, v) -> str:
    if pd.isna(v):
        return f"{col}=missing"
    vv = float(v)
    edges = bin_edges.get(col)
    if edges is None:
        # fallback: coarse sign token only
        if vv > 0:
            return f"{col}=pos"
        if vv < 0:
            return f"{col}=neg"
        return f"{col}=zero"
    # bucket index in 0..(len(edges)-2)
    b = int(np.searchsorted(edges, vv, side="right") - 1)
    b = max(0, min(b, len(edges) - 2))
    # map to Q0..Q9 style (approx)
    q = int(round(b * (N_BINS - 1) / max(1, (len(edges) - 2))))
    return f"{col}=q{q}"


def cat_to_token(col: str, v) -> str:
    if pd.isna(v):
        return f"{col}=missing"
    return f"{col}={clean_value(v)}"


# -----------------------------
# 4) Build all_text (notes + tokens)
# -----------------------------
def build_all_text(row: pd.Series) -> str:
    # 1. Create a natural language summary of key signals
    signals = []

    # Is it a regression? (Critical feature)
    if row.get('single_alert_is_regression') == 1:
        signals.append("This is a performance regression.")

    # Magnitude of change (Verbalize the math)
    pct = row.get('single_alert_amount_pct', 0)
    if pd.notna(pct) and abs(pct) > 10:
        signals.append(f"The performance changed significantly by {pct:.1f} percent.")

    # 2. Add the standard tokens for the rest
    tokens = []
    for c in num_cols:  # Keep your existing binning logic
        q = bin_val(c, row[c])
        tokens.append(num_to_token(c, q))

    # 3. Combine: [Natural Signals] + [Raw Tokens] + [User Notes]
    base_notes = row.get("notes_sanitized", "")
    return " ".join(signals) + " | " + " ".join(tokens) + " | " + base_notes


df_sorted["all_text"] = df_sorted.apply(build_all_text, axis=1)

print("Example all_text:")
print(df_sorted["all_text"].iloc[0][:400])


import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
    EarlyStoppingCallback,
)

# -----------------------------
# 1. Configuration & Checks
# -----------------------------
# Check dependencies from previous cells
assert "df_sorted" in globals(), "df_sorted is missing. Run previous cells."
assert "all_text" in df_sorted.columns, "df_sorted['all_text'] is missing."
assert "make_hf_datasets" in globals(), "make_hf_datasets function is missing."
assert "report_metrics" in globals(), "report_metrics function is missing."

CONFIG = {
    "text_col": "all_text",
    "model_name": "distilbert-base-uncased",
    "seed": 44,
    "max_len": 512,
    "epochs": 6,
    "batch_size_train": 8,
    "batch_size_eval": 16,
    "learning_rate": 2e-5,
    "weight_decay": 0.1,
    "warmup_ratio": 0.1,
}

set_seed(CONFIG["seed"])

# -----------------------------
# 2. Data Preparation
# -----------------------------
# Create Hugging Face datasets
train_ds, val_ds, test_ds = make_hf_datasets(CONFIG["text_col"])

print(f"Model: {CONFIG['model_name']} | Text Column: {CONFIG['text_col']}")
print(f"Train size: {len(train_ds)} | Val size: {len(val_ds)} | Test size: {len(test_ds)}")

# Tokenization
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"], use_fast=True)


def tokenize_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=CONFIG["max_len"]
    )


train_tok = train_ds.map(tokenize_batch, batched=True, remove_columns=["text"])
val_tok = val_ds.map(tokenize_batch, batched=True, remove_columns=["text"])
test_tok = test_ds.map(tokenize_batch, batched=True, remove_columns=["text"])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# -----------------------------
# 2b) Metrics on the validation/test splits (optional but useful for monitoring)
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Convert logits -> P(class=1)
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    labels = np.array(labels)
    return report_metrics(labels, probs, ks=(50, 100, 200))


# -----------------------------
# 3. Dynamic Class Weights
# -----------------------------
# Calculate weights based on actual training data imbalance
y_train = np.array(train_ds["label"])
classes = np.unique(y_train)
cw_values = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)

# Convert to tensor and move to correct device later
class_weights_tensor = torch.tensor(cw_values, dtype=torch.float)

print(f"Computed Class Weights: {dict(zip(classes, cw_values))}")


# -----------------------------
# 4. Custom Weighted Trainer
# -----------------------------
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    # CHANGE HERE: Add num_items_in_batch=None to the arguments
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:
            weights = self.class_weights.to(model.device)
            loss_fct = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# -----------------------------
# 5. Model Initialization & Training
# -----------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    CONFIG["model_name"],
    num_labels=2
)

training_args = TrainingArguments(
    output_dir=f"./bert_runs/{CONFIG['model_name'].replace('/', '-')}-v3",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=CONFIG["learning_rate"],
    per_device_train_batch_size=CONFIG["batch_size_train"],
    per_device_eval_batch_size=CONFIG["batch_size_eval"],
    num_train_epochs=CONFIG["epochs"],
    weight_decay=CONFIG["weight_decay"],
    warmup_ratio=CONFIG["warmup_ratio"],
    load_best_model_at_end=True,
    metric_for_best_model="eval_P@50",
    save_total_limit=1,
)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,  # Validation split (used for model selection)
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    class_weights=class_weights_tensor,  # Pass the calculated weights here
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()

# -----------------------------
# 6. Final Evaluation
# -----------------------------
print("\n--- Final Evaluation ---")
pred = trainer.predict(test_tok)
logits = pred.predictions

# Get probabilities for class 1 (Bug)
proba_pos = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
y_true = np.array(test_ds["label"])

# Report metrics
metrics = report_metrics(y_true, proba_pos, ks=(50, 100, 200))

print(f"Results for {CONFIG['model_name']} on {CONFIG['text_col']}")
for k, v in metrics.items():
    print(f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

print("Best checkpoint:", trainer.state.best_model_checkpoint)
print("Best metric:", trainer.state.best_metric)
# -----------------------------


# Token length analysis
import numpy as np


def token_lengths(texts, tokenizer, add_special_tokens=True, step=2048):
    lens = []
    for i in range(0, len(texts), step):
        batch = texts[i:i + step]
        enc = tokenizer(
            batch,
            add_special_tokens=add_special_tokens,
            truncation=False,  # important: do NOT truncate
            padding=False,
            return_attention_mask=False,
        )
        lens.extend([len(ids) for ids in enc["input_ids"]])
    return np.array(lens, dtype=np.int32)


# Example usage: texts from your HF dataset split (train_ds) or pandas
train_texts = list(df_sorted["all_text"])
lens = token_lengths(train_texts, tokenizer)

print("N:", len(lens))
for p in [50, 75, 80, 90, 95, 97, 99]:
    print(f"p{p}: {np.percentile(lens, p):.0f}")
print("max:", lens.max())

candidates = [64, 128, 256, 384, 512, 1024]
for L in candidates:
    trunc_rate = (lens > (L)).mean()
    print(f"L={L:3d}  trunc_rate={trunc_rate:.3%}")


import os
import json
import csv
import time
import platform
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

import torch


# ----------------------------
# 1) What we will log (schema)
# ----------------------------
@dataclass
class RunRecord:
    run_id: str
    timestamp_utc: str

    # Data + split
    model_name: str
    text_col: str
    seed: int
    n_train: int
    n_val: int
    n_test: int

    # Tokenization
    max_length: int
    truncation_policy: str  # e.g., "head", "tail", "head+tail", "chunked"

    # Training hyperparams
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: float
    early_stopping_patience: int
    class_weights: str  # store as json string

    # Selection
    metric_for_best_model: str
    best_model_checkpoint: str
    best_metric_value: float
    best_epoch: Optional[float]

    # Validation metrics (best checkpoint)
    val_metrics: str  # json string

    # Test metrics (best checkpoint)
    test_metrics: str  # json string

    # Environment
    device: str
    gpu_name: str
    transformers_version: str
    torch_version: str
    python_version: str
    platform: str

    # Runtime
    wall_time_seconds: float


# ----------------------------
# 2) Logger utility
# ----------------------------
class SweepLogger:
    def __init__(self, out_dir: str, csv_name: str = "sweep_results.csv"):
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.csv_path = os.path.join(self.out_dir, csv_name)

        self._t0 = None

        # Create CSV with header if it does not exist
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(asdict(self._empty_record()).keys()))
                writer.writeheader()

    def _empty_record(self) -> RunRecord:
        return RunRecord(
            run_id="",
            timestamp_utc="",
            model_name="",
            text_col="",
            seed=0,
            n_train=0,
            n_val=0,
            n_test=0,
            max_length=0,
            truncation_policy="",
            learning_rate=0.0,
            warmup_ratio=0.0,
            weight_decay=0.0,
            per_device_train_batch_size=0,
            per_device_eval_batch_size=0,
            gradient_accumulation_steps=1,
            num_train_epochs=0.0,
            early_stopping_patience=0,
            class_weights="{}",
            metric_for_best_model="",
            best_model_checkpoint="",
            best_metric_value=float("nan"),
            best_epoch=None,
            val_metrics="{}",
            test_metrics="{}",
            device="",
            gpu_name="",
            transformers_version="",
            torch_version="",
            python_version="",
            platform="",
            wall_time_seconds=0.0,
        )

    def start_timer(self):
        self._t0 = time.time()

    def stop_timer(self) -> float:
        if self._t0 is None:
            return 0.0
        return time.time() - self._t0

    def append(self, record: RunRecord):
        # Write JSON sidecar too (useful for full details)
        json_path = os.path.join(self.out_dir, f"{record.run_id}.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(asdict(record), jf, indent=2)

        # Append to CSV
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self._empty_record()).keys()))
            writer.writerow(asdict(record))

        return json_path


# ----------------------------
# 3) Helper to extract metrics
# ----------------------------
def _jsonify_metrics(metrics: Dict[str, Any]) -> str:
    # Trainer returns keys like eval_loss, eval_p_at_50, etc.
    # Convert non-serializable values safely.
    clean = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            clean[k] = v
        else:
            try:
                clean[k] = float(v)
            except Exception:
                clean[k] = str(v)
    return json.dumps(clean, sort_keys=True)


def _get_env_info(transformers) -> Dict[str, str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    return {
        "device": device,
        "gpu_name": gpu_name,
        "transformers_version": getattr(transformers, "__version__", ""),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


# ----------------------------
# 4) Main function to log a run
# ----------------------------
def log_run_results(
        *,
        logger: SweepLogger,
        transformers_module,
        run_id: str,
        model_name: str,
        text_col: str,
        seed: int,
        train_tok,
        val_tok,
        test_tok,
        max_length: int,
        truncation_policy: str,
        training_args,  # your TrainingArguments object
        class_weights: Dict[int, float],
        early_stopping_patience: int,
        trainer,  # your (Weighted)Trainer AFTER trainer.train()
) -> RunRecord:
    wall = logger.stop_timer()

    # Evaluate with the final loaded checkpoint (should be best if load_best_model_at_end=True)
    val_metrics = trainer.evaluate(val_tok)
    test_metrics = trainer.evaluate(test_tok)

    # Best checkpoint info
    best_ckpt = getattr(trainer.state, "best_model_checkpoint", "") or ""
    best_metric = getattr(trainer.state, "best_metric", float("nan"))
    best_epoch = getattr(trainer.state, "epoch", None)

    env = _get_env_info(transformers_module)

    record = RunRecord(
        run_id=run_id,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),

        model_name=model_name,
        text_col=text_col,
        seed=int(seed),
        n_train=len(train_tok),
        n_val=len(val_tok),
        n_test=len(test_tok),

        max_length=int(max_length),
        truncation_policy=str(truncation_policy),

        learning_rate=float(getattr(training_args, "learning_rate", 0.0)),
        warmup_ratio=float(getattr(training_args, "warmup_ratio", 0.0)),
        weight_decay=float(getattr(training_args, "weight_decay", 0.0)),
        per_device_train_batch_size=int(getattr(training_args, "per_device_train_batch_size", 0)),
        per_device_eval_batch_size=int(getattr(training_args, "per_device_eval_batch_size", 0)),
        gradient_accumulation_steps=int(getattr(training_args, "gradient_accumulation_steps", 1)),
        num_train_epochs=float(getattr(training_args, "num_train_epochs", 0.0)),
        early_stopping_patience=int(early_stopping_patience),
        class_weights=json.dumps(class_weights, sort_keys=True),

        metric_for_best_model=str(getattr(training_args, "metric_for_best_model", "")),
        best_model_checkpoint=best_ckpt,
        best_metric_value=float(best_metric) if best_metric is not None else float("nan"),
        best_epoch=float(best_epoch) if best_epoch is not None else None,

        val_metrics=_jsonify_metrics(val_metrics),
        test_metrics=_jsonify_metrics(test_metrics),

        device=env["device"],
        gpu_name=env["gpu_name"],
        transformers_version=env["transformers_version"],
        torch_version=env["torch_version"],
        python_version=env["python_version"],
        platform=env["platform"],

        wall_time_seconds=float(wall),
    )

    json_path = logger.append(record)
    print(f"Logged run to CSV: {logger.csv_path}")
    print(f"Logged run JSON:  {json_path}")
    return record


# ----------------------------
# 5) Example usage pattern
# ----------------------------

from transformers import TrainingArguments, EarlyStoppingCallback
import transformers

logger = SweepLogger(out_dir="runs_logs")
logger.start_timer()

# ... build tokenizer/model/datasets ...
# ... create training_args ...
# ... create trainer ...
# trainer.train()

record = log_run_results(
    logger=logger,
    transformers_module=transformers,
    run_id="distilbert_lr2e-5_wu0.1_seed42",
    model_name="distilbert-base-uncased",
    text_col="all_text",
    seed=42,
    train_tok=train_tok,
    val_tok=val_tok,
    test_tok=test_tok,
    max_length=512,
    truncation_policy="head",
    training_args=training_args,
    class_weights={0: 0.6046, 1: 2.8892},
    early_stopping_patience=2,
    trainer=trainer,
)


# -----------------------------
# Post-training evaluation (using the best checkpoint selected on validation loss)
# -----------------------------
assert "trainer" in globals(), "Run training cell first (Trainer not found)."
assert "val_tok" in globals() and "test_tok" in globals(), "Run tokenization cell first (val_tok/test_tok not found)."

print("\n--- Evaluate on validation split ---")
val_results = trainer.evaluate(eval_dataset=val_tok)
print(val_results)

print("\n--- Evaluate on held-out test split ---")
test_results = trainer.evaluate(eval_dataset=test_tok)
print(test_results)



