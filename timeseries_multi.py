import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from config import PROJECT_ROOT, SUMMARY_ID, ANCHOR_PUSH_COL, ANCHOR_REV_COL
import concurrent.futures
from tqdm import tqdm

WINDOW_CONFIGS = [
    (5, 5, "short"),
    (20, 10, "med"),
    (50, 20, "long")
]


class TimeSeriesManager:
    def __init__(self):
        self.ts_root = self._find_ts_root()
        self.ts_files = list(self.ts_root.rglob("*_timeseries_data.csv"))
        self.ts_index, self.sig_to_folders = self._index_files()
        self.folders = sorted(self.ts_index.keys())
        print(f"TS Manager init: Found {len(self.ts_files)} files in {len(self.folders)} folders.")

    def _find_ts_root(self):
        candidates = [p for p in PROJECT_ROOT.rglob("timeseries-data") if p.is_dir()]
        if not candidates:
            raise FileNotFoundError("timeseries-data folder not found")
        return candidates[0]

    def _index_files(self):
        idx = defaultdict(dict)
        rev_map = defaultdict(list)
        for p in self.ts_files:
            name = p.name.replace("_timeseries_data.csv", "")
            try:
                sig_id = int(name)
                folder = p.parent.name
                idx[folder][sig_id] = p
                rev_map[sig_id].append(folder)
            except ValueError:
                continue
        return idx, rev_map

    def resolve_path(self, sig_id: int, row: pd.Series) -> Path | None:
        repo = str(row.get("alert_summary_repository", "")).lower()
        fw = str(row.get("alert_summary_framework", "")).lower()
        hay = repo + " " + fw

        candidates = [f for f in self.folders if f.lower() in hay]
        if "autoland" in hay: candidates.extend([f for f in self.folders if f.startswith("autoland")])

        for f in candidates:
            if p := self.ts_index.get(f, {}).get(sig_id): return p

        for f in self.sig_to_folders.get(sig_id, []):
            if p := self.ts_index.get(f, {}).get(sig_id): return p
        return None


def load_timeseries_file(path_str: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path_str, engine="pyarrow")
        df["value"] = pd.to_numeric(df.get("value"), errors="coerce")
        if "push_id" in df.columns:
            df["push_id"] = pd.to_numeric(df["push_id"], errors="coerce")
        return df.dropna(subset=["value"])
    except Exception:
        return pd.DataFrame()


def fast_slope(y):
    n = len(y)
    if n < 2: return 0.0
    x = np.arange(n)
    sum_x = n * (n - 1) / 2
    sum_x2 = (n - 1) * n * (2 * n - 1) / 6
    sum_y = np.sum(y)
    sum_xy = np.dot(x, y)
    denom = (n * sum_x2 - sum_x ** 2)
    if denom == 0: return 0.0
    return (n * sum_xy - sum_x * sum_y) / denom


def compute_window_features(ts: pd.DataFrame, anchor_push=None, anchor_rev=None, configs=None):
    if configs is None: configs = WINDOW_CONFIGS

    out_all = {}
    metrics = ["delta", "z", "slope", "std_ratio", "median_shift", "p90_delta"]
    for _, _, suffix in configs:
        for m in metrics:
            out_all[f"ts_{m}_{suffix}"] = np.nan

    if ts.empty: return out_all

    anchor_idx = None
    if anchor_push and "push_id" in ts.columns:
        matches = np.where(ts["push_id"].values <= float(anchor_push))[0]
        if len(matches): anchor_idx = matches[-1]

    if anchor_idx is None: return out_all

    vals = ts["value"].values

    for n_pre, n_post, suffix in configs:
        pre = vals[max(0, anchor_idx - n_pre):anchor_idx]
        post = vals[anchor_idx + 1: min(len(vals), anchor_idx + 1 + n_post)]

        if len(pre) < 2 or len(post) < 2:
            continue

        pre_mean, post_mean = np.mean(pre), np.mean(post)
        pre_med, post_med = np.median(pre), np.median(post)
        pre_std = np.std(pre, ddof=1)
        post_std = np.std(post, ddof=1)

        pre_p90 = np.percentile(pre, 90)
        post_p90 = np.percentile(post, 90)

        denom_mean = max(abs(pre_mean), 1e-9)
        out_all[f"ts_delta_{suffix}"] = (post_mean - pre_mean) / denom_mean

        if pre_std > 0:
            out_all[f"ts_z_{suffix}"] = (post_mean - pre_mean) / pre_std
        else:
            out_all[f"ts_z_{suffix}"] = 100.0 if abs(post_mean - pre_mean) > 1e-9 else 0.0

        slope_pre = fast_slope(pre)
        slope_post = fast_slope(post)
        out_all[f"ts_slope_{suffix}"] = slope_post - slope_pre

        out_all[f"ts_std_ratio_{suffix}"] = post_std / max(pre_std, 1e-9)
        out_all[f"ts_median_shift_{suffix}"] = (post_med - pre_med) / max(abs(pre_med), 1e-9)
        out_all[f"ts_p90_delta_{suffix}"] = (post_p90 - pre_p90) / max(abs(pre_p90), 1e-9)

    return out_all


def _process_single_row(args):
    row_dict, sigs, ts_mgr = args
    sid = row_dict[SUMMARY_ID]
    row_s = pd.Series(row_dict)

    row_feats = []
    for sig in sigs:
        p = ts_mgr.resolve_path(sig, row_s)
        if p:
            ts = load_timeseries_file(str(p))
            f = compute_window_features(
                ts,
                anchor_push=row_dict.get(ANCHOR_PUSH_COL),
                anchor_rev=row_dict.get(ANCHOR_REV_COL),
                configs=WINDOW_CONFIGS
            )
            row_feats.append(f)

    res = {SUMMARY_ID: sid, "ts_sig_used": len(row_feats)}
    if row_feats:
        df_f = pd.DataFrame(row_feats)
        for c in df_f.columns:
            res[f"{c}__mean_over_sigs"] = df_f[c].mean()
            res[f"{c}__max_over_sigs"] = df_f[c].max()
            if "p90" in c:
                res[f"{c}__min_over_sigs"] = df_f[c].min()

    return res


def enrich_with_ts_features(summary_df: pd.DataFrame, alerts_raw: pd.DataFrame) -> pd.DataFrame:
    cache_dir = Path("./derived_features")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "ts_features_multiscale_v2.parquet"

    if cache_file.exists():
        print(f"Loading cached TS features from {cache_file}")
        ts_feat_df = pd.read_parquet(cache_file)
        return summary_df.merge(ts_feat_df, on=SUMMARY_ID, how="left")

    print(f"Computing Optimised Multi-Scale TS features...")
    ts_mgr = TimeSeriesManager()

    sigs_per_summary = alerts_raw.groupby(SUMMARY_ID)["signature_id"].apply(
        lambda x: sorted(set(pd.to_numeric(x, errors="coerce").dropna().astype(int)))
    )

    work_items = []
    for row in summary_df.itertuples(index=False):
        row_dict = row._asdict()
        sid = row_dict[SUMMARY_ID]
        sigs = sigs_per_summary.get(sid, [])
        work_items.append((row_dict, sigs, ts_mgr))

    print(f"Launching parallel processing on {len(work_items)} items...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=14) as executor:
        results = list(tqdm(
            executor.map(_process_single_row, work_items, chunksize=50),
            total=len(work_items),
            desc="TS Extraction (Optimized)"
        ))

    features = results
    ts_df = pd.DataFrame(features)
    ts_df.to_parquet(cache_file, index=False)
    print(f"Computed and cached TS features: {ts_df.shape}")
    return summary_df.merge(ts_df, on=SUMMARY_ID, how="left")