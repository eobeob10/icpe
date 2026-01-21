from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve().parent.parent / "icpe_data"

SUMMARY_ID = "alert_summary_id"
BUG_ID_COL = "alert_summary_bug_number"
ANCHOR_PUSH_COL = "alert_summary_push_id"
ANCHOR_REV_COL = "alert_summary_revision"

TRAIN_CONFIG = {
    "batch_size_eval": 16,
    "test_frac": 0.20,
}