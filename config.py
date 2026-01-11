from pathlib import Path

PROJECT_ROOT = Path.cwd().resolve().parent.parent / "icpe_data"

SUMMARY_ID = "alert_summary_id"
BUG_ID_COL = "alert_summary_bug_number"
ANCHOR_PUSH_COL = "alert_summary_push_id"
ANCHOR_REV_COL = "alert_summary_revision"

TRAIN_CONFIG = {
    "seed": 42,
    "max_len": 1024,
    "epochs": 6,
    "batch_size_train": 8,
    "batch_size_eval": 16,
    "learning_rate": 2e-5,
    "weight_decay": 0.1,
    "warmup_ratio": 0.1,
    "test_frac": 0.20,
    "val_frac": 0.10,
    "text_col": "all_text"
}