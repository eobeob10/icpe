import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from catboost import Pool


def get_clean_feature_names(df, ignore_cols, blind_mode=False):
    cols_to_ban = [
        'alert_summary_triage_due_date', 'alert_summary_bug_due_date', 'alert_summary_bug_updated',
        'alert_summary_creation_timestamp', 'push_timestamp', 'alert_summary_push_id',
        'alert_summary_prev_push_id', 'alert_summary_revision', 'alert_summary_prev_push_revision',
        'single_alert_series_signature_signature_hash__mode',
        'single_alert_series_signature_option_collection_hash__mode',
        'single_alert_id__min', 'single_alert_id__max', 'single_alert_summary_id__mode',
        'alert_summary_issue_tracker',
        'alert_summary_notes', 'notes_len_chars', 'notes_len_words',
        'single_alert_classifier__mode', 'single_alert_classifier__nunique'
    ]

    if blind_mode:
        magnitude_leaks = [
            'single_alert_amount_abs__mean', 'single_alert_amount_abs__min', 'single_alert_amount_abs__max',
            'single_alert_amount_abs__std', 'single_alert_amount_abs__p90',
            'single_alert_amount_pct__mean', 'single_alert_amount_pct__min', 'single_alert_amount_pct__max',
            'single_alert_amount_pct__std', 'single_alert_amount_pct__p90',
            'single_alert_prev_value__mean', 'single_alert_prev_value__min', 'single_alert_prev_value__max',
            'single_alert_prev_value__std', 'single_alert_prev_value__p90',
            'single_alert_new_value__mean', 'single_alert_new_value__min', 'single_alert_new_value__max',
            'single_alert_new_value__std', 'single_alert_new_value__p90',
            'rcd_ctxt_zscore', 'rcd_ctxt_mean', 'rcd_ctxt_std', 'rcd_ctxt_slope', 'rcd_ctxt_count',
            'n_related_alerts' #
        ]
        cols_to_ban.extend(magnitude_leaks)

    candidates = [c for c in df.columns if c not in ignore_cols]
    return [c for c in candidates if c not in cols_to_ban]


def prepare_matrix_with_pca(df, embeddings=None, pca_model=None, n_components=20, is_train=True, blind_mode=False):
    target_col = "bug_created"
    ignore_cols = ["bug_id", "alert_summary_id", target_col, "alert_summary_notes",
                   "push_timestamp", "alert_summary_creation_timestamp", "sample_weight",
                   "seconds_since_last_push"]

    features_static = get_clean_feature_names(df, ignore_cols, blind_mode=blind_mode)
    cat_cols = [c for c in features_static if (df[c].dtype == "object" or df[c].dtype.name == "category")]

    X_final = df[features_static].copy()
    for c in cat_cols:
        X_final[c] = X_final[c].fillna("MISSING").astype(str)

    # PCA seulement si embeddings fournis
    if embeddings is not None:
        if is_train:
            pca_model = PCA(n_components=n_components, random_state=42)
            emb_pca = pca_model.fit_transform(embeddings)
        else:
            if pca_model is None: raise ValueError("pca_model missing for inference")
            emb_pca = pca_model.transform(embeddings)

        pca_cols = [f"pca_{i}" for i in range(n_components)]
        df_pca = pd.DataFrame(emb_pca, columns=pca_cols, index=X_final.index)
        X_final = pd.concat([X_final, df_pca], axis=1)

    return X_final, cat_cols, pca_model


def make_pool(X, y, cat_features, weights=None):
    return Pool(X, label=y, cat_features=cat_features, weight=weights)


def calculate_pr_at_k(y_true, y_score, k):
    if hasattr(y_true, 'values'): y_true = y_true.values
    if hasattr(y_score, 'values'): y_score = y_score.values
    k = int(min(k, len(y_true)))
    idx = np.argsort(-y_score)[:k]
    y_top = y_true[idx]
    precision = float(y_top.mean())
    total_positives = y_true.sum()
    recall = float(y_top.sum() / max(1, total_positives))
    return precision, recall