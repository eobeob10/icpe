import optuna
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold
from data_loader import load_raw_data, aggregate_alerts
from timeseries_multi import enrich_with_ts_features
from config import TRAIN_CONFIG
from preprocessing import perform_time_split, enrich_and_weight_data, get_embeddings, engineer_complex_features

DB_URL = f"sqlite:///optuna_icpe_fasttext-compressed_preprocessing.db"
STUDY_NAME = f"catboost_hybrid_v1_fasttext-compressed_preprocessing"
N_TRIALS = 500
N_SPLITS = 3
N_REPEATS = 1

BATCH_SIZE = TRAIN_CONFIG.get("batch_size_eval", 32)


def prepare_data():
    print("--- 1. Chargement & Engineering ---")
    alerts, bugs = load_raw_data()
    df = aggregate_alerts(alerts)

    df = enrich_and_weight_data(df, bugs)

    df = engineer_complex_features(df)

    df = enrich_with_ts_features(df, alerts)
    df["bug_created"] = df["bug_created"].fillna(0).astype(int)

    # 2. Embeddings
    print("--- 2. NLP Embeddings ---")
    notes = df["alert_summary_notes"].tolist()
    raw_embeddings = get_embeddings(notes)

    return df, raw_embeddings


FULL_DF, RAW_EMBEDDINGS = prepare_data()
TRAIN_VAL_DF, TEST_DF = perform_time_split(FULL_DF)

n_train = len(TRAIN_VAL_DF)
TRAIN_VAL_EMBS = RAW_EMBEDDINGS[:n_train]
TEST_EMBS = RAW_EMBEDDINGS[n_train:]

print(f"Data Ready: Train+Val={len(TRAIN_VAL_DF)}, Test={len(TEST_DF)}")


def objective(trial):
    grow_policy = trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"])

    params = {
        "iterations": 2000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-1, 20, log=True),
        "border_count": trial.suggest_categorical("border_count", [32, 64, 128, 254]),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 50.0, log=True),
        "grow_policy": grow_policy,
        "eval_metric": "AUC",
        "early_stopping_rounds": 100,
        "verbose": 0,
        "task_type": "CPU",
        "thread_count": 4
    }

    if grow_policy == "Lossguide":
        params["max_leaves"] = trial.suggest_int("max_leaves", 16, 64)

    bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])
    params["bootstrap_type"] = bootstrap_type

    if bootstrap_type == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 10.0)
    elif bootstrap_type in ["Bernoulli", "MVS"]:
        params["subsample"] = trial.suggest_float("subsample", 0.1, 1.0)

    params["one_hot_max_size"] = trial.suggest_categorical("one_hot_max_size", [2, 10, 50])
    pca_n = trial.suggest_int("pca_components", 20, 600)

    target_col = "bug_created"

    ignore_cols = [
            "bug_id", "alert_summary_id", target_col, "alert_summary_notes",
            "push_timestamp", "alert_summary_creation_timestamp", "sample_weight",
            "seconds_since_last_push"  # On ignore la feature brute, on garde le log,
            # Dates techniques inutilisables
            'alert_summary_triage_due_date',
            'alert_summary_bug_due_date',
            'alert_summary_bug_updated',  # Souvent présent, fuite possible

            # Identifiants uniques (Bruit / Overfitting)
            'alert_summary_push_id',
            'alert_summary_prev_push_id',
            'alert_summary_revision',
            'alert_summary_prev_push_revision',
            'single_alert_series_signature_signature_hash__mode',
            'single_alert_series_signature_option_collection_hash__mode',

            # Constantes ou ID internes
            'alert_summary_issue_tracker',
            'single_alert_id__min',
            'single_alert_id__max',
            'single_alert_summary_id__mode'
        ]


    features_static = [c for c in TRAIN_VAL_DF.columns if c not in ignore_cols]

    cat_cols_static = [c for c in features_static if
                       (TRAIN_VAL_DF[c].dtype == "object" or TRAIN_VAL_DF[c].dtype.name == "category")
                       and c != "sample_weight"]

    df_static = TRAIN_VAL_DF.copy()
    for c in cat_cols_static:
        df_static[c] = df_static[c].fillna("MISSING").astype(str)

    X_static = df_static[features_static]
    y = df_static[target_col]

    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)
    auprc_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(X_static, y)):
        X_train_stat, X_val_stat = X_static.iloc[train_idx], X_static.iloc[val_idx]
        y_train_f, y_val_f = y.iloc[train_idx], y.iloc[val_idx]

        w_train = df_static['sample_weight'].iloc[train_idx]
        w_val = df_static['sample_weight'].iloc[val_idx]

        emb_train = TRAIN_VAL_EMBS[train_idx]
        emb_val = TRAIN_VAL_EMBS[val_idx]
        pca = PCA(n_components=pca_n, random_state=42)
        emb_train_pca = pca.fit_transform(emb_train)
        emb_val_pca = pca.transform(emb_val)
        pca_cols = [f"pca_{i}" for i in range(pca_n)]
        df_train_pca = pd.DataFrame(emb_train_pca, columns=pca_cols, index=X_train_stat.index)
        df_val_pca = pd.DataFrame(emb_val_pca, columns=pca_cols, index=X_val_stat.index)

        X_train_final = pd.concat([X_train_stat, df_train_pca], axis=1)
        X_val_final = pd.concat([X_val_stat, df_val_pca], axis=1)

        #X_train_clean = X_train_final.drop(columns=['sample_weight'])
        #X_val_clean = X_val_final.drop(columns=['sample_weight'])

        train_pool = Pool(X_train_final, label=y_train_f, weight=w_train, cat_features=cat_cols_static)
        val_pool = Pool(X_val_final, label=y_val_f, weight=w_val, cat_features=cat_cols_static)

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool)

        probs = model.predict_proba(val_pool)[:, 1]

        score = average_precision_score(y_val_f, probs)

        auprc_scores.append(score)

        trial.report(np.mean(auprc_scores), fold_idx)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(auprc_scores)

def main():
    print(f"--- Start optuna Optuna ({N_TRIALS} trials) ---")
    print(f"BDD: {DB_URL}")

    storage = optuna.storages.RDBStorage(url=DB_URL)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)
    )

    study.optimize(objective, n_trials=N_TRIALS)

    print("\n" + "=" * 50)
    print("BEST RES")
    print(f"Best AUPRC (CV Mean): {study.best_value:.4f}")
    print("Best Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 50)

    print("Final retrain")

    best_params = study.best_params.copy()
    best_pca_n = best_params.pop("pca_components")

    final_cb_params = {
        "iterations": 2000,
        "eval_metric": "AUC",
        "early_stopping_rounds": 100,
        "verbose": 100,
        "task_type": "CPU",
        "thread_count": 4,
        **best_params
    }

    target_col = "bug_created"
    ignore_cols = ["bug_id", "alert_summary_id", target_col, "alert_summary_notes"]
    features_static = [c for c in TRAIN_VAL_DF.columns if c not in ignore_cols]
    cat_cols_static = [c for c in features_static if
                       TRAIN_VAL_DF[c].dtype == "object" or TRAIN_VAL_DF[c].dtype.name == "category"]

    X_static_train_val = TRAIN_VAL_DF[features_static].copy()
    y_train_val = TRAIN_VAL_DF[target_col]
    for c in cat_cols_static:
        X_static_train_val[c] = X_static_train_val[c].fillna("MISSING").astype(str)

    print(f"N DIM PCA (n={best_pca_n})")
    pca = PCA(n_components=best_pca_n, random_state=42)
    emb_train_val_pca = pca.fit_transform(TRAIN_VAL_EMBS)

    pca_cols = [f"pca_{i}" for i in range(best_pca_n)]
    df_train_val_pca = pd.DataFrame(emb_train_val_pca, columns=pca_cols, index=X_static_train_val.index)

    X_final_full = pd.concat([X_static_train_val, df_train_val_pca], axis=1)

    split_idx = int(len(X_final_full) * 0.90)

    X_train_internal = X_final_full.iloc[:split_idx]
    y_train_internal = y_train_val.iloc[:split_idx]

    X_val_internal = X_final_full.iloc[split_idx:]
    y_val_internal = y_train_val.iloc[split_idx:]

    print(f"Split Early Stopping : Train={len(X_train_internal)}, Val={len(X_val_internal)}")

    train_pool_internal = Pool(X_train_internal, label=y_train_internal, cat_features=cat_cols_static)
    val_pool_internal = Pool(X_val_internal, label=y_val_internal, cat_features=cat_cols_static)

    model = CatBoostClassifier(**final_cb_params)
    model.fit(train_pool_internal, eval_set=val_pool_internal)

    print("FINAL EVALUATION ON TEST SET (never seen)")

    X_static_test = TEST_DF[features_static].copy()
    y_test = TEST_DF[target_col]
    for c in cat_cols_static:
        X_static_test[c] = X_static_test[c].fillna("MISSING").astype(str)

    emb_test_pca = pca.transform(TEST_EMBS)
    df_test_pca = pd.DataFrame(emb_test_pca, columns=pca_cols, index=X_static_test.index)
    X_final_test = pd.concat([X_static_test, df_test_pca], axis=1)

    test_pool = Pool(X_final_test, label=y_test, cat_features=cat_cols_static)

    probs = model.predict_proba(test_pool)[:, 1]
    final_auprc = average_precision_score(y_test, probs)

    print(f"FINAL TEST AUPRC: {final_auprc:.4f}")

    def precision_recall_at_k_local(y_true, y_score, k):
        k = int(min(k, len(y_true)))
        idx = np.argsort(-y_score)[:k]
        y_top = np.array(y_true)[idx]
        return float(y_top.mean()), float(y_top.sum() / max(1, np.sum(y_true)))

    print("-" * 30)
    for k in [50, 100, 200]:
        p, r = precision_recall_at_k_local(y_test, probs, k)
        print(f"P@{k}: {p:.4f} | R@{k}: {r:.4f}")
    print("-" * 30)

    model.save_model("catboost_best_optuna.cbm")
    print("Model saved")


if __name__ == "__main__":
    main()