import optuna
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold
from data_loader import load_raw_data, aggregate_alerts
from preprocessing import perform_time_split, enrich_and_weight_data, engineer_complex_features
from model_utils import prepare_matrix_with_pca, make_pool

DB_URL = "sqlite:///optuna_icpe_static_only.db"
STUDY_NAME = "catboost_static_only_optimization"
N_TRIALS = 300
N_SPLITS = 5
N_REPEATS = 1


def load_and_prep_static():
    print("--- 1. Chargement (STATIC ONLY) ---")
    alerts, bugs = load_raw_data()
    df = aggregate_alerts(alerts)
    df = enrich_and_weight_data(df, bugs)
    df = engineer_complex_features(df)

    if "bug_created" in df.columns:
        df["bug_created"] = df["bug_created"].fillna(0).astype(int)

    print(f"   Features disponibles (Static): {df.shape[1]}")
    return df, None


FULL_DF, _ = load_and_prep_static()
TRAIN_VAL_DF, TEST_DF = perform_time_split(FULL_DF)
print(f"Data Ready: Train+Val={len(TRAIN_VAL_DF)}, Test={len(TEST_DF)}")


def objective(trial):
    grow_policy = trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"])

    params = {
        "iterations": 2000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-1, 20, log=True),
        "border_count": trial.suggest_categorical("border_count", [32, 64, 128]),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 50.0, log=True),
        "grow_policy": grow_policy,
        "eval_metric": "AUC",
        "early_stopping_rounds": 100,
        "verbose": 0, "task_type": "CPU", "thread_count": 4,
        "one_hot_max_size": trial.suggest_categorical("one_hot_max_size", [2, 10, 50])
    }

    if grow_policy == "Lossguide": params["max_leaves"] = trial.suggest_int("max_leaves", 16, 64)
    bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])
    params["bootstrap_type"] = bootstrap_type

    if bootstrap_type == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 10.0)
    elif bootstrap_type in ["Bernoulli", "MVS"]:
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)
    scores = []

    X_full, cat_cols, _ = prepare_matrix_with_pca(TRAIN_VAL_DF, embeddings=None, is_train=True, blind_mode=True)
    y_full = TRAIN_VAL_DF["bug_created"]
    w_full = TRAIN_VAL_DF["sample_weight"]

    for i, (train_idx, val_idx) in enumerate(rskf.split(X_full, y_full)):
        X_train, y_train, w_train = X_full.iloc[train_idx], y_full.iloc[train_idx], w_full.iloc[train_idx]
        X_val, y_val, w_val = X_full.iloc[val_idx], y_full.iloc[val_idx], w_full.iloc[val_idx]

        train_pool = make_pool(X_train, y_train, cat_cols, w_train)
        val_pool = make_pool(X_val, y_val, cat_cols, w_val)

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool)

        preds = model.predict_proba(val_pool)[:, 1]
        score = average_precision_score(y_val, preds)
        scores.append(score)

        trial.report(np.mean(scores), i)
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()

    return np.mean(scores)


if __name__ == "__main__":
    print(f"--- Start Optuna STATIC ONLY ---")
    storage = optuna.storages.RDBStorage(url=DB_URL)
    study = optuna.create_study(study_name=STUDY_NAME, storage=storage, direction="maximize", load_if_exists=True)
    study.optimize(objective, n_trials=N_TRIALS)
    print("Best Params:", study.best_params)