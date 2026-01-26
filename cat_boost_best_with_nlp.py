import optuna
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold
from data_loader import load_raw_data, aggregate_alerts
from timeseries_multi import enrich_with_ts_features
from preprocessing import perform_time_split, enrich_and_weight_data, get_embeddings, engineer_complex_features
from model_utils import prepare_matrix_with_pca, make_pool

DB_URL = "sqlite:///optuna_icpe_with_notes.db"
STUDY_NAME = "catboost_with_notes_optimization"
N_TRIALS = 300
N_SPLITS = 5
N_REPEATS = 1


def load_and_prep_all():
    print("--- 1. Chargement & Engineering (WITH NOTES) ---")
    alerts, bugs = load_raw_data()
    df = aggregate_alerts(alerts)
    df = enrich_and_weight_data(df, bugs)
    df = engineer_complex_features(df)
    df = enrich_with_ts_features(df, alerts)
    df["bug_created"] = df["bug_created"].fillna(0).astype(int)

    print("--- 2. NLP Embeddings (Human Notes Only) ---")
    notes = df['alert_summary_notes'].fillna('').astype(str)
    notes = notes.str.replace(r'\s+', ' ', regex=True).str.strip()
    raw_embeddings = get_embeddings(notes.tolist())

    return df, raw_embeddings


FULL_DF, RAW_EMBEDDINGS = load_and_prep_all()
TRAIN_VAL_DF, TEST_DF = perform_time_split(FULL_DF)
TRAIN_VAL_EMBS = RAW_EMBEDDINGS[:len(TRAIN_VAL_DF)]
print(f"Data Ready: Train+Val={len(TRAIN_VAL_DF)}")


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

    pca_n = trial.suggest_int("pca_components", 10, 100)

    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)
    scores = []

    y_full = TRAIN_VAL_DF["bug_created"]
    indices = np.arange(len(TRAIN_VAL_DF))

    for i, (train_idx, val_idx) in enumerate(rskf.split(indices, y_full)):
        df_train = TRAIN_VAL_DF.iloc[train_idx]
        df_val = TRAIN_VAL_DF.iloc[val_idx]

        emb_train = TRAIN_VAL_EMBS[train_idx]
        emb_val = TRAIN_VAL_EMBS[val_idx]

        X_train, cat_cols, pca_model = prepare_matrix_with_pca(
            df_train, emb_train, n_components=pca_n, is_train=True
        )
        X_val, _, _ = prepare_matrix_with_pca(
            df_val, emb_val, pca_model=pca_model, n_components=pca_n, is_train=False
        )

        w_train = TRAIN_VAL_DF['sample_weight'].iloc[train_idx]
        w_val = TRAIN_VAL_DF['sample_weight'].iloc[val_idx]

        train_pool = make_pool(X_train, df_train["bug_created"], cat_cols, w_train)
        val_pool = make_pool(X_val, df_val["bug_created"], cat_cols, w_val)

        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool)

        preds = model.predict_proba(val_pool)[:, 1]
        score = average_precision_score(df_val["bug_created"], preds)
        scores.append(score)

        trial.report(np.mean(scores), i)
        if trial.should_prune(): raise optuna.exceptions.TrialPruned()

    return np.mean(scores)


if __name__ == "__main__":
    print(f"--- Start Optuna WITH NOTES ---")
    storage = optuna.storages.RDBStorage(url=DB_URL)
    study = optuna.create_study(study_name=STUDY_NAME, storage=storage, direction="maximize", load_if_exists=True)
    study.optimize(objective, n_trials=N_TRIALS)
    print("Best Params:", study.best_params)