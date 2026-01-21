import optuna
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold
from data_loader import load_raw_data, aggregate_alerts
from timeseries_multi import enrich_with_ts_features
from preprocessing import perform_time_split, enrich_and_weight_data, get_embeddings, engineer_complex_features
from model_utils import prepare_matrix_with_pca, make_pool, calculate_pr_at_k

# --- CONFIGURATION ---
DB_URL = "sqlite:///optuna_icpe_fasttext-compressed_preprocessing.db"
STUDY_NAME = "catboost_hybrid_v1_fasttext-compressed_preprocessing"
N_TRIALS = 500
N_SPLITS = 5
N_REPEATS = 1


def load_and_prep_all():
    print("--- 1. Chargement & Engineering ---")
    alerts, bugs = load_raw_data()
    df = aggregate_alerts(alerts)
    df = enrich_and_weight_data(df, bugs)
    df = engineer_complex_features(df)
    df = enrich_with_ts_features(df, alerts)
    df["bug_created"] = df["bug_created"].fillna(0).astype(int)

    print("--- 2. NLP Embeddings (Technical Context - NO CHEATING) ---")
    # On construit le contexte technique (disponible en temps réel)
    # Remplacement des notes humaines par : Repo + Framework + Suite + Test
    tech_context = (
            df['alert_summary_repository'].fillna('').astype(str) + " " +
            df['alert_summary_framework'].fillna('').astype(str) + " " +
            df['single_alert_series_signature_suite__mode'].fillna('').astype(str) + " " +
            df['single_alert_series_signature_test__mode'].fillna('').astype(str)
    )
    # Nettoyage des espaces
    tech_context = tech_context.str.replace(r'\s+', ' ', regex=True).str.strip()

    print(f"   Exemple de contexte: '{tech_context.iloc[0]}'")
    raw_embeddings = get_embeddings(tech_context.tolist())

    return df, raw_embeddings


# Chargement Global (une seule fois au début)
FULL_DF, RAW_EMBEDDINGS = load_and_prep_all()
TRAIN_VAL_DF, TEST_DF = perform_time_split(FULL_DF)
TRAIN_VAL_EMBS = RAW_EMBEDDINGS[:len(TRAIN_VAL_DF)]
TEST_EMBS = RAW_EMBEDDINGS[len(TRAIN_VAL_DF):]

print(f"Data Ready: Train+Val={len(TRAIN_VAL_DF)}, Test={len(TEST_DF)}")


def objective(trial):
    # --- 1. Hyperparamètres ---
    grow_policy = trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"])

    params = {
        "iterations": 2000,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-1, 20, log=True),
        "border_count": trial.suggest_categorical("border_count", [32, 64, 128, 254]),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100),
        # On garde une plage large pour gérer le déséquilibre
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 50.0, log=True),
        "grow_policy": grow_policy,
        "eval_metric": "AUC",
        "early_stopping_rounds": 100,
        "verbose": 0,
        "task_type": "CPU",
        "thread_count": 4,
        "one_hot_max_size": trial.suggest_categorical("one_hot_max_size", [2, 10, 50])
    }

    # Params spécifiques
    if grow_policy == "Lossguide":
        params["max_leaves"] = trial.suggest_int("max_leaves", 16, 64)

    bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])
    params["bootstrap_type"] = bootstrap_type

    if bootstrap_type == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 10.0)
    elif bootstrap_type in ["Bernoulli", "MVS"]:
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)

    # Paramètre de Preprocessing
    pca_n = trial.suggest_int("pca_components", 20, 150)

    # --- 2. Cross Validation (Stratégie Vectorisée) ---
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=42)
    scores = []

    # Préparation des matrices (PCA calculée à la volée ou pré-calculée ?)
    # Pour être rigoureux : PCA fit sur Train du Fold, transform sur Val du Fold.
    # C'est un peu plus lent mais c'est la "Golden Standard" pour éviter le Data Leakage de la PCA.

    y_full = TRAIN_VAL_DF["bug_created"]
    indices = np.arange(len(TRAIN_VAL_DF))

    for i, (train_idx, val_idx) in enumerate(rskf.split(indices, y_full)):
        # Split des données brutes
        df_train = TRAIN_VAL_DF.iloc[train_idx]
        df_val = TRAIN_VAL_DF.iloc[val_idx]

        emb_train = TRAIN_VAL_EMBS[train_idx]
        emb_val = TRAIN_VAL_EMBS[val_idx]

        # Préparation via model_utils (Gère le nettoyage, NaN, PCA)
        X_train, cat_cols, pca_model = prepare_matrix_with_pca(
            df_train, emb_train, n_components=pca_n, is_train=True
        )

        X_val, _, _ = prepare_matrix_with_pca(
            df_val, emb_val, pca_model=pca_model, n_components=pca_n, is_train=False
        )

        # Poids
        w_train = df_train['sample_weight']
        w_val = df_val['sample_weight']

        # Création des Pools CatBoost
        train_pool = make_pool(X_train, df_train["bug_created"], cat_cols, w_train)
        val_pool = make_pool(X_val, df_val["bug_created"], cat_cols, w_val)

        # Entraînement
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool)

        # Évaluation
        preds = model.predict_proba(val_pool)[:, 1]
        score = average_precision_score(df_val["bug_created"], preds)
        scores.append(score)

        # Pruning Optuna
        trial.report(np.mean(scores), i)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return np.mean(scores)


def main():
    print(f"--- Start Optuna ({N_TRIALS} trials, {N_SPLITS} folds) ---")
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
    print("BEST RESULT")
    print(f"Best AUPRC (CV Mean): {study.best_value:.4f}")
    print("Best Params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print("=" * 50)

    # --- Ré-entraînement Final ---
    print("Training Final Model (Full Train+Val)...")
    best_params = study.best_params.copy()
    best_pca_n = best_params.pop("pca_components")

    # 1. Prep Full Dataset (Train + Val)
    X_full, cat_cols, final_pca = prepare_matrix_with_pca(
        TRAIN_VAL_DF, TRAIN_VAL_EMBS, n_components=best_pca_n, is_train=True
    )

    # 2. Split Interne pour l'Early Stopping (90/10)
    split_idx = int(len(X_full) * 0.90)

    X_train_int = X_full.iloc[:split_idx]
    y_train_int = TRAIN_VAL_DF["bug_created"].iloc[:split_idx]
    w_train_int = TRAIN_VAL_DF["sample_weight"].iloc[:split_idx]

    X_val_int = X_full.iloc[split_idx:]
    y_val_int = TRAIN_VAL_DF["bug_created"].iloc[split_idx:]
    w_val_int = TRAIN_VAL_DF["sample_weight"].iloc[split_idx:]

    train_pool = make_pool(X_train_int, y_train_int, cat_cols, w_train_int)
    val_pool = make_pool(X_val_int, y_val_int, cat_cols, w_val_int)

    # 3. Fit
    final_cb_params = best_params.copy()
    final_cb_params.update({"iterations": 2000, "verbose": 100})

    model = CatBoostClassifier(**final_cb_params)
    model.fit(train_pool, eval_set=val_pool)

    # 4. Save
    model.save_model("catboost_best_optuna.cbm")
    print("Model saved to catboost_best_optuna.cbm")

    # 5. Final Check on Test (Juste pour info)
    print("Verification sur Test Set (Informatif)...")
    X_test, _, _ = prepare_matrix_with_pca(
        TEST_DF, TEST_EMBS, pca_model=final_pca, n_components=best_pca_n, is_train=False
    )
    test_pool = make_pool(X_test, TEST_DF["bug_created"], cat_cols)
    probs = model.predict_proba(test_pool)[:, 1]
    final_score = average_precision_score(TEST_DF["bug_created"], probs)
    print(f"Final Test AUPRC (Realistic): {final_score:.4f}")
    print("-" * 40)
    print(f"{'Metric':<10} | {'Value':<10}")
    print("-" * 40)
    for k in [50, 100, 200]:
        p, r = calculate_pr_at_k(TEST_DF["bug_created"], probs, k)
        print(f"P@{k:<4}     | {p:.4f}")
        print(f"R@{k:<4}     | {r:.4f}")
    print("-" * 40)


if __name__ == "__main__":
    main()