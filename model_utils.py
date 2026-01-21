import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from catboost import Pool


def get_clean_feature_names(df, ignore_cols):
    """Retourne la liste des features statiques en excluant les colonnes ignorées/toxiques."""
    cols_to_ban = [
        # --- 1. Dates brutes & ID Techniques (Bruit) ---
        'alert_summary_triage_due_date',
        'alert_summary_bug_due_date',
        'alert_summary_bug_updated',
        'alert_summary_creation_timestamp',
        'push_timestamp',
        'alert_summary_push_id',
        'alert_summary_prev_push_id',
        'alert_summary_revision',
        'alert_summary_prev_push_revision',
        'single_alert_series_signature_signature_hash__mode',
        'single_alert_series_signature_option_collection_hash__mode',
        'single_alert_id__min',
        'single_alert_id__max',
        'single_alert_summary_id__mode',
        'alert_summary_issue_tracker',

        # --- 2. DATA LEAKAGE MAJEUR (Contenu Humain Futur) ---
        'alert_summary_notes',  # Le texte des notes
        'notes_len_chars',  # La longueur trahit la présence d'une note
        'notes_len_words',  # Idem

        # --- 3. DATA LEAKAGE MAJEUR (Meta-données Humaines Futures) ---
        'single_alert_classifier__mode',  # L'identité de la personne qui a trié
        'single_alert_classifier__nunique'  # Le nombre de personnes ayant touché l'alerte
    ]

    # Filtrage
    candidates = [c for c in df.columns if c not in ignore_cols and c not in cols_to_ban]
    return candidates


def prepare_matrix_with_pca(df, embeddings, pca_model=None, n_components=50, is_train=True):
    """
    Prépare la matrice X finale (Features statiques + PCA des embeddings).
    """
    target_col = "bug_created"

    # Colonnes à ignorer lors de la sélection des features (mais gardées dans le DF pour info)
    ignore_cols = ["bug_id", "alert_summary_id", target_col, "alert_summary_notes",
                   "push_timestamp", "alert_summary_creation_timestamp", "sample_weight",
                   "seconds_since_last_push"]

    # 1. Sélection des colonnes statiques (Nettoyée)
    features_static = get_clean_feature_names(df, ignore_cols)

    # 2. Séparation Catégoriel / Numérique
    cat_cols = [c for c in features_static if
                (df[c].dtype == "object" or df[c].dtype.name == "category")]

    # 3. Création de la partie Statique (FillNA pour CatBoost)
    X_static = df[features_static].copy()
    for c in cat_cols:
        X_static[c] = X_static[c].fillna("MISSING").astype(str)

    # 4. Gestion PCA (Embeddings)
    if is_train:
        pca_model = PCA(n_components=n_components, random_state=42)
        emb_pca = pca_model.fit_transform(embeddings)
    else:
        if pca_model is None:
            raise ValueError("pca_model must be provided for inference mode")
        emb_pca = pca_model.transform(embeddings)

    pca_cols = [f"pca_{i}" for i in range(n_components)]
    df_pca = pd.DataFrame(emb_pca, columns=pca_cols, index=X_static.index)

    # 5. Assemblage final
    X_final = pd.concat([X_static, df_pca], axis=1)

    return X_final, cat_cols, pca_model


def make_pool(X, y, cat_features, weights=None):
    """Helper rapide pour créer un CatBoost Pool."""
    return Pool(X, label=y, cat_features=cat_features, weight=weights)


def calculate_pr_at_k(y_true, y_score, k):
    """Calcule la Précision et le Rappel pour les k meilleurs scores."""
    # Conversion sécurisée numpy
    if hasattr(y_true, 'values'): y_true = y_true.values
    if hasattr(y_score, 'values'): y_score = y_score.values

    k = int(min(k, len(y_true)))
    # Tri décroissant des scores
    idx = np.argsort(-y_score)[:k]
    y_top = y_true[idx]

    # Precision : Moyenne des 1 dans le top k
    precision = float(y_top.mean())

    # Recall : Somme des 1 dans le top k / Somme totale des 1
    total_positives = y_true.sum()
    recall = float(y_top.sum() / max(1, total_positives))

    return precision, recall