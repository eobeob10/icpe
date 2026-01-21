import time
import threading
import psutil
import os
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score
from sklearn.decomposition import PCA
from data_loader import load_raw_data, aggregate_alerts
from timeseries_multi import enrich_with_ts_features
from cleanlab.filter import find_label_issues
from preprocessing import perform_time_split, enrich_and_weight_data, get_embeddings, engineer_complex_features


# --- CONFIGURATION ---
OUTPUT_DIR = "benchmark_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tes meilleurs paramètres (Hardcoded)
BEST_PARAMS = {
    "grow_policy": "Lossguide",
    "learning_rate": 0.02078155874027329,
    "depth": 5,
    "l2_leaf_reg": 5.696201658338176,
    "border_count": 32,
    "min_data_in_leaf": 44,
    "max_leaves": 19,
    "scale_pos_weight": 1.4399585477497998,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.9069303366018339,
    "one_hot_max_size": 10,
    "pca_components": 50
}


# --- CLASSE DE MONITORING ---
class ResourceMonitor(threading.Thread):
    def __init__(self, interval=0.1):
        super().__init__()
        self.interval = interval
        self.stop_event = threading.Event()
        self.records = []
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())

    def run(self):
        while not self.stop_event.is_set():
            current_time = time.time() - self.start_time
            cpu_pct = psutil.cpu_percent(interval=None)
            mem_info = self.process.memory_info()
            ram_mb = mem_info.rss / (1024 * 1024)

            self.records.append({
                "time": current_time,
                "cpu": cpu_pct,
                "ram": ram_mb
            })
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()

    def get_dataframe(self):
        return pd.DataFrame(self.records)


# --- MAIN BENCHMARK ---
def main():
    print("🚀 Démarrage du Benchmark complet (v2)...")

    monitor = ResourceMonitor(interval=0.1)
    monitor.start()

    timings = {}
    step_starts = {}  # Pour stocker les timestamps relatifs
    monitor_start_ts = monitor.start_time

    # Pour le graphique final
    time_diff_data = None

    try:
        # --- ETAPE 1 : CHARGEMENT ---
        t_start = time.time()
        step_starts["Load"] = t_start - monitor_start_ts
        print("\n[Step 1] Loading & Basic Engineering...")
        alerts, bugs = load_raw_data()
        df = aggregate_alerts(alerts)
        df = enrich_and_weight_data(df, bugs)

        # Sauvegarde pour le graph de fin
        time_diff_data = df['seconds_since_last_push'].copy()

        df = engineer_complex_features(df)

        timings["1_Load_Data"] = time.time() - t_start
        print(f"   Done in {timings['1_Load_Data']:.2f}s")

        # --- ETAPE 2 : TIME SERIES ---
        t_start = time.time()
        step_starts["TS"] = t_start - monitor_start_ts
        print("\n[Step 2] Time Series Extraction...")
        df = enrich_with_ts_features(df, alerts)
        df["bug_created"] = df["bug_created"].fillna(0).astype(int)
        timings["2_Time_Series"] = time.time() - t_start
        print(f"   Done in {timings['2_Time_Series']:.2f}s")

        # --- ETAPE 3 : NLP ---
        t_start = time.time()
        step_starts["NLP"] = t_start - monitor_start_ts
        print("\n[Step 3] NLP Embeddings (FastText)...")
        notes = df["alert_summary_notes"].tolist()
        raw_embeddings = get_embeddings(notes)
        timings["3_NLP_Embeddings"] = time.time() - t_start
        print(f"   Done in {timings['3_NLP_Embeddings']:.2f}s")

        # --- ETAPE 4 : PREP & PCA ---
        t_start = time.time()
        step_starts["PCA"] = t_start - monitor_start_ts
        print("\n[Step 4] PCA & Splitting...")

        # Split Time
        train_val, test = perform_time_split(df)
        n_train = len(train_val)
        emb_train = raw_embeddings[:n_train]
        emb_test = raw_embeddings[n_train:]

        # Free memory
        del raw_embeddings
        gc.collect()

        # PCA
        best_pca_n = BEST_PARAMS["pca_components"]
        pca = PCA(n_components=best_pca_n, random_state=42)
        emb_train_pca = pca.fit_transform(emb_train)
        emb_test_pca = pca.transform(emb_test)

        pca_cols = [f"pca_{i}" for i in range(best_pca_n)]

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

        features_static = [c for c in train_val.columns if c not in ignore_cols]
        cat_cols = [c for c in features_static if
                    (train_val[c].dtype == "object" or train_val[c].dtype.name == "category")]

        X_train_static = train_val[features_static].copy()
        X_test_static = test[features_static].copy()

        for c in cat_cols:
            X_train_static[c] = X_train_static[c].fillna("MISSING").astype(str)
            X_test_static[c] = X_test_static[c].fillna("MISSING").astype(str)

        X_train_full = pd.concat([X_train_static, pd.DataFrame(emb_train_pca, columns=pca_cols, index=train_val.index)],
                                 axis=1)
        X_test_full = pd.concat([X_test_static, pd.DataFrame(emb_test_pca, columns=pca_cols, index=test.index)], axis=1)

        y_train = train_val[target_col]
        y_test = test[target_col]
        w_train = train_val['sample_weight'] if 'sample_weight' in train_val.columns else None

        timings["4_Prep_PCA"] = time.time() - t_start
        print(f"   Done in {timings['4_Prep_PCA']:.2f}s")

        # --- ETAPE 5 : TRAINING ---
        t_start = time.time()
        step_starts["Train"] = t_start - monitor_start_ts
        print("\n[Step 5] Training CatBoost...")

        train_params = BEST_PARAMS.copy()
        del train_params["pca_components"]

        split_idx = int(len(X_train_full) * 0.90)
        X_tr_int = X_train_full.iloc[:split_idx]
        y_tr_int = y_train.iloc[:split_idx]
        w_tr_int = w_train.iloc[:split_idx] if w_train is not None else None

        X_val_int = X_train_full.iloc[split_idx:]
        y_val_int = y_train.iloc[split_idx:]
        w_val_int = w_train.iloc[split_idx:] if w_train is not None else None

        train_pool = Pool(X_tr_int, label=y_tr_int, cat_features=cat_cols, weight=w_tr_int)
        val_pool = Pool(X_val_int, label=y_val_int, cat_features=cat_cols, weight=w_val_int)

        model = CatBoostClassifier(
            iterations=2000,
            eval_metric="AUC",
            early_stopping_rounds=100,
            verbose=100,
            task_type="CPU",
            thread_count=4,
            **train_params
        )
        model.fit(train_pool, eval_set=val_pool)

        timings["5_Training"] = time.time() - t_start
        print(f"   Done in {timings['5_Training']:.2f}s")

        # --- ETAPE 6 : INFERENCE ---
        t_start = time.time()
        step_starts["Infer"] = t_start - monitor_start_ts
        print("\n[Step 6] Inference on Test Set...")

        test_pool = Pool(X_test_full, cat_features=cat_cols)
        probs = model.predict_proba(test_pool)[:, 1]

        timings["6_Inference"] = time.time() - t_start
        print(f"   Done in {timings['6_Inference']:.2f}s")

        final_auprc = average_precision_score(y_test, probs)

        def precision_recall_at_k_local(y_true, y_score, k):
            k = int(min(k, len(y_true)))
            idx = np.argsort(-y_score)[:k]
            y_top = np.array(y_true)[idx]
            return float(y_top.mean()), float(y_top.sum() / max(1, np.sum(y_true)))

        print(f"   🏆 FINAL AUPRC: {final_auprc:.4f}")
        print("-" * 30)
        for k in [50, 100, 200]:
            p, r = precision_recall_at_k_local(y_test, probs, k)
            print(f"P@{k}: {p:.4f} | R@{k}: {r:.4f}")
        print("-" * 30)

        # --- ETAPE 8 : ANALYSE DU BRUIT (Cleanlab) ---
        print("\n[Step 8] Estimating Label Noise (Cleanlab)...")
        try:
            # Cleanlab a besoin d'une matrice (N, 2) pour les probas [Proba_0, Proba_1]
            # probs contient déjà la proba de la classe 1 (Bug)
            pred_probs = np.column_stack((1 - probs, probs))

            # Détection des erreurs
            issues = find_label_issues(
                labels=y_test.values,
                pred_probs=pred_probs,
                return_indices_ranked_by='self_confidence'
            )

            n_issues = len(issues)
            noise_rate = n_issues / len(y_test)

            print(f"   -> Analyse terminée.")
            print(f"   -> Labels suspects détectés : {n_issues} sur {len(y_test)} exemples.")
            print(f"   -> Taux de bruit estimé (Noise Rate) : {noise_rate:.2%}")

            if n_issues > 0:
                print("\n" + "=" * 60)
                print("🕵️‍♂️ INSPECTION DÉTAILLÉE DES 5 PREMIERS SUSPECTS")
                print("=" * 60)

                # On récupère les indices des 5 cas les plus flagrants
                top_issues_indices = issues[:5]

                # On itère pour afficher les détails
                for i, idx in enumerate(top_issues_indices):
                    # On récupère la ligne brute dans le dataframe 'test' (avant transformation)
                    row = test.iloc[idx]
                    pred_prob = probs[idx]

                    human_label = int(row['bug_created'])
                    model_verdict = "BUG" if pred_prob > 0.5 else "PAS BUG"
                    contradiction = "Faux Positif (Label=0, Modèle=1)" if human_label == 0 else "Faux Négatif (Label=1, Modèle=0)"

                    print(f"\n🔴 SUSPECT #{i + 1} (Indice Test: {idx})")
                    print(f"   ID                : {row['alert_summary_id']}")
                    print(f"   CONTRADICTION     : {contradiction}")
                    print(f"   CONFIDENCE MODÈLE : {pred_prob:.4f} (Le modèle est très sûr de lui)")

                    # 1. Le Texte (Feature NLP) - Souvent la clé de l'énigme
                    txt = str(row['alert_summary_notes']).replace('\n', ' ').strip()
                    if len(txt) > 300: txt = txt[:300] + "..."
                    print(f"   📝 TEXTE           : \"{txt}\"")

                    # 2. Les Features Clés (Top Importance)
                    print(f"   🔗 RELATED ALERTS  : {row.get('alert_summary_related_alerts', 'N/A')} (Feature #1)")
                    print(f"   📊 T-VALUE (Mean)  : {row.get('single_alert_t_value__mean', 'N/A')} (Score statistique)")
                    print(f"   🔧 FRAMEWORK       : {row.get('alert_summary_framework', 'N/A')}")

                print("=" * 60 + "\n")

                # --- ETAPE 9 : ANALYSE SPÉCIFIQUE DES TEXTES MANQUANTS (NAN) ---
                print("\n" + "=" * 60)
                print("🧩 ANALYSE DES TEXTES 'NAN' (MISSING NOTES)")
                print("=" * 60)

                # 1. On isole toutes les lignes où le texte est vide ou NaN
                # On gère les vrais NaN et les chaines vides ou "nan" string
                nan_mask = test['alert_summary_notes'].isna() | \
                           (test['alert_summary_notes'].astype(str).str.lower().str.strip() == 'nan') | \
                           (test['alert_summary_notes'].astype(str).str.strip() == '')

                nan_rows = test[nan_mask]
                nan_indices = nan_rows.index

                print(f"Stats sur les textes manquants :")
                print(
                    f"   -> Nombre total de lignes 'NaN' dans le Test : {len(nan_rows)} / {len(test)} ({len(nan_rows) / len(test):.1%})")

                if len(nan_rows) > 0:
                    # On regarde la vérité terrain pour ces lignes
                    n_bugs_real = nan_rows['bug_created'].sum()
                    ratio_bugs = n_bugs_real / len(nan_rows)

                    print(f"   -> Parmi ces 'NaN', il y a {n_bugs_real} vrais bugs.")
                    print(f"   -> Probabilité qu'un 'NaN' soit un bug (Ground Truth) : {ratio_bugs:.1%}")

                    # On regarde ce que le modèle en pense
                    # probs est un numpy array aligné avec test
                    # On doit récupérer les probs correspondant aux indices des nan_rows
                    # Attention: probs est un array, nan_rows est un DataFrame avec potentiellement un index discontinu
                    # Si 'test' a été reset_index, l'index correspond à la position dans probs.
                    # Dans benchmark.py, perform_time_split fait un reset_index(drop=True), donc les indices sont alignés [0, N]

                    nan_probs = probs[nan_rows.index]
                    avg_model_conf = np.mean(nan_probs)
                    print(f"   -> Confiance moyenne du modèle sur ces lignes : {avg_model_conf:.1%}")

                    print("\n🔍 ÉCHANTILLON DE LIGNES 'NAN' (Vrais Bugs vs Faux Positifs)")
                    print("-" * 60)

                    # On va afficher quelques cas intéressants
                    # Cas A : C'est un Bug (Label=1) et le modèle l'a vu (Prob > 0.5) -> Le modèle compense le manque de texte
                    # Cas B : C'est pas un Bug (Label=0) et le modèle s'est trompé (Prob > 0.5) -> Le manque de texte a piégé le modèle ?

                    # On trie par T-Value décroissante pour voir si c'est le couplage T-Value/Nan qui joue
                    nan_rows_sorted = nan_rows.sort_values('single_alert_t_value__mean', ascending=False).head(10)

                    for idx, row in nan_rows_sorted.iterrows():
                        pred_prob = probs[idx]
                        label = row['bug_created']

                        status = "✅ CORRECT" if (pred_prob > 0.5) == label else "❌ ERREUR"
                        type_err = ""
                        if status == "❌ ERREUR":
                            type_err = "(Faux Positif)" if label == 0 else "(Faux Négatif)"

                        print(
                            f"ID: {row['alert_summary_id']} | Label: {label} | Pred: {pred_prob:.4f} | {status} {type_err}")
                        print(
                            f"   📊 T-Value: {row.get('single_alert_t_value__mean', 'N/A'):.1f} | Related: {row.get('alert_summary_related_alerts', 'N/A')}")
                        print(f"   🔧 Framework: {row.get('alert_summary_framework', 'N/A')}")
                        print("-" * 30)
        except ImportError:
            print("   ⚠️ Cleanlab n'est pas installé. Lance 'pip install cleanlab' pour activer cette étape.")
        except Exception as e:
            print(f"   ⚠️ Erreur lors de l'analyse Cleanlab : {e}")


    finally:
        monitor.stop()
        monitor.join()

    # --- ANALYSE & PLOTS ---

        # AFFICHER LE TEXTE DES IMPORTANCES (C'est ça que je veux voir)
        print("\n" + "=" * 40)
        print("       TOP 20 FEATURE IMPORTANCE")
        print("=" * 40)

        feature_importance = model.get_feature_importance()
        feature_names = model.feature_names_

        # Créer un DataFrame pour trier
        fi_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
        fi_df = fi_df.sort_values(by='importance', ascending=False).head(20)

        print(fi_df.to_string(index=False))

        # Vérification rapide de la nouvelle colonne Z-Score
        if 'rcd_ctxt_zscore' in df.columns:
            print("\n--- Stats de rcd_ctxt_zscore ---")
            print(df['rcd_ctxt_zscore'].describe())
            print(f"Nombre de 1000.0 exacts : {(df['rcd_ctxt_zscore'] == 1000.0).sum()}")

    print("\n[Step 7] Generating Reports...")

    # 1. Feature Importance Grouped
    print("   -> Feature Importance Plot...")
    fi = model.get_feature_importance(type="PredictionValuesChange")
    fi_df = pd.DataFrame({"feature": X_test_full.columns, "importance": fi})
    pca_mask = fi_df["feature"].str.startswith("pca_")
    pca_importance = fi_df.loc[pca_mask, "importance"].sum()
    fi_grouped = fi_df[~pca_mask].copy()
    fi_grouped = pd.concat([
        fi_grouped,
        pd.DataFrame([{"feature": "Alert Summary Notes (NLP Embedding)", "importance": pca_importance}])
    ], ignore_index=True)
    fi_grouped = fi_grouped.sort_values("importance", ascending=False).head(20)


    plt.figure(figsize=(10, 8))
    sns.barplot(data=fi_grouped, x="importance", y="feature", color="#4c72b0")
    plt.title("Top 20 Features (With NLP Grouped)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_grouped.png"))
    plt.close()

    # 2. Resource Usage Plot (AMÉLIORÉ)
    print("   -> Resource Plot...")
    res_df = monitor.get_dataframe()
    if not res_df.empty:
        fig, ax1 = plt.subplots(figsize=(14, 7))

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('CPU Usage (%)', color='tab:red')
        ax1.plot(res_df['time'], res_df['cpu'], color='tab:red', label='CPU', linewidth=1)
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.grid(True, linestyle=':', alpha=0.6)

        ax2 = ax1.twinx()
        ax2.set_ylabel('RAM Usage (MB)', color='tab:blue')
        ax2.plot(res_df['time'], res_df['ram'], color='tab:blue', label='RAM', linewidth=1.5)
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        # Ajout des lignes verticales pour les étapes
        # On définit des couleurs de fond pour les zones
        sorted_steps = sorted(step_starts.items(), key=lambda x: x[1])
        colors = ['#f0f0f0', '#ffffff']  # alternance gris/blanc

        for i, (step_name, start_time) in enumerate(sorted_steps):
            ax1.axvline(x=start_time, color='black', linestyle='--', alpha=0.5)
            # Label
            y_pos = 105 if i % 2 == 0 else 98  # Alterner la hauteur des labels
            ax1.text(start_time + 0.5, y_pos, step_name, rotation=0, fontsize=9, fontweight='bold',
                     transform=ax1.get_xaxis_transform())

        plt.title("System Resource Usage Pipeline Breakdown")
        fig.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "resource_usage_annotated.png"))
        plt.close()

        max_ram = res_df['ram'].max()
        avg_cpu = res_df['cpu'].mean()

    # 3. New Graph: Time Distribution
    print("   -> Commit Time Distribution Plot...")
    if time_diff_data is not None:
        # Convertir en heures pour la lisibilité
        hours_diff = time_diff_data / 3600
        plt.figure(figsize=(10, 6))
        # Log scale sur X car distribution exponentielle
        sns.histplot(hours_diff, log_scale=True, bins=30, color="orange", kde=True)
        plt.xlabel("Time since last push (Hours) - Log Scale")
        plt.title("Distribution of Time Intervals Between Commits")
        # Ajout de lignes repères
        plt.axvline(1, color='red', linestyle='--', alpha=0.5, label='1 Hour')
        plt.axvline(24, color='green', linestyle='--', alpha=0.5, label='1 Day')
        plt.axvline(168, color='blue', linestyle='--', alpha=0.5, label='1 Week')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "commit_time_distribution.png"))
        plt.close()

    # 4. Timings Report Enhanced
    print("\n" + "=" * 40)
    print("       BENCHMARK SUMMARY")
    print("=" * 40)
    total_time = sum(timings.values())

    for step, duration in timings.items():
        pct = (duration / total_time) * 100
        print(f"{step:<20}: {duration:8.4f} s ({pct:4.1f}%)")

    print("-" * 40)
    print(f"{'TOTAL TIME':<20}: {total_time:8.4f} s")
    print(f"{'MAX RAM':<20}: {max_ram:8.1f} MB")
    print(f"{'AVG CPU':<20}: {avg_cpu:8.1f} %")
    print("=" * 40)

    print(f"\nAll results saved in '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main()