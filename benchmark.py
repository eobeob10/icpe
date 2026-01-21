import time
import os
import gc
import threading
import psutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix
from cleanlab.filter import find_label_issues
from data_loader import load_raw_data, aggregate_alerts
from timeseries_multi import enrich_with_ts_features
from preprocessing import perform_time_split, enrich_and_weight_data, get_embeddings, engineer_complex_features
from model_utils import prepare_matrix_with_pca, make_pool, calculate_pr_at_k

# Config
OUTPUT_DIR = "benchmark_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_PARAMS = {
    "grow_policy": "Depthwise",
    "learning_rate": 0.009608567895102135,
    "depth": 7,
    "l2_leaf_reg": 2.32345478451899,
    "border_count": 32,
    "min_data_in_leaf": 54,
    "scale_pos_weight": 1.1518865052450915,
    "bootstrap_type": "MVS",
    "subsample": 0.9581959175648539,
    "one_hot_max_size": 50,
    "pca_components": 20,
    "iterations": 2000,
    "eval_metric": "AUC",
    "early_stopping_rounds": 100,
    "verbose": 100,
    "task_type": "CPU",
    "thread_count": 4
}

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
            curr = time.time() - self.start_time
            self.records.append({
                "time": curr,
                "cpu": psutil.cpu_percent(interval=None),
                "ram": self.process.memory_info().rss / (1024 * 1024)
            })
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()


class BenchmarkPipeline:
    def __init__(self):
        self.timings = {}
        self.data = {}
        self.model = None
        self.pca_model = None
        self.cat_cols = []
        self.probs = None
        self.monitor = ResourceMonitor()

    @contextmanager
    def log_step(self, name):
        print(f"\n[Step] {name}...")
        t0 = time.time()
        yield
        duration = time.time() - t0
        self.timings[name] = duration
        print(f"   Done in {duration:.2f}s")

    def run(self):
        print("🚀 Démarrage du Benchmark 'REALISTE' (No Human Notes)...")
        self.monitor.start()

        try:
            with self.log_step("1. Loading & Feature Engineering"):
                self._step_1_loading()

            with self.log_step("2. Time Series Extraction"):
                self._step_2_timeseries()

            with self.log_step("3. NLP Embeddings (Technical Context)"):
                self._step_3_nlp()

            with self.log_step("4. Prep & Splitting"):
                self._step_4_prep_split()

            with self.log_step("5. Model Training"):
                self._step_5_training()

            with self.log_step("6. Inference & Evaluation"):
                self._step_6_evaluation()

            self._step_7_cleanlab_analysis()
            self._step_8_scientific_reporting()
            self._step_9_export_errors()

        finally:
            self.monitor.stop()
            self.monitor.join()
            print("\nBenchmark Finished. Cleaning up memory...")
            self.data.clear()
            gc.collect()

    def _step_1_loading(self):
        alerts, bugs = load_raw_data()
        df = aggregate_alerts(alerts)
        df = enrich_and_weight_data(df, bugs)
        df = engineer_complex_features(df)

        self.data['df'] = df
        self.data['alerts_raw'] = alerts
        self.data['n_commits'] = df['push_timestamp'].nunique()

    def _step_2_timeseries(self):
        self.data['df'] = enrich_with_ts_features(self.data['df'], self.data['alerts_raw'])
        self.data['df']["bug_created"] = self.data['df']["bug_created"].fillna(0).astype(int)

    def _step_3_nlp(self):
        df = self.data['df']

        tech_context = (
                df['alert_summary_repository'].fillna('').astype(str) + " " +
                df['alert_summary_framework'].fillna('').astype(str) + " " +
                df['single_alert_series_signature_suite__mode'].fillna('').astype(str) + " " +
                df['single_alert_series_signature_test__mode'].fillna('').astype(str)
        )

        tech_context = tech_context.str.replace(r'\s+', ' ', regex=True).str.strip()

        print(f"   -> Embedding source: Technical Context (Ex: '{tech_context.iloc[0]}')")
        self.data['embeddings'] = get_embeddings(tech_context.tolist())

    def _step_4_prep_split(self):
        train_val, test = perform_time_split(self.data['df'])
        n_train = len(train_val)

        emb_train = self.data['embeddings'][:n_train]
        emb_test = self.data['embeddings'][n_train:]

        del self.data['embeddings']
        gc.collect()

        pca_n = BEST_PARAMS["pca_components"]

        X_train_full, self.cat_cols, self.pca_model = prepare_matrix_with_pca(
            train_val, emb_train, n_components=pca_n, is_train=True
        )

        X_test_full, _, _ = prepare_matrix_with_pca(
            test, emb_test, pca_model=self.pca_model, n_components=pca_n, is_train=False
        )

        self.data['train_val'] = train_val
        self.data['test'] = test
        self.data['X_train_full'] = X_train_full
        self.data['X_test_full'] = X_test_full

    def _step_5_training(self):
        X = self.data['X_train_full']
        y = self.data['train_val']['bug_created']
        w = self.data['train_val']['sample_weight']

        split_idx = int(len(X) * 0.90)

        train_pool = make_pool(X.iloc[:split_idx], y.iloc[:split_idx], self.cat_cols, w.iloc[:split_idx])
        val_pool = make_pool(X.iloc[split_idx:], y.iloc[split_idx:], self.cat_cols, w.iloc[split_idx:])

        params = BEST_PARAMS.copy()
        del params['pca_components']

        self.model = CatBoostClassifier(**params)
        self.model.fit(train_pool, eval_set=val_pool)

    def _step_6_evaluation(self):
        test_pool = make_pool(self.data['X_test_full'], None, self.cat_cols)
        self.probs = self.model.predict_proba(test_pool)[:, 1]

        y_test = self.data['test']['bug_created']
        auprc = average_precision_score(y_test, self.probs)
        print(f"   🏆 FINAL TEST AUPRC (Realistic): {auprc:.4f}")

        print("-" * 40)
        print(f"{'Metric':<10} | {'Value':<10}")
        print("-" * 40)
        for k in [50, 100, 200]:
            p, r = calculate_pr_at_k(y_test, self.probs, k)
            print(f"P@{k:<4}     | {p:.4f}")
            print(f"R@{k:<4}     | {r:.4f}")
        print("-" * 40)

    def _step_7_cleanlab_analysis(self):
        print("\n--- Cleanlab Analysis (Suspects Inspection) ---")
        try:
            y_test = self.data['test']['bug_created'].values
            pred_probs = np.column_stack((1 - self.probs, self.probs))

            issues_idx = find_label_issues(labels=y_test, pred_probs=pred_probs,
                                           return_indices_ranked_by='self_confidence')

            print(f"   Labels suspects détectés : {len(issues_idx)}")

            if len(issues_idx) > 0:
                print("\n   🔍 TOP 3 SUSPECTS (Z-Score vs Label):")
                for i, idx in enumerate(issues_idx[:3]):
                    row = self.data['test'].iloc[idx]
                    prob = self.probs[idx]
                    label = int(y_test[idx])
                    z_score = row.get('rcd_ctxt_zscore', 0.0)

                    print(f"   #{i + 1} [ID {row.get('alert_summary_id')}] Label: {label} vs Pred: {prob:.4f}")
                    print(
                        f"      Z-Score: {z_score:.2f} | Suite: {row.get('single_alert_series_signature_suite__mode', 'N/A')}")

        except Exception as e:
            print(f"   Cleanlab analysis skipped: {e}")

    def _step_8_scientific_reporting(self):
        print("\n--- Generating Scientific Graphs ---")
        self._plot_feature_importance_grouped()
        self._plot_pipeline_latency_waterfall()
        self._plot_pr_curve()
        self._plot_confusion_matrix()
        self._plot_prediction_distribution()
        print(f"   Graphs saved in {OUTPUT_DIR}")

    def _step_9_export_errors(self):
        print("\n--- [AUDIT] Exporting Misclassified Instances ---")

        # 1. Récupération des données de Test
        df_test = self.data['test'].copy()
        y_true = df_test['bug_created']
        y_prob = self.probs
        y_pred = (y_prob > 0.5).astype(int)  # Seuil standard

        # 2. Ajout des prédictions au DataFrame
        df_test['model_probability'] = y_prob
        df_test['model_prediction'] = y_pred

        # 3. Filtrage : On ne garde que les erreurs
        # Error condition: Label != Prediction
        mask_error = (y_true != y_pred)
        df_errors = df_test[mask_error].copy()

        # 4. Catégorisation de l'erreur (FP vs FN)
        # FP = Label 0, Pred 1
        # FN = Label 1, Pred 0
        df_errors['error_type'] = df_errors.apply(
            lambda row: "Faux Positif (Alarme inutile)" if row['model_prediction'] == 1 else "Faux Negatif (Bug raté)",
            axis=1
        )

        # 5. Calcul de la "Gravité" de l'erreur (Confidence Gap)
        # Si c'est un FP avec proba 0.99, c'est une erreur grave.
        # Si c'est un FN avec proba 0.01, c'est une erreur grave.
        df_errors['error_severity'] = np.abs(df_errors['model_probability'] - df_errors['bug_created'])

        # 6. Sélection des colonnes utiles pour l'analyse (Contextualisation)
        # On recrée le "Technical Context" pour que je puisse le lire
        df_errors['tech_context_str'] = (
                df_errors['alert_summary_repository'].fillna('').astype(str) + " | " +
                df_errors['single_alert_series_signature_suite__mode'].fillna('').astype(str) + " | " +
                df_errors['single_alert_series_signature_test__mode'].fillna('').astype(str)
        )

        cols_to_export = [
            'alert_summary_id',
            'error_type',
            'model_probability',
            'error_severity',
            'tech_context_str',  # Ce que le modèle a "lu"
            'rcd_ctxt_zscore',  # La métrique principale
            'n_related_alerts',  # L'heuristique principale
            'tag_infra',  # Le tag principal
            'single_alert_prev_value__p90'  # Contexte de valeur
        ]

        # On s'assure que les colonnes existent
        cols_final = [c for c in cols_to_export if c in df_errors.columns]

        # 7. Tri par gravité (les pires erreurs en premier) et Export
        df_errors_sorted = df_errors[cols_final].sort_values('error_severity', ascending=False)

        filename = os.path.join(OUTPUT_DIR, "benchmark_errors.csv")
        df_errors_sorted.to_csv(filename, index=False)

        print(f"   🚨 {len(df_errors)} erreurs trouvées sur {len(df_test)} exemples.")
        print(f"   📂 Détails sauvegardés dans : {filename}")
        print("   -> Envoie-moi ce fichier pour l'analyse des justifications.")

    def _plot_feature_importance_grouped(self):
        fi = self.model.get_feature_importance(type="PredictionValuesChange")
        feature_names = self.data['X_test_full'].columns
        df_fi = pd.DataFrame({"feature": feature_names, "importance": fi})
        pca_mask = df_fi["feature"].str.startswith("pca_")
        pca_total_imp = df_fi.loc[pca_mask, "importance"].sum()
        ts_mask = df_fi["feature"].str.startswith("ts_")
        ts_total_imp = df_fi.loc[ts_mask, "importance"].sum()
        df_grouped = df_fi[~(pca_mask | ts_mask)].copy()
        new_rows = [
            {"feature": "NLP: Technical Context", "importance": pca_total_imp, "type": "NLP"},
            {"feature": "Time Series (Signals)", "importance": ts_total_imp, "type": "TS"}
        ]
        df_grouped["type"] = "Other"
        df_grouped = pd.concat([df_grouped, pd.DataFrame(new_rows)], ignore_index=True)
        df_grouped = df_grouped.sort_values("importance", ascending=False).head(15)
        plt.figure(figsize=(10, 6))
        def get_color(row):
            if row['feature'] == "NLP: Technical Context": return '#c44e52'  # Rouge
            if row['feature'] == "Time Series (Signals)": return '#55a868'  # Vert
            return '#4c72b0'  # Bleu

        clrs = df_grouped.apply(get_color, axis=1).tolist()

        sns.barplot(data=df_grouped, x="importance", y="feature", palette=clrs)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4c72b0', label='Metadata / Static'),
            Patch(facecolor='#c44e52', label='NLP (Context)'),
            Patch(facecolor='#55a868', label='Time Series (History)')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        plt.title(" Feature Importance (Realistic Mode) - Grouped")
        plt.xlabel("Importance (%)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "scientific_feature_importance.png"), dpi=300)
        plt.close()

    def _plot_pipeline_latency_waterfall(self):
        """Barre empilée horizontale (Stacked Bar) du temps de traitement."""
        n_commits = self.data['n_commits']
        if n_commits == 0: n_commits = 1

        label_map = {
            "1. Loading & Feature Engineering": "Load & Eng.",
            "2. Time Series Extraction": "Time Series",
            "3. NLP Embeddings (Technical Context)": "NLP (Inference)",
            "4. Prep & Splitting": "Preprocessing",
            "6. Inference & Evaluation": "Model Inference"
        }

        data = []
        total_inference_time = 0

        for full_name, short_name in label_map.items():
            if full_name in self.timings:
                t = self.timings[full_name]
                data.append({"Step": short_name, "Time": t})
                total_inference_time += t

        df_time = pd.DataFrame(data)
        df_time['Percentage'] = (df_time['Time'] / total_inference_time) * 100

        fig, ax = plt.subplots(figsize=(12, 4))
        left = 0
        colors = sns.color_palette("husl", len(df_time))

        for i, row in df_time.iterrows():
            ax.barh(0, row['Percentage'], left=left, color=colors[i], edgecolor='white', height=0.5, label=row['Step'])
            if row['Percentage'] > 5:
                ax.text(left + row['Percentage'] / 2, 0, f"{row['Percentage']:.1f}%",
                        ha='center', va='center', color='white', fontweight='bold', fontsize=10)
            left += row['Percentage']

        ax.set_yticks([])
        ax.set_xlabel("Percentage of Total Pipeline Time")
        ms_per_commit = (total_inference_time / n_commits) * 1000
        ax.set_title(f" Latency Breakdown per Commit (~{ms_per_commit:.1f}ms)")
        ax.set_xlim(0, 100)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(df_time), frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "scientific_latency_breakdown.png"), dpi=300)
        plt.close()

    def _plot_pr_curve(self):
        y_test = self.data['test']['bug_created']
        precision, recall, _ = precision_recall_curve(y_test, self.probs)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='#2b5797', lw=2,
                 label=f'CatBoost (AUPRC = {average_precision_score(y_test, self.probs):.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(' Precision-Recall Curve (Realistic)')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "scientific_pr_curve.png"), dpi=300)
        plt.close()

    def _plot_confusion_matrix(self):
        y_test = self.data['test']['bug_created']
        preds = (self.probs > 0.5).astype(int)
        cm = confusion_matrix(y_test, preds, normalize='true')

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues",
                    xticklabels=["No Bug", "Bug"], yticklabels=["No Bug", "Bug"])
        plt.title(" Normalized Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "scientific_confusion_matrix.png"))
        plt.close()

    def _plot_prediction_distribution(self):
        plt.figure(figsize=(8, 5))
        df_res = pd.DataFrame({
            "Probability": self.probs,
            "True Label": self.data['test']['bug_created'].replace({0: "No Bug", 1: "Bug"})
        })
        sns.histplot(data=df_res, x="Probability", hue="True Label", bins=20, multiple="layer", alpha=0.6)
        plt.title(" Model Confidence Distribution")
        plt.xlabel("Predicted Probability of Bug")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "scientific_prediction_dist.png"))
        plt.close()


if __name__ == "__main__":
    pipeline = BenchmarkPipeline()
    pipeline.run()