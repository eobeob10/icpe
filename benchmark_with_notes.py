import time
import os
import pandas as pd
from contextlib import contextmanager
from catboost import CatBoostClassifier
from data_loader import load_raw_data, aggregate_alerts
from timeseries_multi import enrich_with_ts_features
from preprocessing import perform_time_split, enrich_and_weight_data, get_embeddings, engineer_complex_features
from model_utils import prepare_matrix_with_pca, make_pool, calculate_pr_at_k
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix
import seaborn as sns


OUTPUT_DIR = "benchmark_results_with_notes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_PARAMS = {
    "iterations": 2000,
    "eval_metric": "AUC",
    "early_stopping_rounds": 100,
    "verbose": 100,
    "task_type": "CPU",
    "thread_count": 4,
    "grow_policy": "Lossguide",
    "learning_rate": 0.04105602915025481,
    "depth": 7,
    "l2_leaf_reg": 4.009382263503614,
    "border_count": 32,
    "min_data_in_leaf": 34,
    "scale_pos_weight": 1.0003751949471695,
    "one_hot_max_size": 10,
    "max_leaves": 63,
    "bootstrap_type": "MVS",
    "subsample": 0.9718798313316739,
    "pca_components": 32
}

class BenchmarkPipeline:
    def __init__(self):
        self.data = {}
        self.model = None

    @contextmanager
    def log_step(self, name):
        print(f"\n[Step] {name}...");
        t0 = time.time();
        yield
        print(f"   Done in {time.time() - t0:.2f}s")

    def run(self):
        print("🚀 Benchmark [WITH NOTES] Started...")
        with self.log_step("1. Loading"): self._step_1_loading()
        with self.log_step("2. Time Series"): self._step_2_timeseries()
        with self.log_step("3. NLP (Notes)"): self._step_3_nlp()
        with self.log_step("4. Prep"): self._step_4_prep()
        with self.log_step("5. Train"): self._step_5_train()
        with self.log_step("6. Eval"): self._step_6_eval()
        self._step_8_reporting()

    def _step_1_loading(self):
        alerts, bugs = load_raw_data()
        df = aggregate_alerts(alerts)
        df = enrich_and_weight_data(df, bugs)
        df = engineer_complex_features(df)
        self.data['df'] = df
        self.data['alerts_raw'] = alerts

    def _step_2_timeseries(self):
        self.data['df'] = enrich_with_ts_features(self.data['df'], self.data['alerts_raw'])
        self.data['df']["bug_created"] = self.data['df']["bug_created"].fillna(0).astype(int)

    def _step_3_nlp(self):
        df = self.data['df']
        print("   -> Computing Embeddings on Notes...")
        notes = df['alert_summary_notes'].fillna('').astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
        self.data['embeddings'] = get_embeddings(notes.tolist())

    def _step_4_prep(self):
        train_val, test = perform_time_split(self.data['df'])
        n_train = len(train_val)

        emb_train = self.data['embeddings'][:n_train]
        emb_test = self.data['embeddings'][n_train:]

        pca_n = BEST_PARAMS["pca_components"]

        X_train, self.cat_cols, self.pca_model = prepare_matrix_with_pca(
            train_val, emb_train, n_components=pca_n, is_train=True
        )
        X_test, _, _ = prepare_matrix_with_pca(
            test, emb_test, pca_model=self.pca_model, n_components=pca_n, is_train=False
        )

        self.data['train_val'], self.data['test'] = train_val, test
        self.data['X_train'], self.data['X_test'] = X_train, X_test

    def _step_5_train(self):
        X, y = self.data['X_train'], self.data['train_val']['bug_created']
        w = self.data['train_val']['sample_weight']
        split = int(len(X) * 0.90)

        train_pool = make_pool(X.iloc[:split], y.iloc[:split], self.cat_cols, w.iloc[:split])
        val_pool = make_pool(X.iloc[split:], y.iloc[split:], self.cat_cols, w.iloc[split:])

        params = BEST_PARAMS.copy()
        if 'pca_components' in params: del params['pca_components']

        self.model = CatBoostClassifier(**params)
        self.model.fit(train_pool, eval_set=val_pool)

    def _step_6_eval(self):
        test_pool = make_pool(self.data['X_test'], None, self.cat_cols)
        self.probs = self.model.predict_proba(test_pool)[:, 1]
        y_test = self.data['test']['bug_created']
        print(f"   🏆 AUPRC: {average_precision_score(y_test, self.probs):.4f}")
        for k in [50, 100, 200]:
            p, r = calculate_pr_at_k(y_test, self.probs, k)
            print(f"P@{k:<4} | {p:.4f}  R@{k:<4} | {r:.4f}")

    def _step_8_reporting(self):
        print("\n--- Generating Scientific Graphs ---")
        y_test = self.data['test']['bug_created']

        fi = self.model.get_feature_importance(type="PredictionValuesChange")
        df_fi = pd.DataFrame({"feature": self.data['X_test'].columns, "importance": fi})

        pca_mask = df_fi["feature"].str.startswith("pca_")
        ts_mask = df_fi["feature"].str.startswith("ts_")

        pca_total = df_fi.loc[pca_mask, "importance"].sum()
        ts_total = df_fi.loc[ts_mask, "importance"].sum()

        df_top = df_fi[~(pca_mask | ts_mask)].copy()
        df_top = pd.concat([
            df_top,
            pd.DataFrame([{"feature": "Time Series (Signals)", "importance": ts_total}]),
            pd.DataFrame([{"feature": "NLP: Human Notes", "importance": pca_total}])
        ], ignore_index=True)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_top.sort_values("importance", ascending=False).head(15), x="importance", y="feature",
                    palette="viridis")
        plt.title("Feature Importance (Full Hybrid)")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/feature_importance_hybrid.png")
        plt.close()

        precision, recall, _ = precision_recall_curve(y_test, self.probs)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'Hybrid (AUPRC = {average_precision_score(y_test, self.probs):.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{OUTPUT_DIR}/pr_curve.png")
        plt.close()

        y_pred = (self.probs > 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', xticklabels=['No Bug', 'Bug'],
                    yticklabels=['No Bug', 'Bug'])
        plt.title('Normalized Confusion Matrix')
        plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png")
        plt.close()

        print(f"Graphs saved to {OUTPUT_DIR}")

if __name__ == "__main__": BenchmarkPipeline().run()