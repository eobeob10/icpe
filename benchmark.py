import time
import os
import pandas as pd
import threading
import psutil
from contextlib import contextmanager
from catboost import CatBoostClassifier
from data_loader import load_raw_data, aggregate_alerts
from timeseries_multi import enrich_with_ts_features
from preprocessing import perform_time_split, enrich_and_weight_data, engineer_complex_features
from model_utils import prepare_matrix_with_pca, make_pool, calculate_pr_at_k
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix
import seaborn as sns
from pathlib import Path

OUTPUT_DIR = "benchmark_results_no_nlp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BEST_PARAMS = {
    "iterations": 2000,
    "eval_metric": "AUC",
    "early_stopping_rounds": 100,
    "verbose": 100,
    "task_type": "CPU",
    "thread_count": 4,
    "grow_policy": "Depthwise",
    "learning_rate": 0.04885409006852588,
    "depth": 6,
    "l2_leaf_reg": 0.6897929548921232,
    "border_count": 128,
    "min_data_in_leaf": 62,
    "scale_pos_weight": 1.1894491861597762,
    "one_hot_max_size": 50,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8197289074825262
}


class ResourceMonitor(threading.Thread):
    def __init__(self, interval=0.1):
        super().__init__()
        self.interval = interval
        self.stop_event = threading.Event()
        self.cpu_readings = []
        self.ram_readings = []
        self.process = psutil.Process(os.getpid())

    def run(self):
        self.process.cpu_percent()
        while not self.stop_event.is_set():
            try:
                children = self.process.children(recursive=True)
                all_procs = [self.process] + children

                total_ram = 0
                total_cpu = 0.0

                for p in all_procs:
                    try:
                        total_ram += p.memory_info().rss / (1024 * 1024)
                        total_cpu += p.cpu_percent(interval=None)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                self.cpu_readings.append(total_cpu)
                self.ram_readings.append(total_ram)

            except Exception:
                pass

            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()
        self.join()

    def get_stats(self):
        if not self.ram_readings:
            return 0.0, 0.0
        peak_ram = max(self.ram_readings)
        avg_cpu = sum(self.cpu_readings) / len(self.cpu_readings)
        return peak_ram, avg_cpu


class BenchmarkPipeline:
    def __init__(self):
        self.data = {}
        self.model = None
        self.metrics_history = []

    @contextmanager
    def log_step(self, name):
        print(f"\n[Step] {name}...")
        monitor = ResourceMonitor(interval=0.2)
        monitor.start()
        t0 = time.time()

        yield

        duration = time.time() - t0
        monitor.stop()
        peak_ram, avg_cpu = monitor.get_stats()

        print(f"   Done in {duration:.4f}s | Peak RAM: {peak_ram:.1f} MB | Avg CPU: {avg_cpu:.1f}%")
        self.metrics_history.append({
            "Stage": name,
            "Time (s)": round(duration, 4),
            "Peak RAM (MB)": round(peak_ram, 1),
            "Average CPU (%)": round(avg_cpu, 1)
        })

    def run(self):
        ts_cache = Path("./derived_features/ts_features_multiscale_v2.parquet")
        if ts_cache.exists():
            print(f"WARNING THERE IS A CACHE FILE, RESOURCES METRICS COULD BE CORRUPTED")

        print("Benchmark no nlp Started...")

        with self.log_step("Load data"):
            self._step_1_loading()

        with self.log_step("Time-series features"):
            self._step_2_timeseries()

        self.metrics_history.append(
            {"Stage": "NLP embeddings", "Time (s)": 0, "Peak RAM (MB)": 0, "Average CPU (%)": 0})

        with self.log_step("PCA preparation"):
            self._step_4_prep()

        with self.log_step("Training"):
            self._step_5_train()

        with self.log_step("Inference"):
            self._step_6_eval()

        self._step_8_reporting()
        self._print_final_table()

    def _print_final_table(self):
        print("\n" + "=" * 60)
        print("Table 3: Runtime breakdown and resource usage (Full Process Tree)")
        print("=" * 60)
        df_stats = pd.DataFrame(self.metrics_history)

        total_row = {
            "Stage": "Total Pipeline",
            "Time (s)": df_stats["Time (s)"].sum(),
            "Peak RAM (MB)": df_stats["Peak RAM (MB)"].max(),
            "Average CPU (%)": df_stats["Average CPU (%)"].mean()
        }
        df_stats = pd.concat([df_stats, pd.DataFrame([total_row])], ignore_index=True)

        print(df_stats.to_string(index=False))
        print("=" * 60)
        df_stats.to_csv(f"{OUTPUT_DIR}/runtime_metrics_full.csv", index=False)

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

    def _step_4_prep(self):
        train_val, test = perform_time_split(self.data['df'])
        X_train, self.cat_cols, _ = prepare_matrix_with_pca(train_val, embeddings=None, is_train=True)
        X_test, _, _ = prepare_matrix_with_pca(test, embeddings=None, is_train=False)
        self.data['train_val'], self.data['test'] = train_val, test
        self.data['X_train'], self.data['X_test'] = X_train, X_test

    def _step_5_train(self):
        X, y = self.data['X_train'], self.data['train_val']['bug_created']
        w = self.data['train_val']['sample_weight']
        split = int(len(X) * 0.90)
        train_pool = make_pool(X.iloc[:split], y.iloc[:split], self.cat_cols, w.iloc[:split])
        val_pool = make_pool(X.iloc[split:], y.iloc[split:], self.cat_cols, w.iloc[split:])
        self.model = CatBoostClassifier(**BEST_PARAMS)
        self.model.fit(train_pool, eval_set=val_pool)

    def _step_6_eval(self):
        test_pool = make_pool(self.data['X_test'], None, self.cat_cols)
        self.probs = self.model.predict_proba(test_pool)[:, 1]
        y_test = self.data['test']['bug_created']
        print(f"   AUPRC: {average_precision_score(y_test, self.probs):.4f}")
        for k in [50, 100, 200]:
            p, r = calculate_pr_at_k(y_test, self.probs, k)
            print(f"P@{k:<4} | {p:.4f}  R@{k:<4} | {r:.4f}")

    def _step_8_reporting(self):
        print("\n--- Generating Graphs ---")
        y_test = self.data['test']['bug_created']

        fi = self.model.get_feature_importance(type="PredictionValuesChange")
        df_fi = pd.DataFrame({"feature": self.data['X_test'].columns, "importance": fi})
        ts_mask = df_fi["feature"].str.startswith("ts_")
        ts_total = df_fi.loc[ts_mask, "importance"].sum()
        df_top = df_fi[~ts_mask].copy()
        df_top = pd.concat([df_top, pd.DataFrame([{"feature": "Time Series Aggregated", "importance": ts_total}])],
                           ignore_index=True)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_top.sort_values("importance", ascending=False).head(15), x="importance", y="feature",
                    palette="viridis")
        plt.title("Feature Importance (TS + Static)")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/feature_importance_grouped.png")
        plt.close()

        precision, recall, _ = precision_recall_curve(y_test, self.probs)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'TS Enabled (AUPRC = {average_precision_score(y_test, self.probs):.2f})')
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

        if 'rcd_ctxt_zscore' in self.data['test'].columns:
            plt.figure(figsize=(8, 6))
            df_plot = self.data['test'].copy()
            df_plot['prob'] = self.probs
            subset = df_plot[(df_plot['rcd_ctxt_zscore'] > -5) & (df_plot['rcd_ctxt_zscore'] < 5)]
            sns.regplot(x='rcd_ctxt_zscore', y='prob', data=subset, scatter_kws={'alpha': 0.1},
                        line_kws={'color': 'red'})
            plt.title('Sensitivity to Z-Score')
            plt.savefig(f"{OUTPUT_DIR}/zscore_sensitivity.png")
            plt.close()

        print(f"Graphs saved to {OUTPUT_DIR}")


if __name__ == "__main__": BenchmarkPipeline().run()