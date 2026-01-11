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
from tqdm import tqdm

# Imports de tes modules
from data_loader import load_raw_data, aggregate_alerts
from timeseries_multi import enrich_with_ts_features
from config import TRAIN_CONFIG
import compress_fasttext

# --- CONFIGURATION ---
OUTPUT_DIR = "benchmark_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Tes meilleurs paramètres (Hardcoded)
BEST_PARAMS = {
    "learning_rate": 0.05121825335140923,
    "depth": 6,
    "l2_leaf_reg": 3.5938074222624334,
    "random_strength": 0.0016127559270012492,
    "bagging_temperature": 0.5883202468150027,
    "border_count": 128,
    "min_data_in_leaf": 19,
    "scale_pos_weight": 1.2223024563767102,
    "grow_policy": "SymmetricTree",
    "pca_components": 161
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


# --- FONCTIONS UTILITAIRES ---
def get_embeddings(text_list):
    print(f"   -> Chargement FastText...")
    small_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(
        'https://github.com/avidale/compress-fasttext/releases/download/v0.0.4/cc.en.300.compressed.bin'
    )
    embeddings = []
    print(f"   -> Vectorisation de {len(text_list)} textes...")
    for text in tqdm(text_list, disable=True):
        if pd.isna(text) or str(text).strip() == "":
            embeddings.append(np.zeros(600))
        else:
            tokens = str(text).split()
            if not tokens:
                embeddings.append(np.zeros(600))
                continue
            word_vecs = [small_model[word] for word in tokens]
            sent_vec = np.concatenate([np.mean(word_vecs, axis=0), np.max(word_vecs, axis=0)])
            embeddings.append(sent_vec)
    return np.vstack(embeddings).astype(np.float32)


def perform_time_split(df):
    time_col = "push_timestamp" if "push_timestamp" in df.columns else "alert_summary_creation_timestamp"
    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)
    n_test = int(n * TRAIN_CONFIG["test_frac"])
    n_train_val = n - n_test
    return df.iloc[:n_train_val].copy().reset_index(drop=True), df.iloc[n_train_val:].copy().reset_index(drop=True)


def enrich_and_weight_data(df_alerts, df_bugs):
    if 'id' in df_bugs.columns and 'bug_id' not in df_bugs.columns:
        df_bugs = df_bugs.rename(columns={'id': 'bug_id'})

    if df_alerts["push_timestamp"].dtype == 'object':
        df_alerts["push_timestamp"] = pd.to_datetime(df_alerts["push_timestamp"])

    df_alerts['is_weekend'] = (df_alerts['push_timestamp'].dt.dayofweek >= 5).astype(int)
    df_alerts['hour_sin'] = np.sin(2 * np.pi * df_alerts['push_timestamp'].dt.hour / 24)
    df_alerts['hour_cos'] = np.cos(2 * np.pi * df_alerts['push_timestamp'].dt.hour / 24)

    df_alerts = df_alerts.sort_values("push_timestamp")
    # Feature brute pour le graphique final
    df_alerts['seconds_since_last_push'] = df_alerts["push_timestamp"].diff().dt.total_seconds().fillna(3600)
    df_alerts['log_time_since_last_push'] = np.log1p(df_alerts['seconds_since_last_push'])

    if 'priority' in df_bugs.columns:
        prio_map = {'P1': 10.0, 'P2': 5.0, 'P3': 2.0, '--': 1.0, 'CRITICAL': 10.0, 'MAJOR': 5.0}
        df_merged = df_alerts.merge(df_bugs[['bug_id', 'priority']], on='bug_id', how='left')
        df_merged['priority'] = df_merged['priority'].fillna('--')

        def get_weight(row):
            return 1.0 if row['bug_created'] == 0 else prio_map.get(str(row['priority']).strip().upper(), 1.0)

        df_alerts['sample_weight'] = df_merged.apply(get_weight, axis=1)
    else:
        df_alerts['sample_weight'] = 1.0

    return df_alerts


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
            "seconds_since_last_push"  # On ignore la feature brute, on garde le log
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


    finally:
        monitor.stop()
        monitor.join()

    # --- ANALYSE & PLOTS ---
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