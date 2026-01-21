import pandas as pd
import numpy as np
from tqdm import tqdm
from config import TRAIN_CONFIG
import compress_fasttext
import json
from scipy.stats import linregress


def enrich_and_weight_data(df_alerts, df_bugs):
    if 'id' in df_bugs.columns and 'bug_id' not in df_bugs.columns:
        df_bugs = df_bugs.rename(columns={'id': 'bug_id'})

    if df_alerts["push_timestamp"].dtype == 'object':
        df_alerts["push_timestamp"] = pd.to_datetime(df_alerts["push_timestamp"])

    df_alerts['is_weekend'] = (df_alerts['push_timestamp'].dt.dayofweek >= 5).astype(int)

    df_alerts['hour_sin'] = np.sin(2 * np.pi * df_alerts['push_timestamp'].dt.hour / 24)
    df_alerts['hour_cos'] = np.cos(2 * np.pi * df_alerts['push_timestamp'].dt.hour / 24)

    df_alerts = df_alerts.sort_values("push_timestamp")
    time_diff = df_alerts["push_timestamp"].diff().dt.total_seconds().fillna(3600)
    df_alerts['log_time_since_last_push'] = np.log1p(time_diff)

    df_alerts['seconds_since_last_push'] = time_diff

    if 'priority' in df_bugs.columns:
        prio_map = {
            'P1': 10.0,
            'P2': 5.0,
            'P3': 2.0,
            '--': 1.0,
            'CRITICAL': 10.0,
            'MAJOR': 5.0
        }

        df_merged = df_alerts.merge(
            df_bugs[['bug_id', 'priority']],
            on='bug_id',
            how='left'
        )

        df_merged['priority'] = df_merged['priority'].fillna('--')

        def get_weight(row):
            if row['bug_created'] == 0:
                return 1.0

            p_str = str(row['priority']).strip().upper()
            return prio_map.get(p_str, 1.0)

        df_alerts['sample_weight'] = df_merged.apply(get_weight, axis=1)

        print(f"Distribution des poids (Top 5):\n{df_alerts['sample_weight'].value_counts().head()}")

    else:
        print("WARNING priority not found")
        df_alerts['sample_weight'] = 1.0

    return df_alerts

def perform_time_split(df):
    time_col = "push_timestamp" if "push_timestamp" in df.columns else "alert_summary_creation_timestamp"
    df = df.sort_values(time_col).reset_index(drop=True)

    n = len(df)
    n_test = int(n * TRAIN_CONFIG["test_frac"])
    n_train_val = n - n_test

    train_val = df.iloc[:n_train_val].copy().reset_index(drop=True)
    test = df.iloc[n_train_val:].copy().reset_index(drop=True)

    return train_val, test

def get_embeddings(text_list):
    print(f"DL FastText")
    small_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load(
        'https://github.com/avidale/compress-fasttext/releases/download/v0.0.4/cc.en.300.compressed.bin'
    )

    embeddings = []
    for text in tqdm(text_list):
        if pd.isna(text) or str(text).strip() == "":
            embeddings.append(np.zeros(600))
        else:
            tokens = str(text).split()
            if not tokens:
                embeddings.append(np.zeros(600))
                continue

            word_vecs = [small_model[word] for word in tokens]
            sent_vec = np.concatenate([
                np.mean(word_vecs, axis=0),
                np.max(word_vecs, axis=0)
            ])
            embeddings.append(sent_vec)

    return np.vstack(embeddings).astype(np.float32)

def parse_related_count(val):
    if pd.isna(val) or str(val).strip() == "":
        return 0

    val_str = str(val).strip()
    val_str = val_str.strip('|')

    if not val_str:
        return 0

    parts = val_str.split('|')

    valid_ids = [p for p in parts if p.isdigit()]

    return len(valid_ids)


def parse_platform(val):
    if pd.isna(val):
        return "unknown", "unknown"

    val = str(val).lower()

    if "android" in val or "fenix" in val:
        os_name = "android"
    elif "linux" in val:
        os_name = "linux"
    elif "mac" in val or "osx" in val:
        os_name = "mac"
    elif "win" in val:
        os_name = "windows"
    else:
        os_name = val.split('-')[0]

    if "aarch64" in val or "arm64" in val:
        arch = "aarch64"
    elif "arm" in val:
        arch = "arm"
    elif "x86_64" in val or "x64" in val or "64-bit" in val:
        arch = "x64"
    elif "x86" in val or "32-bit" in val:
        arch = "x86"
    elif "-64" in val or " 64" in val:
        arch = "x64"
    elif "-32" in val or " 32" in val:
        arch = "x86"
    else:
        arch = "unknown"

    return os_name, arch


def parse_backfill(val):
    """
    Version ROBUSTE "BULLDOZER" :
    Cherche directement le JSON [...] dans la bouillie de caractères.
    """
    default_res = (-1.0, -1.0, 0.0, 0)  # Mean, Std, Slope, Count

    if pd.isna(val):
        return default_res

    s_val = str(val)

    # 1. Extraction chirurgicale : on cherche la liste JSON [...]
    start = s_val.find('[')
    end = s_val.rfind(']')

    if start == -1 or end == -1:
        return default_res

    json_candidate = s_val[start:end + 1]

    # 2. Nettoyage des échappements toxiques (le CSV double parfois les quotes)
    # On remplace les \" par " et les "" par "
    json_candidate = json_candidate.replace('\\"', '"').replace('""', '"')

    try:
        data = json.loads(json_candidate)
    except json.JSONDecodeError:
        # Dernière chance : parfois c'est des simple quotes en Python string
        try:
            json_candidate = json_candidate.replace("'", '"')
            data = json.loads(json_candidate)
        except:
            return default_res

    # 3. Validation de la structure
    if not isinstance(data, list) or len(data) < 2:
        return default_res

    # 4. Extraction des points (Time Series)
    ts_points = []
    for d in data:
        # On accepte int ou str pour la valeur, on convertit
        if isinstance(d, dict) and 'value' in d and 'push_timestamp' in d:
            try:
                v = float(d['value'])
                t = str(d['push_timestamp'])
                ts_points.append((t, v))
            except:
                continue

    if len(ts_points) < 2:
        return default_res

    # 5. Tri Temporel (Crucial pour la pente !)
    ts_points.sort(key=lambda x: x[0])
    values = [p[1] for p in ts_points]
    n = len(values)

    # 6. Stats
    mean_val = float(np.mean(values))
    std_val = float(np.std(values))

    # 7. Pente (Slope)
    try:
        if std_val < 1e-9:  # Si c'est plat, pente = 0
            slope = 0.0
        else:
            slope, _, _, _, _ = linregress(range(n), values)
    except:
        slope = 0.0

    return mean_val, std_val, slope, n

def engineer_complex_features(df):
    col_related = 'alert_summary_related_alerts'
    if col_related in df.columns:
        df['n_related_alerts'] = df[col_related].apply(parse_related_count)
        df.drop(col_related, axis=1, inplace=True)

    col_plat = 'single_alert_series_signature_machine_platform__mode'
    if col_plat in df.columns:
        parsed = df[col_plat].apply(parse_platform)
        df['platform_os'], df['platform_arch'] = zip(*parsed)
        df.drop(col_plat, axis=1, inplace=True)

    col_context = 'single_alert_backfill_record_context__mode'
    if col_context in df.columns:
        stats = df[col_context].apply(parse_backfill)
        df['rcd_ctxt_mean'], df['rcd_ctxt_std'], df['rcd_ctxt_slope'], df['rcd_ctxt_count'] = zip(*stats)
        val_col = 'single_alert_new_value__mean'
        if val_col not in df.columns:
            print("WARNING single_alert_new_value__mean not found")
            df['rcd_ctxt_zscore'] = 0.0
        else:
            epsilon = 1e-4
            raw_z = (df[val_col] - df['rcd_ctxt_mean']) / (df['rcd_ctxt_std'] + epsilon)
            df['rcd_ctxt_zscore'] = np.sign(raw_z) * np.log1p(np.abs(raw_z))
            df.loc[df['rcd_ctxt_count'] < 2, 'rcd_ctxt_zscore'] = 0.0
            df['rcd_ctxt_zscore'] = df['rcd_ctxt_zscore'].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        df.drop(col_context, axis=1, inplace=True)

    col_tags = 'alert_summary_performance_tags'
    if col_tags in df.columns:
        s_tags = df[col_tags].fillna("").astype(str).str.lower().str.replace(' ', '')
        tags_dummies = s_tags.str.get_dummies(sep=',')
        tags_dummies = tags_dummies.add_prefix('tag_')
        df = pd.concat([df, tags_dummies], axis=1)
        df.drop(col_tags, axis=1, inplace=True)

    col_options = "single_alert_series_signature_extra_options__mode"
    if col_options in df.columns:
        s_opts = df[col_options].fillna("").astype(str).str.lower().str.replace(' ', '')
        s_opts = s_opts.str.replace(r'taskcluster-projects/[^,]*', 'taskcluster', regex=True)
        opts_dummies = s_opts.str.get_dummies(sep=',')
        opts_dummies = opts_dummies.add_prefix('opt_')
        df = pd.concat([df, opts_dummies], axis=1)
        df.drop(col_options, axis=1, inplace=True)

    return df