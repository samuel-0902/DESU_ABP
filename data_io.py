'''
Pipeline de pré-traitement complet pour signaux ABP :
- normalisation temporelle et fréquentielle
- détection / suppression des plateaux
- construction d’un tableau global patient×temps
- création des datasets supervisé / non supervisé pour entraînement
'''
import os
import numpy as np
import pandas as pd
from .config import (BASE_DIR, LISTING_PATH, SIGNAL_COL, TARGET_FS, USE_PLATEAU_FILTER,
                     PL_WIN_S, PL_STD_THR, PL_RANGE_THR, PL_MIN_DUR, PL_BRIDGE_S, AGE_MAX_YEARS)
from .utils import log
from sklearn.preprocessing import StandardScaler
from vae_abp.preprocessing_labeled_window import build_supervised_dataset_with_context


def clean_time_index(df_or_series):
    '''
    Uniformise l’index temporel : conversion en datetime UTC-naïf,
    suppression des doublons et tri chronologique.
    '''
    if isinstance(df_or_series, pd.Series):
        s = df_or_series.copy()
        s.index = pd.to_datetime(s.index, errors="coerce").tz_localize(None)
        s = s[~s.index.duplicated(keep="first")].sort_index()
        return s
    df = df_or_series.copy()
    df.index = pd.to_datetime(df.index, errors="coerce").tz_localize(None)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def _maybe_import_read_csv():
    '''
    Importe le module read_csv local s’il existe, sinon lève une erreur claire.
    '''
    try:
        import vae_abp.read_csv as r
        return r
    except Exception as e:
        raise ImportError(f"Impossible d'importer read_csv: {e}")


def read_csv_fallback(path, sep=';'):
    '''
    Lecture basique d’un CSV ABP : colonne 0 → index temps, dernière colonne → signal.
    '''
    df = pd.read_csv(path, sep=sep)
    df.index = pd.to_datetime(df.iloc[:,0], errors="coerce")
    if SIGNAL_COL not in df.columns:
        df[SIGNAL_COL] = pd.to_numeric(df.iloc[:, -1], errors='coerce')
    return df


def normalize_abp_fs(df, signal_col, target_fs=TARGET_FS, tol_hz=10.0,
                     keep_cols=("cec","clampage")):
    '''
    Ré-échantillonne le signal à target_fs Hz et recalcule les flags CEC/clampage
    sur la nouvelle grille temporelle.
    '''
    df = clean_time_index(df)
    s = pd.to_numeric(df[signal_col], errors="coerce").dropna()

    dt_s = np.median(np.diff(s.index.asi8)) / 1e9
    fs_est = 1.0 / dt_s if np.isfinite(dt_s) and dt_s > 0 else target_fs

    period = pd.to_timedelta(1/target_fs, unit="s")
    start, stop = s.index[0].ceil(period), s.index[-1].floor(period)
    grid = pd.date_range(start, stop, freq=period)

    out = pd.DataFrame(index=grid)
    out[signal_col] = s.reindex(grid, method="nearest").astype("float32")

    if "cec" in df.columns:
        t0, t1 = df.index[df["cec"]==1].min(), df.index[df["cec"]==1].max()
        out["cec"] = ((out.index >= t0) & (out.index <= t1)).astype("int8")
    else:
        out["cec"] = 0

    if "clampage" in df.columns:
        t0, t1 = df.index[df["clampage"]==1].min(), df.index[df["clampage"]==1].max()
        out["clampage"] = ((out.index >= t0) & (out.index <= t1)).astype("int8")
    else:
        out["clampage"] = 0

    print(f"[normalize_abp_fs SAFE] Resampled to {target_fs}Hz → "
          f"CEC={out['cec'].sum()}, Clamp={out['clampage'].sum()}")

    return out


def rm_plateaus_robust_safe(df, col: str, action="nan",
                            win_s=PL_WIN_S, std_thr=PL_STD_THR, range_thr=PL_RANGE_THR,
                            min_dur_s=PL_MIN_DUR, bridge_s=PL_BRIDGE_S):
    '''
    Détecte et supprime ou masque les plateaux plats (std & range faibles) dans le signal.
    '''
    if col not in df.columns or df.empty: return df
    s = pd.to_numeric(df[col], errors="coerce").astype("float32")
    if s.isna().all(): return df
    dt = np.median(np.diff(df.index.asi8)) / 1e9 if len(df) > 1 else (1.0/TARGET_FS)
    win_len = int(max(1, round(win_s / max(dt, 1e-6))))
    std_ = s.rolling(win_len, min_periods=max(3, win_len//5)).std()
    rng_ = s.rolling(win_len, min_periods=max(3, win_len//5)).apply(lambda x: np.nanmax(x)-np.nanmin(x), raw=False)
    flat = (std_.abs() <= std_thr) & (rng_.abs() <= range_thr)
    min_len = int(round(min_dur_s / max(dt, 1e-6)))
    bridge  = int(round(bridge_s / max(dt, 1e-6)))
    flat_idx = flat.to_numpy()
    if bridge > 0 and flat_idx.any():
        gaps = (~flat_idx).astype(np.int8)
        i = 0
        while i < len(gaps):
            if gaps[i] == 1:
                j = i
                while j < len(gaps) and gaps[j] == 1: j += 1
                gap_len = j - i
                left_flat  = (i-1 >= 0 and flat_idx[i-1])
                right_flat = (j < len(gaps) and flat_idx[j])
                if gap_len <= bridge and left_flat and right_flat:
                    flat_idx[i:j] = True
                i = j
            else:
                i += 1
    if min_len > 1 and flat_idx.any():
        i = 0; n = len(flat_idx)
        while i < n:
            if flat_idx[i]:
                j = i
                while j < n and flat_idx[j]: j += 1
                if (j - i) < min_len: flat_idx[i:j] = False
                i = j
            else:
                i += 1
    out = df.copy()
    if action == "nan":
        s2 = s.copy(); s2[flat_idx] = np.nan; out[col] = s2
    elif action == "drop":
        out = out.loc[~flat_idx]
    else:
        out[col] = s
    return out


def build_chunk_table(patient_ids, signal_col, listing_path, verbose=1):
    '''
    Agrège tous les patients en un seul DataFrame temporel
    avec colonnes : pid, time, abp, age, state.
    '''
    r = _maybe_import_read_csv()
    rows = []

    for pid in patient_ids:
        if verbose:
            log(f"[table] Lecture PID={pid}…")

        csv_path = os.path.join(BASE_DIR, f"csv_patient_{pid}.csv")
        if not os.path.exists(csv_path):
            if verbose:
                log(f"[table] WARN: fichier manquant → {csv_path}")
            continue

        try:
            if r is None:
                df = read_csv_fallback(csv_path, sep=";")
                _mark = lambda df, *args, **kw: mark_periods(df, *args, **kw)
                _caract = get_patient_caracteristics_aligned
            else:
                df = r.read_csv_same_output(csv_path, sep=";")
                _mark = r.mark_periods
                _caract = r.get_patient_caracteristics_aligned
        except Exception as e:
            log(f"[table] ERROR lecture CSV pid={pid} → {e}")
            continue

        df = clean_time_index(df)

        try:
            caract = _caract(df, listing_path, int(pid))
            df = _mark(df, caract, "H CEC_full", "H Fin CEC_full", "cec")
            df = _mark(df, caract, "H X clamp_full", "H Fin X Clamp_full", "clampage")
        except Exception as e:
            if verbose:
                log(f"[table] pid={pid} WARN mark_periods: {e}")
            caract = None

        try:
            df = normalize_abp_fs(
                df,
                signal_col=SIGNAL_COL,
                target_fs=TARGET_FS,
                tol_hz=10.0,
                keep_cols=("cec", "clampage"),
            )
        except Exception as e:
            log(f"[table] ERROR normalize_abp_fs pid={pid} → {e}")
            continue

        if (
            caract is not None
            and "age_cec" in caract.columns
            and "birth_date" in caract.columns
        ):
            try:
                age_years = (
                    pd.to_datetime(caract.iloc[0]["age_cec"])
                    - pd.to_datetime(caract.iloc[0]["birth_date"])
                ).days / 365.0
            except Exception:
                age_years = 0.0
                print('Age Missing')
        elif (
            caract is not None
            and "age" in caract.columns
            and pd.notna(caract.iloc[0].get("age"))
        ):
            val = float(caract.iloc[0]["age"])
            age_years = (val / 365.0)
        age_norm = np.float32(np.clip(age_years / AGE_MAX_YEARS, 0.0, 1.0))

        if "cec" in df.columns or "clampage" in df.columns:
            cec = (
                pd.to_numeric(df.get("cec", 0), errors="coerce")
                .fillna(0)
                .astype("int8")
                > 0
            )
            clp = (
                pd.to_numeric(df.get("clampage", 0), errors="coerce")
                .fillna(0)
                .astype("int8")
                > 0
            )

            state = np.zeros(len(df), dtype=np.int8)
            state[clp.values & ~cec.values] = 0
            state[cec.values & ~clp.values] = 1
            state[cec.values & clp.values]   = 2

            if verbose:
                log(
                    f"[table] PID={pid} → "
                    f"clamp_only={(clp & ~cec).sum()}, "
                    f"cec_only={(cec & ~clp).sum()}, "
                    f"both={(cec & clp).sum()}, "
                    f"state uniques={np.unique(state)}"
                )
        else:
            state = np.zeros(len(df), dtype=np.int8)

        sub = pd.DataFrame(
            {
                "pid":   np.full(len(df), int(pid), dtype=np.int32),
                "time":  df.index.values,
                "abp":   pd.to_numeric(df[SIGNAL_COL], errors="coerce").astype("float32").values,
                "age":   np.full(len(df), age_norm, dtype=np.float32),
                "state": state,
            }
        )
        rows.append(sub)
        del df, sub

    if not rows:
        return pd.DataFrame(columns=["pid", "time", "abp", "age", "state"])

    tbl = pd.concat(rows, ignore_index=True)
    for c, typ in [
        ("pid", "int32"),
        ("abp", "float32"),
        ("age", "float32"),
        ("state", "int8"),
    ]:
        tbl[c] = tbl[c].astype(typ)
    tbl["time"] = pd.to_datetime(tbl["time"])

    if verbose:
        mb = tbl.memory_usage(index=True, deep=True).sum() / 1e6
        log(f"[table] Chunk final: {len(tbl):,} lignes, ~{mb:.1f} MB")

    return tbl


def build_final_datasets(supervised_file,
                         unsupervised_dir,
                         train_ids_label,
                         train_ids_unlabel,
                         test_ids_label,
                         test_ids_unlabel):
    '''
    Crée les datasets train/test supervisés et non supervisés :
    - charge le fichier .npz supervisé
    - charge les fichiers .npz patients non supervisés
    - ajoute un canal « moyenne »
    - harmonise la longueur des fenêtres à 575 pts
    '''
    ds = build_supervised_dataset_with_context(supervised_file)
    X, Y, M, SD, meta = ds["X"], ds["Y"], ds["M"], ds["SD"], ds["meta"]

    mask_train = meta["patient_id"].isin(train_ids_label).to_numpy()
    mask_test  = meta["patient_id"].isin(test_ids_label).to_numpy()

    X_train, Y_train = X[mask_train], Y[mask_train]
    X_test,  Y_test  = X[mask_test],  Y[mask_test]

    def load_unsup(ids):
        X_list = []
        for pid in ids:
            path = os.path.join(unsupervised_dir, f"patients_{pid}_clean.npz")
            if not os.path.exists(path):
                print(f"[WARN] fichier manquant : {path}")
                continue
            data = np.load(path, allow_pickle=True)
            X_list.append(data["X"])
        if not X_list:
            return np.empty((0, 0, 0))
        return np.concatenate(X_list, axis=0)

    X_unsup_train = load_unsup(train_ids_unlabel)
    X_unsup_test  = load_unsup(test_ids_unlabel)

    def add_mean_channel(X):
        if X.size == 0:
            return X
        m = np.nanmean(X[:, :, 0], axis=1, keepdims=True)
        m = np.nan_to_num(m, nan=0.0)
        m = StandardScaler().fit_transform(m)
        m = np.repeat(m[:, None, :], X.shape[1], axis=1)
        return np.concatenate([X, m], axis=2)

    X_train       = add_mean_channel(X_train)
    X_test        = add_mean_channel(X_test)
    X_unsup_train = add_mean_channel(X_unsup_train)
    X_unsup_test  = add_mean_channel(X_unsup_test)

    target_len = 575
    def harmonize_length(X, target_len):
        if X.size == 0:
            return X
        if X.shape[1] > target_len:
            return X[:, :target_len, :]
        return X

    X_train       = harmonize_length(X_train,       target_len)
    X_test        = harmonize_length(X_test,        target_len)
    X_unsup_train = harmonize_length(X_unsup_train, target_len)
    X_unsup_test  = harmonize_length(X_unsup_test,  target_len)

    return {
        "X_supervised_train":   X_train,
        "Y_supervised_train":   Y_train,
        "X_supervised_test":    X_test,
        "Y_supervised_test":    Y_test,
        "X_unsupervised_train": X_unsup_train,
        "X_unsupervised_test":  X_unsup_test,
    }
