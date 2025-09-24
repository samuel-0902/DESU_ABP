'''
Génération du dataset supervisé à partir des fichiers annotations + signaux ABP.
Chaque patient est découpé en fenêtres fixes, normalisées, et enrichies
(âge, état CEC/CPB) pour alimenter l’entraînement du VAE semi-supervisé.
'''

import numpy as np
import pandas as pd
from vae_abp.utils import log
from read_csv import (
    read_csv_same_output as _read_signals,
    get_patient_caracteristics_aligned,
    mark_periods,
)

LABEL_TO_ID = {"normal": 0, "pas_normal": 2, "mauvaise_qualite": 1}


def _fmt_paths(pid):
    '''
    Retourne le préfixe patient ("P3") et l’ID simple ("3") pour nommer les fichiers.
    '''
    return "P" + str(int(pid)), str(int(pid))


def _interp_segment_to_len(seg: pd.Series, target_len: int) -> np.ndarray:
    '''
    Ré-échantillonne un segment temporel à exactement `target_len` points
    par interpolation linéaire.
    '''
    t_ns = seg.index.view("int64")
    t = (t_ns - t_ns.min()) / 1e9
    v = seg.to_numpy(dtype=np.float32)
    t_u, idx = np.unique(t, return_index=True)
    v = v[idx]
    if len(t_u) < 2:
        return np.full((target_len,), v[0] if len(v) else np.nan, dtype=np.float32)
    grid = np.linspace(t_u[0], t_u[-1], target_len, dtype=np.float64)
    return np.interp(grid, t_u, v).astype(np.float32)


def make_np_dataset(
    patient_ids,
    signals_tpl,
    annots_tpl,
    listing_path=None,
    win_s=5,
    stride_s=2.5,
    min_points=5,
    coverage=0.6,
    target_len=575,
):
    '''
    Parcourt la liste de patients, charge signaux + annotations,
    découpe en fenêtres fixes, normalise et renvoie X, y, contexte, metadata.
    '''
    from vae_abp.data_io import normalize_abp_fs

    X_all, y_all, ctx_all, meta_all = [], [], [], []

    for pid in patient_ids:
        sig_path = signals_tpl.format(pid=pid)
        ann_path = annots_tpl.format(P="P" + str(pid))

        try:
            df_sig = _read_signals(sig_path, sep=";")
        except Exception as e:
            log(f"[WARN] Impossible de lire le signal pour PID={pid}: {e}")
            continue

        if not isinstance(df_sig.index, pd.DatetimeIndex):
            if "datetime" in df_sig.columns:
                df_sig["datetime"] = pd.to_datetime(df_sig["datetime"], errors="coerce")
                df_sig = df_sig.set_index("datetime").sort_index()
            elif "DateTime" in df_sig.columns:
                df_sig["DateTime"] = pd.to_datetime(df_sig["DateTime"], errors="coerce")
                df_sig = df_sig.set_index("DateTime").sort_index()
            else:
                df_sig.index = pd.to_datetime(df_sig.index, errors="coerce")

        try:
            annots = pd.read_parquet(ann_path)
        except Exception as e:
            log(f"[WARN] Impossible de lire les annotations pour PID={pid}: {e}")
            continue

        age_val = np.nan
        if listing_path is not None:
            try:
                df_caract = get_patient_caracteristics_aligned(df_sig, listing_path, int(pid))
                if "age" in df_caract.columns:
                    age_val = float(df_caract["age"].iloc[0])
                    df_sig["age"] = np.full(len(df_sig), age_val, dtype=float)
                else:
                    df_sig["age"] = np.nan
                df_sig = mark_periods(df_sig, df_caract, "H CEC_full", "H Fin CEC_full", "cec")
                df_sig = mark_periods(df_sig, df_caract, "H X clamp_full", "H Fin X Clamp_full", "clampage")
            except Exception as e:
                df_sig["cec"] = 0
                df_sig["clampage"] = 0
                df_sig["age"] = np.nan

        df_sig = normalize_abp_fs(df_sig, "abp[mmHg]", target_fs=115)

        try:
            segments, y_series, meta = build_fixed_windows_abp(
                annots, df_sig, win_s=win_s, stride_s=stride_s,
                min_points=min_points, coverage=coverage
            )
        except Exception as e:
            log(f"[WARN] PID={pid} → erreur pendant fenêtrage: {e}")
            continue

        if len(segments) == 0:
            continue

        segments_fixed = [_interp_segment_to_len(seg, target_len) for seg in segments]

        X_all.append(np.stack(segments_fixed))
        y_all.append(y_series.to_numpy())
        ctx_all.append(meta[["cec", "cpb"]].to_numpy())

        meta["patient_id"] = pid
        meta["sig_path"] = sig_path
        meta["ann_path"] = ann_path
        meta["age"] = age_val

        meta_all.append(meta)

    if not X_all:
        raise RuntimeError("Aucune fenêtre générée.")

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    ctx = np.concatenate(ctx_all, axis=0)
    meta = pd.concat(meta_all, axis=0).reset_index(drop=True)

    return X, y, ctx, meta


def build_fixed_windows_abp(
    data: pd.DataFrame,
    data_signaux: pd.DataFrame,
    win_s: int = 5,
    stride_s: int | None = None,
    min_points: int = 5,
    coverage: float = 0.6,
):
    '''
    Découpe une annotation en fenêtres fixes non chevauchantes (ou chevauchantes si stride_s < win_s)
    et retourne les segments temporels ABP associés ainsi que leurs labels/contextes.
    '''
    if stride_s is None:
        stride_s = win_s

    if "datetime" in data_signaux.columns:
        t_sig = _to_naive_datetime(data_signaux["datetime"])
    elif "DateTime" in data_signaux.columns:
        t_sig = _to_naive_datetime(data_signaux["DateTime"])
    elif isinstance(data_signaux.index, pd.DatetimeIndex):
        t_sig = pd.to_datetime(data_signaux.index, errors="coerce")
    else:
        raise ValueError("data_signaux doit contenir 'datetime' ou 'DateTime' ou un index DatetimeIndex.")

    abp = pd.to_numeric(data_signaux["abp[mmHg]"], errors="coerce")
    s_abp = pd.Series(abp.values, index=pd.to_datetime(t_sig)).dropna().sort_index()

    s_cec = None
    s_cpb = None
    if "cec" in data_signaux.columns:
        s_cec = pd.Series(pd.to_numeric(data_signaux["cec"], errors="coerce").fillna(0).astype(int).values,
                          index=s_abp.index)
        s_cec = s_cec.reindex(s_abp.index, method=None).fillna(0).astype(int)
    if "clampage" in data_signaux.columns:
        s_cpb = pd.Series(pd.to_numeric(data_signaux["clampage"], errors="coerce").fillna(0).astype(int).values,
                          index=s_abp.index)
        s_cpb = s_cpb.reindex(s_abp.index, method=None).fillna(0).astype(int)

    data = data.copy()
    data["start_time"] = _to_naive_datetime(data["start_time"])
    data["end_time"] = _to_naive_datetime(data["end_time"])

    seg_list, y_list, meta_rows = [], [], []

    for parent_idx, row in data.iterrows():
        t0, t1 = row["start_time"], row["end_time"]
        if pd.isna(t0) or pd.isna(t1):
            continue
        if t1 < t0:
            t0, t1 = t1, t0

        cur = t0
        while cur < t1:
            w_end = cur + pd.Timedelta(seconds=win_s)
            if w_end > t1:
                break

            seg = s_abp.loc[cur:w_end]
            n = len(seg)
            if n < min_points:
                cur += pd.Timedelta(seconds=stride_s)
                continue
            span = (seg.index.max() - seg.index.min()).total_seconds() if n > 1 else 0.0
            if span < win_s * coverage:
                cur += pd.Timedelta(seconds=stride_s)
                continue

            seg.name = f"win_parent{parent_idx}_{cur.strftime('%H%M%S')}"
            seg_list.append(seg)
            y_list.append(row.get("label_main", None))

            a_type = row.get("artefact_type", None)
            cec_flag = int(s_cec.loc[cur:w_end].max() > 0) if s_cec is not None else None
            cpb_flag = int(s_cpb.loc[cur:w_end].max() > 0) if s_cpb is not None else None

            meta_rows.append({
                "parent_row": parent_idx,
                "start_time": cur,
                "end_time": w_end,
                "duration_s": float((w_end - cur).total_seconds()),
                "n_samples": int(n),
                "label_main": row.get("label_main", None),
                "artefact_type": a_type,
                "cec": cec_flag,
                "cpb": cpb_flag,
                "patient_id": row.get("patient_id", None)
            })

            cur += pd.Timedelta(seconds=stride_s)

    segments = pd.Series(seg_list, name="abp_window")
    y = pd.Series(y_list, index=segments.index, name="label")
    meta = pd.DataFrame(meta_rows, index=segments.index)
    return segments, y, meta


def _to_naive_datetime(s):
    '''
    Convertit une série en datetime UTC-naïf.
    '''
    dt = pd.to_datetime(s, errors="coerce")
    try:
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_convert(None)
    except Exception:
        pass
    return dt


AGE_MAX_YEARS = 18.0


def build_supervised_dataset_with_context(labeled_file):
    '''
    Ajoute canaux âge et état CEC/CPB au signal ABP déjà fenêtré.
    Retourne X, Y, stats et metadata alignés.
    '''
    data = np.load(labeled_file, allow_pickle=True)
    X, y, ctx = data["X"], data["y"], data["ctx"]
    meta = pd.DataFrame(list(data["meta"])) if "meta" in data.files else None
    if meta is None:
        raise RuntimeError("meta absent dans data_label.npz")

    N, L = X.shape[0], X.shape[1]

    m = np.nanmean(X, axis=1, keepdims=True)
    sd = np.nanstd(X, axis=1, keepdims=True)
    sd = np.where(sd > 1e-6, sd, 1e-6)
    abp = ((X - m) / sd)[..., None].astype(np.float32)

    if "age" not in meta.columns:
        raise RuntimeError("La colonne 'age' est absente de meta")
    age_days = meta["age"].astype(float).to_numpy()
    age_years = age_days / 365.0
    age_norm = np.clip(age_years / AGE_MAX_YEARS, 0.0, 1.0)
    age = np.repeat(age_norm[:, None], L, axis=1)[..., None].astype(np.float32)

    cec = meta["cec"].astype(int).to_numpy()
    cpb = meta["cpb"].astype(int).to_numpy()
    state = np.where((cec == 0) & (cpb == 0), 0.0,
                     np.where((cec == 1) & (cpb == 0), 0.5, 1.0))
    state = np.repeat(state[:, None], L, axis=1)[..., None].astype(np.float32)

    X_label = np.concatenate([abp, age, state], axis=2)
    Y_label = (y > 0).astype("int64")

    return {
        "X": X_label,
        "Y": Y_label,
        "M": m.squeeze(),
        "SD": sd.squeeze(),
        "meta": meta
    }
