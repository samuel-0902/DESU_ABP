"""
Découpe un signal ABP continu en fenêtres fixes de longueur TARGET_LEN
et construit un tenseur 3-D prêt pour le VAE :
    (N fenêtres, L points, C canaux)

Canaux produits :
    0 : ABP z-scorée
    1 : âge normalisé 0-1 (si ADD_AGE_CHANNEL=True)
    2 : état CEC/clampage 0/0.5/1 (si ADD_STATE_CHANNEL=True)

Fonctions clés :
- _series_to_windows_interped_mmHg : découpe + interpolation linéaire
- windows_from_table_std : pipeline complet depuis la table chunkée
"""

import numpy as np
import pandas as pd
from .config import WIN_S, STRIDE_S, TARGET_LEN, MIN_POINTS, ADD_AGE_CHANNEL, ADD_STATE_CHANNEL
from .data_io import clean_time_index


def _series_to_windows_interped_mmHg(
    s: pd.Series,
    win_s: float = WIN_S,
    stride_s: float = STRIDE_S,
    target_len: int = TARGET_LEN,
    min_points: int = MIN_POINTS,
) -> tuple[np.ndarray, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """
    Découpe une Series temporelle ABP en fenêtres fixes et interpole chaque fenêtre
    à exactement `target_len` points.

    Args:
        s : Series pandas (index DatetimeIndex) – signal ABP brut en mmHg
        win_s : durée d'une fenêtre (s)
        stride_s : pas entre fenêtres (s)
        target_len : nombre de points de sortie souhaité
        min_points : nombre minimal de points valides dans la fenêtre

    Returns:
        X : ndarray(float32) de shape (N, target_len) – fenêtres interpolées
        times : list[tuple(ts, te)] – bornes temporelles de chaque fenêtre
    """
    s = clean_time_index(s)
    s = pd.to_numeric(s, errors="coerce")
    s = s[np.isfinite(s)]
    if len(s) < min_points:
        return np.empty((0, target_len), np.float32), []

    t_ns = s.index.asi8
    t0_ns = int(t_ns.min())
    t = (t_ns - t0_ns) / 1e9  # secondes depuis t0
    v = s.to_numpy(np.float32)

    # bornes de découpe
    starts = np.arange(float(t.min()), float(t.max()) - win_s + 1e-9, stride_s, dtype=np.float64)
    X_list, times = [], []
    base_ts = pd.to_datetime(t0_ns)

    for ts in starts:
        te = ts + win_s
        m = (t >= ts) & (t <= te)
        if m.sum() < min_points:
            continue

        tw, vw = t[m], v[m]
        finite = np.isfinite(vw)
        if finite.sum() < 2:
            continue
        tw, vw = tw[finite], vw[finite]

        tw_u, idx = np.unique(tw, return_index=True)
        vw = vw[idx]
        if len(tw_u) < 2:
            continue

        # interpolation linéaire régulière
        grid = np.linspace(tw_u[0], tw_u[-1], target_len, dtype=np.float64)
        x = np.interp(grid, tw_u, vw).astype(np.float32)
        if not np.all(np.isfinite(x)):
            continue

        X_list.append(x)
        times.append((base_ts + pd.to_timedelta(ts, unit="s"),
                      base_ts + pd.to_timedelta(te, unit="s")))

    if not X_list:
        return np.empty((0, target_len), np.float32), []

    return np.stack(X_list, axis=0), times


def windows_from_table_std(
    tbl: pd.DataFrame,
    add_age_channel: bool = ADD_AGE_CHANNEL,
    add_state_channel: bool = ADD_STATE_CHANNEL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """
    Pipeline complet : table chunkée multi-patients → tenseur 3-D prêt pour le VAE.

    Args:
        tbl : DataFrame issu de build_chunk_table (cols: pid, time, abp, age, state)
        add_age_channel : ajoute un canal âge constant par fenêtre
        add_state_channel : ajoute un canal état CEC/clampage (0/0.5/1)

    Returns:
        X : ndarray(float32) – shape (N, TARGET_LEN, C)
        M : ndarray(float32) – moyenne ABP par fenêtre (pour dénormalisation)
        SD : ndarray(float32) – écart-type ABP par fenêtre
        meta : list[dict] – métadonnées par fenêtre (pid, ts, te)
    """
    Xstd_all, M_all, SD_all, meta_all = [], [], [], []

    for pid, g in tbl.groupby("pid", sort=False):
        s_abp = pd.Series(g["abp"].values, index=g["time"].values)
        s_state = pd.Series(g["state"].values, index=g["time"].values).astype("int8")
        age_val = float(g["age"].iloc[0])

        X_mmHg, times = _series_to_windows_interped_mmHg(s_abp)
        if X_mmHg.shape[0] == 0:
            continue

        # normalisation z-score par fenêtre
        m = np.nanmean(X_mmHg, axis=1).astype(np.float32)
        sd = np.nanstd(X_mmHg, axis=1).astype(np.float32)
        sd = np.where(sd > 1e-6, sd, 1e-6).astype(np.float32)

        Xz = ((X_mmHg - m[:, None]) / sd[:, None]).astype(np.float32)
        Xz = Xz[..., None]  # (N, L, 1)
        chans = [Xz]

        # canal âge (constant pour toute la fenêtre)
        if add_age_channel:
            chans.append(
                np.full((Xz.shape[0], Xz.shape[1], 1),
                        np.float32(np.clip(age_val, 0, 1)),
                        dtype=np.float32)
            )

        # canal état CEC/clampage (0 = rien, 0.5 = CEC seul, 1 = CEC+clampage)
        if add_state_channel:
            STATE_list = []
            for (ts, te) in times:
                grid = pd.to_datetime(np.linspace(ts.value, te.value, TARGET_LEN, dtype=np.int64))
                lab = s_state.reindex(grid, method="nearest").fillna(0).to_numpy(dtype=np.int8)
                st = np.where(lab == 2, 1.0,
                     np.where(lab == 1, 0.5, 0.0)).astype(np.float32)
                STATE_list.append(st)
            STATE = np.stack(STATE_list, axis=0)[..., None]
            chans.append(STATE)

        # concaténation finale
        Xc = np.concatenate(chans, axis=2)
        Xstd_all.append(Xc)
        M_all.append(m)
        SD_all.append(sd)
        meta_all.extend([{"pid": int(pid), "ts": ts, "te": te} for (ts, te) in times])

    if not Xstd_all:
        C_in = 1 + int(add_age_channel) + int(add_state_channel)
        return (
            np.empty((0, TARGET_LEN, C_in), np.float32),
            np.empty((0,), np.float32),
            np.empty((0,), np.float32),
            [],
        )

    X = np.concatenate(Xstd_all, axis=0).astype(np.float32)
    M = np.concatenate(M_all, axis=0).astype(np.float32)
    SD = np.concatenate(SD_all, axis=0).astype(np.float32)

    return X, M, SD, meta_all
