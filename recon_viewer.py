# recon_viewer.py

"""
Viewer reconstruction + SQI

Rôle
----
- Charge un patient NPZ et (optionnellement) un modèle Keras (depuis chemin).
- Génére des fenêtres compatibles modèle + masque de validité (bridge 4/6 retours).
- Calcule prédiction de reconstruction et probabilité d’anomalie p_abn.
- Construit séries continues (reprojection overlap-add) + SQI combiné (classif/reco).
- Affiche un viewer interactif (Plotly + ipywidgets) avec contrôle des paramètres.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

FINAL_DATASET_PATH = "/workspace/venv/vae_final/final_dataset.npz"

# --- modules projet ---
import sqi_auc as SQI   
                         
import ae as A          


import numpy as _np

if not hasattr(SQI, "_rmse_norm_percentile_rolling_nan"):
    def _rmse_norm_percentile_rolling_nan(rmse, win_size=720, p_lo=1, p_hi=99, eps=1e-8):
        """
        Rôle : normalisation causale robuste (percentiles glissants) NaN-safe.
        Pour chaque i, percentiles sur rmse[max(0,i-win_size+1):i+1] (sur valeurs finies).
        """
        r = _np.asarray(rmse, float)
        N = len(r)
        out = _np.zeros(N, dtype=_np.float32)
        for i in range(N):
            start = max(0, i - int(win_size) + 1)
            block = r[start:i+1]
            block = block[_np.isfinite(block)]
            if block.size == 0:
                out[i] = _np.nan
                continue
            lo = _np.percentile(block, p_lo)
            hi = _np.percentile(block, p_hi)
            if not _np.isfinite(lo): lo = _np.nanmin(block)
            if not _np.isfinite(hi): hi = _np.nanmax(block)
            if hi <= lo: hi = lo + eps
            out[i] = float(_np.clip((r[i] - lo) / (hi - lo + eps), 0.0, 1.0)) if _np.isfinite(r[i]) else _np.nan
        return out.astype(_np.float32)
    SQI._rmse_norm_percentile_rolling_nan = _rmse_norm_percentile_rolling_nan  # shim


if not hasattr(SQI, "compose_sqi"):
    def _alpha_curve(p, alpha_min=0.2, alpha_max=0.8, k=5.0):
        """Rôle : α(p) pour le mélange (linéaire ↓ jusqu’à 0.5 puis croissance exp)."""
        p = _np.clip(_np.asarray(p, float), 0.0, 1.0)
        a = _np.empty_like(p)
        left = p <= 0.5
        a[left] = alpha_max - (alpha_max - alpha_min) * (p[left] / 0.5)
        right = ~left
        x = (p[right] - 0.5) / 0.5
        grow = 1.0 - _np.exp(-k * x)
        a[right] = alpha_min + (alpha_max - alpha_min) * grow
        return _np.clip(a, 0.0, 1.0)

    def compose_sqi(p_abn, rmse_norm, *, mode="alpha",
                    alpha_min=0.2, alpha_max=0.8, k=5.0,
                    const_w=0.5, cls_is_abnormal=True, invert_rmse=False):
        """Rôle : combine classif/reco en SQI selon mode 'alpha' ou 'const'."""
        p  = _np.clip(_np.asarray(p_abn, float), 0.0, 1.0)
        rn = _np.clip(_np.asarray(rmse_norm, float), 0.0, 1.0)
        score_cls = (1.0 - p) if cls_is_abnormal else p
        score_rec = (rn if invert_rmse else (1.0 - rn))
        if mode == "alpha":
            a = _alpha_curve(p, alpha_min, alpha_max, k)
            sqi = a * score_cls + (1.0 - a) * score_rec
        else:
            w = float(_np.clip(const_w, 0.0, 1.0))
            sqi = w * score_cls + (1.0 - w) * score_rec
        return _np.clip(sqi, 0.0, 1.0)
    SQI.compose_sqi = compose_sqi

def _safe_series(y, t):
    """Rôle : crée une série temporelle indexée (remplace ±Inf par NaN)."""
    idx = pd.to_datetime(t)
    s = pd.Series(np.asarray(y, dtype=np.float64), index=idx).sort_index()
    return s.replace([np.inf, -np.inf], np.nan)

def _dbg(name, s: pd.Series):
    """Rôle : stats rapides d’une série (min, max, NaN, étendue temporelle)."""
    vals = s.values
    n = len(vals)
    n_nan = int(np.isnan(vals).sum())
    n_fin = n - n_nan
    vmin = float(np.nanmin(vals)) if n_fin > 0 else np.nan
    vmax = float(np.nanmax(vals)) if n_fin > 0 else np.nan
    tmin = s.index.min() if n > 0 else None
    tmax = s.index.max() if n > 0 else None
    print(f"{name}: shape={s.shape}, time=[{tmin} .. {tmax}], "
          f"min={vmin:.6g}, max={vmax:.6g}, NaN={n_nan}/{n}")

def _thin_series(s: pd.Series, target_points: int = 8000):
    """Rôle : sous-échantillonne uniformément une série longue pour l’overview."""
    n = len(s)
    if n == 0 or n <= target_points:
        return s
    step = max(1, n // target_points)
    return s.iloc[::step]

def _load_train_mean_scaler(final_dataset_path: str) -> StandardScaler:
    """Rôle : StandardScaler sur la moyenne PAR FENÊTRE du canal 0 (X_supervised_train)."""
    d = np.load(final_dataset_path, allow_pickle=True)
    Xtr = d["X_supervised_train"]  # (N, L, C)
    m_win = np.nanmean(Xtr[:, :, 0], axis=1).astype(np.float32).reshape(-1, 1)
    scaler = StandardScaler().fit(m_win)
    return scaler

# ====================== Causal EMA (préserve les trous) ===============
def _ema_series(s: pd.Series, fs: float, tau_s: float | None):
    """Rôle : EMA causale sur valeurs finies (laisse NaN en sortie là où trous)."""
    if not tau_s or tau_s <= 0:
        return s
    x = s.values.astype(np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return s
    lam = np.exp(-(1.0 / fs) / float(tau_s))
    y = np.full_like(x, np.nan, dtype=np.float64)
    i0 = int(np.argmax(finite))
    prev = x[i0]
    y[i0] = prev
    for i in range(i0 + 1, len(x)):
        xi = x[i] if finite[i] else prev
        prev = (1.0 - lam) * xi + lam * prev
        y[i] = prev if finite[i] else np.nan
    return pd.Series(y, index=s.index, dtype=np.float64)

# ======================Overlap-average NaN-safe ===================
def _overlap_average_ignore_nan(values_win, starts, L_win, L_total):
    """Rôle : reprojette scores fenêtre → échantillons en ignorant les NaN."""
    out   = np.zeros((L_total,), dtype=np.float64)
    count = np.zeros((L_total,), dtype=np.float64)
    v = np.asarray(values_win, float)
    for s, val in zip(starts, v):
        if not np.isfinite(val):
            continue
        e = min(s + L_win, L_total)
        out[s:e]   += float(val)
        count[s:e] += 1.0
    res = np.full_like(out, np.nan, dtype=np.float64)
    np.divide(out, count, out=res, where=(count > 0))
    return res.astype(np.float32)

def _series_from_windows(scores_win, starts, L_win, L_total, t0, fs, ema_tau_s=0.0):
    """Rôle : reprojection fenêtre→continu puis lissage EMA causal (préserve NaN)."""
    per_sample = _overlap_average_ignore_nan(np.asarray(scores_win, np.float32),
                                             starts, L_win, L_total)
    ts_index = (pd.to_datetime(t0) + pd.to_timedelta(np.arange(L_total)/fs, unit="s"))
    s = pd.Series(per_sample, index=ts_index)
    if ema_tau_s and ema_tau_s > 0:
        s = _ema_series(s, fs, ema_tau_s)
    return s

# ===================== Masque temporel (depuis mask_win) ===============
def _mask_series_from_maskwin(mask_win, starts, L_total, fs, t0):
    """
    Rôle : proportion de fenêtres marquant chaque échantillon comme valide.
    Entrée : mask_win (N,L,1). Sortie : série [0..1] indexée au temps.
    """
    mw = np.asarray(mask_win, np.float32)
    assert mw.ndim == 3 and mw.shape[-1] == 1, f"mask_win shape attendu (N,L,1), obtenu {mw.shape}"
    Lm = int(mw.shape[1])
    mask2d = (mw[..., 0] > 0).astype(np.float32)  # (N, Lm)

    out   = np.zeros((L_total,), dtype=np.float32)
    count = np.zeros((L_total,), dtype=np.float32)
    for s, seg in zip(starts, mask2d):
        e = min(s + Lm, L_total)
        lw = e - s
        if lw > 0:
            out[s:e]   += seg[:lw]
            count[s:e] += 1.0

    prop = np.full((L_total,), np.nan, dtype=np.float32)
    np.divide(out, count, out=prop, where=(count > 0))
    ts_index = (pd.to_datetime(t0) + pd.to_timedelta(np.arange(L_total)/fs, unit="s"))
    return pd.Series(prop, index=ts_index)

# ====================== Reco continue (overlap-add) ======================
def _overlap_add(wins, starts, L_total, use_hann=True):
    """Rôle : recomposition 1D à partir de fenêtres (N,L,1) avec poids Hann optionnels."""
    wins = np.asarray(wins, np.float32)
    assert wins.ndim == 3 and wins.shape[-1] == 1
    L_win = wins.shape[1]
    out   = np.zeros((L_total,), dtype=np.float32)
    wsum  = np.zeros((L_total,), dtype=np.float32)
    w = np.hanning(L_win).astype(np.float32) if use_hann else np.ones((L_win,), np.float32)
    for s, seg in zip(starts, wins):
        e = min(s + L_win, L_total)
        lw = e - s
        out[s:e]  += (seg[:lw, 0] * w[:lw])
        wsum[s:e] += w[:lw]
    wsum[wsum == 0.0] = 1.0
    return out / wsum

# ====================== Bridge pour l'API windows_from_meta_any ==================
import inspect

def _windows_api_bridge(X_any, meta, fs, model):
    """
    Rôle : compatibilité ancienne/nouvelle signatures de SQI.windows_from_meta_any.
    - Ancienne : (X_any, meta, fs) -> 4 retours, on recalcule L_total/t0.
    - Nouvelle : accepte un scaler de moyenne de fenêtre depuis le TRAIN.
    - Adapte ensuite (wins, mask_win) au modèle.
    """
    fn = SQI.windows_from_meta_any
    params = list(inspect.signature(fn).parameters.keys())

    if "mean_scaler" in params or "mean_mwin_train" in params:
        scaler = _load_train_mean_scaler(FINAL_DATASET_PATH)
        kwargs = {}
        if "mean_scaler" in params:
            kwargs["mean_scaler"] = scaler
        elif "mean_mwin_train" in params and "std_mwin_train" in params:
            mu = float(scaler.mean_.ravel()[0])
            sigma = float(scaler.scale_.ravel()[0])
            kwargs["mean_mwin_train"] = mu
            kwargs["std_mwin_train"]  = max(sigma, 1e-6)
        if "mean_mode" in params:
            kwargs["mean_mode"] = "patient_const"
        r = fn(X_any, meta, fs, **kwargs)
    else:
        r = fn(X_any, meta, fs)

    if len(r) == 6:
        wins, mask_win, starts, L_win, L_total, t0 = r
    elif len(r) == 4:
        wins, mask_win, starts, L_win = r
        t0 = pd.to_datetime(meta[0]['ts'])
        te_last = pd.to_datetime(meta[-1]['te'])
        L_total = int(np.round((te_last - t0).total_seconds() * fs))
        L_total = max(L_total, int(starts.max() + L_win))
    else:
        raise ValueError(f"windows_from_meta_any retourne {len(r)} valeurs (attendu 4 ou 6).")

    wins, mask_win = SQI.adapt_for_model(wins, mask_win, model)
    return wins, mask_win, starts, L_win, L_total, t0

# ====================== Viewer ================
def _viewer_with_controls(s_true, s_pred, s_pabn, s_rn, * ,
                          fs=115, blend_mode="alpha",
                          alpha_min=0.2, alpha_max=0.8, k=5.0, const_w=0.5,
                          tau_s=2.5, window_s=20,
                          fig_width=1500, margin_l=40, margin_r=10,
                          overview_h=180, detail_h=440, overview_points=8000,
                          cls_is_abnormal=True, invert_rmse=True):
    """
    Rôle : construit les figures Plotly + contrôles (widgets) et gère les callbacks.
    - Overview : original / recon / SQI.
    - Détail : fenêtre glissante avec ajustements en temps réel du mélange.
    """
    sqi0 = SQI.compose_sqi(
        s_pabn.values, s_rn.values,
        mode=blend_mode,
        alpha_min=alpha_min, alpha_max=alpha_max, k=k,
        const_w=const_w,
        cls_is_abnormal=cls_is_abnormal,
        invert_rmse=invert_rmse
    )
    s_sqi_base = pd.Series(sqi0, index=s_true.index)
    s_sqi = _ema_series(s_sqi_base, fs, tau_s)

    t0, t1 = s_true.index.min(), s_true.index.max()
    total_secs = int(max(1, np.floor((t1 - t0).total_seconds())))
    win_secs = max(1, int(window_s))
    max_start = max(0, total_secs - win_secs)

    s_true_ds = _thin_series(s_true, overview_points)
    s_pred_ds = _thin_series(s_pred, overview_points)
    s_sqi_ds  = _thin_series(s_sqi,  overview_points)

    fig_over = go.FigureWidget()
    fig_over.add_scatter(x=list(s_true_ds.index.to_pydatetime()), y=s_true_ds.values.astype(float).tolist(),
                         mode="lines", name="Original", connectgaps=False)
    fig_over.add_scatter(x=list(s_pred_ds.index.to_pydatetime()), y=s_pred_ds.values.astype(float).tolist(),
                         mode="lines", name="Reconstruit", connectgaps=False)
    fig_over.add_scatter(x=list(s_sqi_ds.index.to_pydatetime()),  y=s_sqi_ds.values.astype(float).tolist(),
                         mode="lines", name="SQI", yaxis="y2", connectgaps=False)

    cx0 = t0 + pd.to_timedelta(win_secs/2, unit="s")
    fig_over.add_shape(type="line", x0=cx0, x1=cx0, y0=0, y1=1,
                       xref="x", yref="paper", line=dict(color="red", width=2))
    fig_over.update_layout(
        height=overview_h, width=fig_width,
        margin=dict(l=margin_l, r=margin_r, t=20, b=0),
        showlegend=True,
        xaxis=dict(type="date", range=[t0, t1]),
        yaxis=dict(title="Amplitude"),
        yaxis2=dict(title="SQI", overlaying="y", side="right", range=[0, 1])
    )

    def slice_window(sec, win_s):
        st = t0 + pd.to_timedelta(int(sec), unit="s")
        et = st + pd.to_timedelta(int(win_s), unit="s")
        a = s_true.loc[st:et]
        b = s_pred.loc[st:et]
        c = s_sqi.loc[st:et]
        return a, b, c, st, et

    seg_true0, seg_pred0, seg_sqi0, st0, _ = slice_window(0, win_secs)

    fig_det = go.FigureWidget()
    fig_det.add_scatter(x=list(seg_true0.index.to_pydatetime()), y=seg_true0.values.astype(float).tolist(),
                        mode="lines", name="Original", connectgaps=False)
    fig_det.add_scatter(x=list(seg_pred0.index.to_pydatetime()), y=seg_pred0.values.astype(float).tolist(),
                        mode="lines", name="Reconstruit", connectgaps=False)
    fig_det.add_scatter(x=list(seg_sqi0.index.to_pydatetime()),  y=seg_sqi0.values.astype(float).tolist(),
                        mode="lines", name="SQI", yaxis="y2", connectgaps=False)
    fig_det.update_layout(
        height=440, width=fig_width,
        margin=dict(l=margin_l, r=margin_r, t=26, b=36),
        showlegend=True,
        xaxis=dict(type="date", title="Temps"),
        yaxis=dict(title="Amplitude"),
        yaxis2=dict(title="SQI", overlaying="y", side="right", range=[0, 1]),
        title=f"Fenêtre détaillée ({win_secs}s @ {fs} Hz)"
    )

    inner_width = fig_width - margin_l - margin_r
    pos_slider = widgets.IntSlider(value=0, min=0, max=max_start, step=1,
                                   layout=widgets.Layout(width=f"{inner_width}px"))
    win_slider = widgets.IntSlider(value=win_secs, min=5, max=min(300, max(10, total_secs)), step=1,
                                   description="Fenêtre (s)", style={'description_width': 'initial'},
                                   layout=widgets.Layout(width=f"{inner_width}px"))

    cb_true = widgets.Checkbox(value=True, description="Original")
    cb_pred = widgets.Checkbox(value=True, description="Reconstruit")
    cb_sqi  = widgets.Checkbox(value=True, description="SQI")

    mode_dd  = widgets.Dropdown(options=[("Alpha(p_abn)", "alpha"), ("Mélange constant w", "const")],
                                value=blend_mode, description="Mélange", layout=widgets.Layout(width="230px"))
    a_min_sl = widgets.FloatSlider(value=float(alpha_min), min=0, max=1, step=0.01, description="alpha_min",
                                   layout=widgets.Layout(width="240px"))
    a_max_sl = widgets.FloatSlider(value=float(alpha_max), min=0, max=1, step=0.01, description="alpha_max",
                                   layout=widgets.Layout(width="240px"))
    k_sl     = widgets.FloatSlider(value=float(k),        min=0, max=20, step=0.1,  description="k",
                                   layout=widgets.Layout(width="200px"))
    w_sl     = widgets.FloatSlider(value=float(const_w),  min=0, max=1, step=0.01, description="w (cls)",
                                   layout=widgets.Layout(width="200px"))
    tau_sl   = widgets.FloatSlider(value=float(tau_s),    min=0, max=20, step=0.1,  description="EMA τ (s)",
                                   layout=widgets.Layout(width="240px"))

    def _toggle_controls():
        """Rôle : activer/désactiver sliders selon mode de mélange."""
        is_alpha = (mode_dd.value == "alpha")
        a_min_sl.disabled = not is_alpha
        a_max_sl.disabled = not is_alpha
        k_sl.disabled     = not is_alpha
        w_sl.disabled     = is_alpha
    _toggle_controls()

    def update_visibility(*_):
        """Rôle : toggle visibilité des courbes."""
        fig_over.data[0].visible = cb_true.value
        fig_over.data[1].visible = cb_pred.value
        fig_over.data[2].visible = cb_sqi.value
        fig_det.data[0].visible  = cb_true.value
        fig_det.data[1].visible  = cb_pred.value
        fig_det.data[2].visible  = cb_sqi.value

    def recompute_and_refresh(*_):
        """Rôle : recalculer SQI avec les paramètres et rafraîchir les figures."""
        nonlocal s_sqi, s_sqi_base
        a_min = min(a_min_sl.value, a_max_sl.value)
        a_max = max(a_min_sl.value, a_max_sl.value)
        sqi_raw = SQI.compose_sqi(
            s_pabn.values, s_rn.values,
            mode=mode_dd.value,
            alpha_min=a_min, alpha_max=a_max, k=k_sl.value,
            const_w=w_sl.value,
            cls_is_abnormal=True,
            invert_rmse=True
        )
        s_sqi_base = pd.Series(sqi_raw, index=s_true.index)
        s_sqi = _ema_series(s_sqi_base, fs, tau_sl.value)
        s_sqi_ds = _thin_series(s_sqi, overview_points)
        with fig_over.batch_update():
            fig_over.data[2].x = list(s_sqi_ds.index.to_pydatetime())
            fig_over.data[2].y = s_sqi_ds.values.astype(float).tolist()
        a, b, c, st, _ = slice_window(int(pos_slider.value), int(win_slider.value))
        with fig_det.batch_update():
            fig_det.data[2].x = list(c.index.to_pydatetime())
            fig_det.data[2].y = c.values.astype(float).tolist()

    def update_window(sec, win_s):
        """Rôle : met à jour la fenêtre détaillée et le curseur overview."""
        a, b, c, st, _ = slice_window(sec, win_s)
        cx = (st + pd.to_timedelta(win_s/2, unit="s")).to_pydatetime()
        with fig_det.batch_update():
            fig_det.data[0].x, fig_det.data[0].y = list(a.index.to_pydatetime()), a.values.astype(float).tolist()
            fig_det.data[1].x, fig_det.data[1].y = list(b.index.to_pydatetime()), b.values.astype(float).tolist()
            fig_det.data[2].x, fig_det.data[2].y = c.index.to_pydatetime(), c.values.astype(float).tolist()
        with fig_over.batch_update():
            fig_over.layout.shapes[0].x0 = cx
            fig_over.layout.shapes[0].x1 = cx

    # observers
    cb_true.observe(update_visibility, names="value")
    cb_pred.observe(update_visibility, names="value")
    cb_sqi.observe(update_visibility,  names="value")
    pos_slider.observe(lambda ch: update_window(int(ch["new"]), int(win_slider.value)), names="value")
    win_slider.observe(lambda ch: update_window(int(pos_slider.value), int(ch["new"])), names="value")
    mode_dd.observe(lambda ch: (_toggle_controls(), recompute_and_refresh()), names="value")
    a_min_sl.observe(recompute_and_refresh, names="value")
    a_max_sl.observe(recompute_and_refresh, names="value")
    k_sl.observe(recompute_and_refresh,     names="value")
    w_sl.observe(recompute_and_refresh,     names="value")
    tau_sl.observe(recompute_and_refresh,   names="value")

    toggles = widgets.HBox([cb_true, cb_pred, cb_sqi])
    mix_row = widgets.HBox([mode_dd, a_min_sl, a_max_sl, k_sl, w_sl, tau_sl])
    sliders_row = widgets.HBox([pos_slider, win_slider])

    display(fig_over, sliders_row, toggles, mix_row, fig_det)

    update_visibility()
    recompute_and_refresh()
    update_window(0, win_secs)

# ====================== Entrée simplifiée ======================
def run_viewer(npz_path: str, model_path: str | None = None, model=None,
               *, fs: float = 115.0, target_channel: int = 0, cls_is_abnormal: bool = True,
               blend_mode: str = "alpha", alpha_min: float = 0.2, alpha_max: float = 0.8, k: float = 5.0,
               const_w: float = 0.5, tau_s: float = 2.5, window_s: int = 20,
               overview_points: int = 8000, batch_size: int = 256,
               invert_rmse: bool = True, vis_denorm: bool = True, min_coverage: float = 0.10):
    """
    Rôle : pipeline de visualisation end-to-end pour un fichier patient NPZ.
    - Charge modèle (si non fourni) avec objets custom AE.
    - Construit fenêtres, prédictions, RMSE masquée, normalisation causale.
    - Reconstruit signaux continus et lance le viewer interactif.
    """
    # 1) Modèle
    if model is None:
        custom_objects = {"FiniteSanitizer": A.FiniteSanitizer, "CropToRef": A.CropToRef}
        model = keras.models.load_model(model_path, compile=False, custom_objects=custom_objects)

    # 2) Chargement patient
    data = np.load(npz_path, allow_pickle=True)
    X_any = data["X"].astype("float32")
    meta  = list(data["meta"])

    # 3) Fenêtres + timings + MASQUE (bridge 4/6-retours) + adapt
    wins, mask_win, starts, L_win, L_total, t0 = _windows_api_bridge(X_any, meta, fs, model)

    # 4) Prédictions
    recon_pred, p_abn = SQI._pick_outputs(model, wins, batch_size=batch_size, verbose=0)
    if recon_pred.shape[-1] > 1:
        recon_pred = recon_pred[..., target_channel:target_channel+1]
    y_true_recon = wins[..., target_channel:target_channel+1]

    # 5) Dé-normalisation optionnelle (unité physique) si M/SD présents
    if vis_denorm and ("M" in data.files) and ("SD" in data.files):
        M  = np.asarray(data["M"],  np.float32)
        SD = np.asarray(data["SD"], np.float32)
        if M.ndim == 1:  M = M[:, None, None]
        if SD.ndim == 1: SD = SD[:, None, None]
        if M.ndim == 2:  M = M[:, :, None]
        if SD.ndim == 2: SD = SD[:, :, None]
        SD = np.maximum(SD, 1e-6)
        y_true_phys = y_true_recon * SD + M
        recon_phys  = recon_pred   * SD + M
    else:
        y_true_phys = y_true_recon
        recon_phys  = recon_pred
        if vis_denorm:
            print("[WARN] M/SD introuvables — affichage en z-score.")

    # 6) RMSE MASQUÉE (fenêtre) + normalisation causale NaN-safe (percentiles glissants)
    rmse = SQI._rmse_per_window_masked(y_true_phys, recon_phys, mask_win, min_coverage=min_coverage)
    rmse_norm = SQI._rmse_norm_percentile_rolling_nan(rmse)

    # 7) Séries p_abn / rmse_norm
    s_pabn = _series_from_windows(p_abn,     starts, L_win, L_total, t0, fs, ema_tau_s=0.0)
    s_rn   = _series_from_windows(rmse_norm, starts, L_win, L_total, t0, fs, ema_tau_s=0.0)

    # 8) Reco continue + vrai signal (overlap-add Hann)
    sig_pred = _overlap_add(recon_phys,  starts, L_total, use_hann=True)
    sig_true = _overlap_add(y_true_phys, starts, L_total, use_hann=True)

    # 9) Séries temporelles indexées (sans comblement des trous)
    ts = pd.date_range(pd.to_datetime(t0), periods=L_total, freq=f"{1000/fs:.6f}ms")
    s_true = _safe_series(sig_true, ts)
    s_pred = _safe_series(sig_pred, ts)

    # 10) Masque/validité par échantillon (proportion de fenêtres valides)
    s_valid_prop = _mask_series_from_maskwin(mask_win, starts, L_total, fs, t0)
    s_valid = s_valid_prop >= 0.5

    # 11) Masquage visuel des courbes
    for ser in (s_true, s_pred, s_pabn, s_rn):
        ser[~s_valid.values] = np.nan

    # Debug succinct
    print("=== DEBUG (aligné sur Original) ===")
    _dbg("Original", s_true)
    _dbg("Reconstruit", s_pred)
    _dbg("p_abn", s_pabn)
    _dbg("rmse_norm", s_rn)

    # 12) Viewer
    _viewer_with_controls(s_true, s_pred, s_pabn, s_rn,
                          fs=fs, blend_mode=blend_mode,
                          alpha_min=alpha_min, alpha_max=alpha_max, k=k,
                          const_w=const_w, tau_s=tau_s,
                          window_s=window_s, overview_points=overview_points,
                          cls_is_abnormal=cls_is_abnormal,
                          invert_rmse=invert_rmse)
