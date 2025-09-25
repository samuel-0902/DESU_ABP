# plot_result.py


# ================================================================
# Section 1 — Overlay de reconstructions sur une fenêtre donnée
# ================================================================
import os, re, glob, json, math, inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

import ae as A
import sqi_auc as SQI   
                      

FINAL_DATASET_PATH = "/workspace/venv/vae_final/final_dataset.npz"


def _load_train_mean_scaler(final_dataset_path: str) -> StandardScaler:
    """StandardScaler sur la moyenne PAR FENÊTRE du canal 0 du TRAIN."""
    d = np.load(final_dataset_path, allow_pickle=True)
    Xtr = d["X_supervised_train"]  # (N, L, C)
    m_win = np.nanmean(Xtr[:, :, 0], axis=1).astype(np.float32).reshape(-1, 1)
    return StandardScaler().fit(m_win)


def _windows_api_bridge_for_overlay(X_any, meta, fs):
    """
    Supporte:
      - ancienne signature: (X_any, meta, fs)
      - nouvelle: (X_any, meta, fs, mean_scaler, mean_mode=...) OU
                  (X_any, meta, fs, mean_mwin_train, std_mwin_train, mean_mode=...)
    Recalcule L_total/t0 si non fournis.
    """
    fn = SQI.windows_from_meta_any
    params = list(inspect.signature(fn).parameters.keys())

    if "mean_scaler" in params or "mean_mwin_train" in params:
        scaler = _load_train_mean_scaler(FINAL_DATASET_PATH)
        kwargs = {}
        if "mean_scaler" in params:
            kwargs["mean_scaler"] = scaler
        elif "mean_mwin_train" in params and "std_mwin_train" in params:
            kwargs["mean_mwin_train"] = float(scaler.mean_.ravel()[0])
            kwargs["std_mwin_train"]  = float(max(scaler.scale_.ravel()[0], 1e-6))
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

    return wins, mask_win, starts, L_win, L_total, t0


def _safe_series(y, t):
    """Série temporelle (remplace ±Inf par NaN)."""
    idx = pd.to_datetime(t)
    s = pd.Series(np.asarray(y, dtype=np.float64), index=idx).sort_index()
    return s.replace([np.inf, -np.inf], np.nan)


def _mask_series_from_maskwin(mask_win, starts, L_total, fs, t0):
    """Proportion de fenêtres valides par échantillon (0..1)."""
    mw = np.asarray(mask_win, np.float32)
    Lm = int(mw.shape[1])
    mask2d = (mw[..., 0] > 0).astype(np.float32)  # (N,Lm)
    out   = np.zeros((L_total,), dtype=np.float32)
    count = np.zeros((L_total,), dtype=np.float32)
    for s, seg in zip(starts, mask2d):
        e = min(s + Lm, L_total)
        lw = e - s
        if lw > 0:
            out[s:e]   += seg[:lw]
            count[s:e] += 1.0
    count[count == 0.0] = 1.0
    prop = out / count
    ts_index = (pd.to_datetime(t0) + pd.to_timedelta(np.arange(L_total)/fs, unit="s"))
    return pd.Series(prop, index=ts_index)


def _overlap_add(wins, starts, L_total, use_hann=True):
    """Recompose un signal 1D à partir de (N,L,1) et starts (échantillons)."""
    wins = np.asarray(wins, np.float32)
    assert wins.ndim == 3 and wins.shape[-1] == 1
    L_win = wins.shape[1]
    out   = np.zeros((L_total,), np.float32)
    wsum  = np.zeros((L_total,), np.float32)
    w = np.hanning(L_win).astype(np.float32) if use_hann else np.ones((L_win,), np.float32)
    for s, seg in zip(starts, wins):
        e = min(s + L_win, L_total)
        lw = e - s
        if lw > 0:
            out[s:e]  += (seg[:lw, 0] * w[:lw])
            wsum[s:e] += w[:lw]
    wsum[wsum == 0.0] = 1.0
    return out / wsum


def overlay_recon_window(
    npz_path,
    model_paths,                 
    *, fs=115.0,
    t_start_sec=10357, win_s=5,
    target_channel=0,
    vis_denorm=True,
    valid_strict=0.999            #
):
    """Superpose plusieurs reconstructions modèle sur une fenêtre temporelle donnée."""
    data = np.load(npz_path, allow_pickle=True)
    X_any = data["X"].astype("float32")
    meta  = list(data["meta"])

    wins0, mask_win0, starts, L_win, L_total, t0 = _windows_api_bridge_for_overlay(X_any, meta, fs)

    y_true_recon0 = wins0[..., target_channel:target_channel+1]
    if vis_denorm and ("M" in data.files) and ("SD" in data.files):
        M  = np.asarray(data["M"],  np.float32)
        SD = np.asarray(data["SD"], np.float32)
        if M.ndim == 1:  M = M[:, None, None]
        if SD.ndim == 1: SD = SD[:, None, None]
        if M.ndim == 2:  M = M[:, :, None]
        if SD.ndim == 2: SD = SD[:, :, None]
        SD = np.maximum(SD, 1e-6)
        y_true_phys0 = y_true_recon0 * SD + M
    else:
        y_true_phys0 = y_true_recon0

    sig_true = _overlap_add(y_true_phys0, starts, L_total, use_hann=True)
    ts = pd.date_range(pd.to_datetime(t0), periods=L_total, freq=f"{1000/fs:.6f}ms")
    s_true = _safe_series(sig_true, ts)

    s_valid_global = pd.Series(False, index=s_true.index)
    fig = go.Figure()

    st = pd.to_datetime(t0) + pd.to_timedelta(int(t_start_sec), unit="s")
    et = st + pd.to_timedelta(int(win_s), unit="s")
    seg_true = s_true.loc[st:et]
    fig.add_scatter(
        x=list(seg_true.index.to_pydatetime()),
        y=seg_true.values.astype(float).tolist(),
        mode="lines", name="Original", line=dict(width=2),
        connectgaps=False
    )

    if isinstance(model_paths, (list, tuple)):
        model_paths = {os.path.basename(p): p for p in model_paths}

    for label, mpath in model_paths.items():
        custom_objects = {"FiniteSanitizer": A.FiniteSanitizer, "CropToRef": A.CropToRef}
        model = keras.models.load_model(mpath, compile=False, custom_objects=custom_objects)

        wins, mask_win = SQI.adapt_for_model(wins0.copy(), mask_win0.copy(), model)

        recon_pred, _ = SQI._pick_outputs(model, wins, batch_size=256, verbose=0)
        if recon_pred.shape[-1] > 1:
            recon_pred = recon_pred[..., target_channel:target_channel+1]

        if vis_denorm and ("M" in data.files) and ("SD" in data.files):
            recon_phys = recon_pred * SD + M
        else:
            recon_phys = recon_pred

        sig_pred = _overlap_add(recon_phys, starts, L_total, use_hann=True)
        s_pred = _safe_series(sig_pred, ts)

        s_valid_prop = _mask_series_from_maskwin(mask_win, starts, L_total, fs, t0)
        s_valid = s_valid_prop >= float(valid_strict)
        s_valid_global |= s_valid

        s_pred_masked = s_pred.copy()
        s_pred_masked[~s_valid.values] = np.nan
        seg_pred = s_pred_masked.loc[st:et]

        fig.add_scatter(
            x=list(seg_pred.index.to_pydatetime()),
            y=seg_pred.values.astype(float).tolist(),
            mode="lines", name=f"Recon {label}",
            connectgaps=False
        )

    fig.update_layout(
        title=f"Overlay reconstructions | fenêtre {t_start_sec}s → {t_start_sec+win_s}s",
        xaxis_title="Temps", yaxis_title="Amplitude",
        width=1200, height=420, legend=dict(orientation="h")
    )
    fig.show()


# =====================================================================
# Section 2 — AUC vs latent (subset fixe) & meilleure AUC par latent
# =====================================================================
# ---------- SETTINGS ----------
ROOTS_AUC_LATENT = [
    "/workspace/venv/vae_final/experiments/run_p100_p50_global_20250925-142111",
    "/workspace/venv/vae_final/experiments/run_p100_p50_global_20250925-094442",
]
SCALE = 0.7
FIG_W, FIG_H, DPI = 12 * SCALE, 6 * SCALE, 120

ALPHA_MIN_FIXED, ALPHA_MAX_FIXED, K_FIXED = 0.0, 0.5, 5.0
LATENTS_TO_PLOT_FIXED = None
VERBOSE_FIXED = True

LAT_DIR_RE = re.compile(r"lat(?P<latent>\d+)(?:\D|$)", re.IGNORECASE)
AK_DIR_RE  = re.compile(r"^amin(?P<amin>[0-9p]+)_amax(?P<amax>[0-9p]+)_k(?P<k>[0-9p]+)$")


def _p2dot(s: str) -> float:
    return float(s.replace("p", "."))


def _safe_read_json(path: str):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        if VERBOSE_FIXED: print(f"[WARN] JSON illisible: {path} ({e})")
        return None


def _safe_ratio(num, den):
    num = float(num) if num is not None else np.nan
    den = float(den) if den is not None else np.nan
    return (num / den) if (den and den != 0) else np.nan


def _metrics_from_counts(tp, fp, tn, fn):
    sen = _safe_ratio(tp, tp + fn)
    spe = _safe_ratio(tn, tn + fp)
    ppv = _safe_ratio(tp, tp + fp)
    npv = _safe_ratio(tn, tn + fn)
    acc = _safe_ratio(tp + tn, tp + tn + fp + fn)
    f1  = _safe_ratio(2 * tp, 2 * tp + fp + fn)
    return dict(sen=sen, spe=spe, ppv=ppv, npv=npv, acc=acc, f1=f1)


def _fmt_pct(x):
    return "NA" if (x is None or isinstance(x, float) and (np.isnan(x) or np.isinf(x))) else f"{100.0*float(x):.1f}%"


def _auc_from_roc_csv(csv_path: str):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        if VERBOSE_FIXED: print(f"[WARN] ROC CSV illisible: {csv_path} ({e})")
        return None
    cols = {c.lower(): c for c in df.columns}
    if "fpr" not in cols or "tpr" not in cols:
        for a in ("fp_rate", "fprate", "false_positive_rate"):
            if a in cols: cols["fpr"] = cols[a]
        for b in ("tp_rate", "tprate", "true_positive_rate", "recall", "tprc"):
            if b in cols and "tpr" not in cols: cols["tpr"] = cols[b]
    if "fpr" not in cols or "tpr" not in cols:
        if VERBOSE_FIXED: print(f"[WARN] Pas de colonnes FPR/TPR dans {csv_path} (cols={list(df.columns)})")
        return None
    x = np.asarray(df[cols["fpr"]], dtype=float)
    y = np.asarray(df[cols["tpr"]], dtype=float)
    order = np.argsort(x)
    x, y = np.clip(x[order], 0.0, 1.0), np.clip(y[order], 0.0, 1.0)
    auc = float(np.trapz(y, x))
    if x[0] > 0.0:  auc += y[0] * (x[0] - 0.0)
    if x[-1] < 1.0: auc += y[-1] * (1.0 - x[-1])
    return auc


def _find_entries_auc_latent(roots):
    rows = []
    for root in roots:
        if not os.path.isdir(root):
            if VERBOSE_FIXED: print(f"[WARN] Dossier absent: {root}")
            continue
        for d in os.listdir(root):
            lat_dir = os.path.join(root, d)
            if not os.path.isdir(lat_dir):
                continue
            mlat = LAT_DIR_RE.search(d)
            if not mlat:
                continue
            latent = int(mlat.group("latent"))
            for d2 in os.listdir(lat_dir):
                ak_dir = os.path.join(lat_dir, d2)
                if not os.path.isdir(ak_dir):
                    continue
                mak = AK_DIR_RE.match(d2)
                if not mak:
                    continue
                amin = _p2dot(mak.group("amin"))
                amax = _p2dot(mak.group("amax"))
                kval = _p2dot(mak.group("k"))
                metrics_json = os.path.join(ak_dir, "global_metrics.json")
                auc = None; md = None; src = None
                if os.path.isfile(metrics_json):
                    md = _safe_read_json(metrics_json)
                    if md is not None:
                        auc = md.get("auc", None); src = metrics_json
                if auc is None:
                    roc_csv = os.path.join(ak_dir, "roc_global.csv")
                    if os.path.isfile(roc_csv):
                        auc = _auc_from_roc_csv(roc_csv); src = roc_csv
                if auc is None or (isinstance(auc, float) and (math.isnan(auc) or np.isinf(auc))):
                    if VERBOSE_FIXED: print(f"[WARN] AUC indisponible pour: {ak_dir}")
                    continue
                row = {
                    "latent": latent, "alpha_min": float(amin), "alpha_max": float(amax), "k": float(kval),
                    "auc": float(auc),
                    "thr_youden": None if md is None else md.get("thr_youden", None),
                    "TN": None if md is None else md.get("TN", None),
                    "FP": None if md is None else md.get("FP", None),
                    "FN": None if md is None else md.get("FN", None),
                    "TP": None if md is None else md.get("TP", None),
                    "n_annotated": None if md is None else md.get("n_annotated", None),
                    "n_patients_used": None if md is None else md.get("n_patients_used", None),
                    "metrics_path": src, "run_dir": root,
                }
                rows.append(row)
    return rows


def plot_auc_vs_latent():
    entries = _find_entries_auc_latent(ROOTS_AUC_LATENT)
    if not entries:
        raise SystemExit("[ERROR] Aucun run trouvé (ni global_metrics.json ni roc_global.csv).")
    df = pd.DataFrame(entries)
    df.sort_values(["latent", "auc"], ascending=[True, False], inplace=True, kind="stable")
    df.reset_index(drop=True, inplace=True)
    if LATENTS_TO_PLOT_FIXED is not None:
        df = df[df["latent"].isin(LATENTS_TO_PLOT_FIXED)].copy()
        if df.empty:
            raise SystemExit("[ERROR] Table vide après filtre LATENTS_TO_PLOT.")

    subset = df[
        np.isclose(df["alpha_min"], ALPHA_MIN_FIXED, atol=1e-9) &
        np.isclose(df["alpha_max"], ALPHA_MAX_FIXED, atol=1e-9) &
        np.isclose(df["k"], K_FIXED, atol=1e-9)
    ].copy()
    subset.sort_values("latent", inplace=True)

    best_idx = df.groupby("latent")["auc"].idxmax()
    best_by_latent = df.loc[best_idx.values].copy().sort_values("latent")

    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), dpi=DPI)

    ax = axes[0]
    if not subset.empty:
        ax.plot(subset["latent"], subset["auc"], linestyle="--", linewidth=1.1, color="0.65", alpha=0.9)
        uniq_lat_sub = sorted(map(int, subset["latent"].unique()))
        cmap1 = plt.get_cmap("tab20")
        color_map1 = {lat: cmap1(i % cmap1.N) for i, lat in enumerate(uniq_lat_sub)}
        ax.scatter(subset["latent"], subset["auc"], s=42,
                   c=[color_map1[int(l)] for l in subset["latent"]])
        handles1 = [Line2D([0], [0], marker='o', linestyle='None',
                           color=color_map1[lat], label=f"lat{lat}", markersize=5)
                    for lat in uniq_lat_sub]
        ax.legend(handles=handles1, title="Latent", loc="lower right",
                  fontsize=7, title_fontsize=8, frameon=True, framealpha=0.85,
                  ncol=min(3, max(1, int(len(handles1) / 6) + 1)))
    else:
        ax.text(0.5, 0.5, "Aucun point pour ces (αmin, αmax, k)",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_title(f"AUC vs latent (αmin={ALPHA_MIN_FIXED}, αmax={ALPHA_MAX_FIXED}, k={K_FIXED})", fontsize=10)
    ax.set_xlabel("Latent", fontsize=9); ax.set_ylabel("AUC", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.45)

    ax = axes[1]
    ax.plot(best_by_latent["latent"], best_by_latent["auc"],
            linestyle="--", linewidth=1.1, color="0.65", alpha=0.9)

    uniq_lat_best = sorted(map(int, best_by_latent["latent"].unique()))
    cmap2 = plt.get_cmap("tab10")
    color_map2 = {lat: cmap2(i % cmap2.N) for i, lat in enumerate(uniq_lat_best)}
    ax.scatter(best_by_latent["latent"], best_by_latent["auc"], s=42,
               c=[color_map2[int(l)] for l in best_by_latent["latent"]])

    handles2 = [Line2D([0], [0], marker='o', linestyle='None',
                       color=color_map2[lat], label=f"lat{lat}", markersize=5)
                for lat in uniq_lat_best]
    ax.legend(handles=handles2, title="Latent", loc="lower left",
              fontsize=7, title_fontsize=8, frameon=True, framealpha=0.85,
              ncol=min(3, max(1, int(len(handles2) / 6) + 1)))

    best_row = best_by_latent.loc[best_by_latent["auc"].idxmax()]
    ax.scatter([best_row["latent"]], [best_row["auc"]],
               s=110, marker="s", facecolors="none", edgecolors="black", linewidths=1.2)

    best_handle = Line2D([0], [0], marker='s', linestyle='None',
                         markerfacecolor='none', markeredgecolor='black',
                         label=(f"BEST lat{int(best_row['latent'])} | "
                                f"αmin={best_row['alpha_min']}, "
                                f"αmax={best_row['alpha_max']}, k={best_row['k']}"),
                         markersize=7)

    tp, fp, tn, fn = (best_row.get("TP", None), best_row.get("FP", None),
                      best_row.get("TN", None), best_row.get("FN", None))
    metrics = _metrics_from_counts(tp, fp, tn, fn)
    text_box = (
        f"Se:  {_fmt_pct(metrics['sen'])}\n"
        f"Sp:  {_fmt_pct(metrics['spe'])}\n"
        f"VPP: {_fmt_pct(metrics['ppv'])}\n"
        f"VPN: {_fmt_pct(metrics['npv'])}\n"
        f"F1:  {_fmt_pct(metrics['f1'])}\n"
        f"Acc: {_fmt_pct(metrics['acc'])}"
    )
    ax.text(0.02, 0.98, text_box, transform=ax.transAxes, ha="left", va="top",
            fontsize=7, linespacing=1.2,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.9))

    ax.add_artist(ax.legend(handles=[best_handle], loc="upper right",
                            fontsize=7, frameon=True, framealpha=0.9, borderpad=0.6))

    ax.set_title("Meilleure AUC par latent", fontsize=10)
    ax.set_xlabel("Latent", fontsize=9); ax.set_ylabel("AUC", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.45)

    for ax in axes:
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.show()


# ============================================================
# Section 3 — ROC sweep (meilleure ROC par latent, robuste)
# ============================================================
ROOTS_ROC_SWEEP = [
    "/workspace/venv/vae_final/experiments/run_p100_p50_global_20250925-142111",
    "/workspace/venv/vae_final/experiments/run_p100_p50_global_20250925-094442",
]
LATENTS_TO_PLOT_ROC = None
SCALE_ROC = 0.8
FIG_W_ROC, FIG_H_ROC, DPI_ROC = 12 * SCALE_ROC, 8 * SCALE_ROC, 120
VERBOSE_PICK = True
VERBOSE_SCAN = True

LAT_RE = re.compile(r"lat(?P<latent>\d+)", re.IGNORECASE)
AK_RE  = re.compile(r"^amin(?P<amin>[0-9p]+)_amax(?P<amax>[0-9p]+)_k(?P<k>[0-9p]+)$")


def _coerce_float(x):
    if x is None: return None
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)): return None
        return float(x)
    if isinstance(x, str):
        try:
            fx = float(x)
            if math.isnan(fx) or math.isinf(fx): return None
            return fx
        except Exception:
            return None
    return None


def _safe_read_json_scan(path: str):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        if VERBOSE_SCAN:
            print(f"[WARN] JSON unreadable: {path} ({e})")
        return None


def _extract_auc_from_json(md: dict):
    for key in ("auc", "AUC", "roc_auc", "val_auc", "auc_global", "global_auc"):
        if key in md:
            v = _coerce_float(md[key])
            if v is not None:
                return v
    return None


def _auc_from_roc_csv_scan(csv_path: str):
    try:
        roc = pd.read_csv(csv_path)
    except Exception as e:
        if VERBOSE_SCAN:
            print(f"[WARN] ROC CSV unreadable: {csv_path} ({e})")
        return None
    cols = {c.lower(): c for c in roc.columns}
    fpr_col = cols.get("fpr") or cols.get("fp_rate") or cols.get("fprate") or cols.get("false_positive_rate")
    tpr_col = cols.get("tpr") or cols.get("tp_rate") or cols.get("tprate") or cols.get("true_positive_rate") or cols.get("recall")
    if not fpr_col or not tpr_col:
        if VERBOSE_SCAN:
            print(f"[WARN] Missing FPR/TPR columns in {csv_path} (cols={list(roc.columns)})")
        return None
    x = np.asarray(roc[fpr_col], dtype=float)
    y = np.asarray(roc[tpr_col], dtype=float)
    if x.size == 0 or y.size == 0:
        return None
    order = np.argsort(x)
    x = np.clip(x[order], 0.0, 1.0)
    y = np.clip(y[order], 0.0, 1.0)
    auc = float(np.trapz(y, x))
    if x[0] > 0.0:  auc += y[0]  * (x[0] - 0.0)
    if x[-1] < 1.0: auc += y[-1] * (1.0 - x[-1])
    return auc


def _read_roc_xy(csv_path: str):
    try:
        roc = pd.read_csv(csv_path)
    except Exception as e:
        if VERBOSE_SCAN:
            print(f"[WARN] ROC CSV unreadable: {csv_path} ({e})")
        return None, None
    cols = {c.lower(): c for c in roc.columns}
    fpr_col = cols.get("fpr") or cols.get("fp_rate") or cols.get("fprate") or cols.get("false_positive_rate")
    tpr_col = cols.get("tpr") or cols.get("tp_rate") or cols.get("tprate") or cols.get("true_positive_rate") or cols.get("recall")
    if not fpr_col or not tpr_col:
        if VERBOSE_SCAN:
            print(f"[WARN] Missing FPR/TPR columns in {csv_path} (cols={list(roc.columns)})")
        return None, None
    fpr = np.asarray(roc[fpr_col], dtype=float)
    tpr = np.asarray(roc[tpr_col], dtype=float)
    order = np.argsort(fpr)
    return np.clip(fpr[order], 0.0, 1.0), np.clip(tpr[order], 0.0, 1.0)


def _find_entries_roc(roots):
    rows = []
    for root in roots:
        if not os.path.isdir(root):
            if VERBOSE_SCAN:
                print(f"[WARN] Not a directory: {root}")
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            base = os.path.basename(dirpath)
            mak = AK_RE.match(base)
            if not mak:
                continue
            latent = None
            for part in dirpath.split(os.sep):
                mlat = LAT_RE.search(part)
                if mlat:
                    latent = int(mlat.group("latent"))
                    break
            if latent is None:
                if VERBOSE_SCAN:
                    print(f"[WARN] No 'lat<d>' in path: {dirpath}")
                continue
            amin = _p2dot(mak.group("amin")); amax = _p2dot(mak.group("amax")); kval = _p2dot(mak.group("k"))
            metrics_json = os.path.join(dirpath, "global_metrics.json")
            roc_csv      = os.path.join(dirpath, "roc_global.csv")
            auc = None; src = None; md = None
            if os.path.isfile(metrics_json):
                md = _safe_read_json_scan(metrics_json)
                if md is not None:
                    auc = _extract_auc_from_json(md)
                    if auc is not None: src = metrics_json
            if auc is None and os.path.isfile(roc_csv):
                auc = _auc_from_roc_csv_scan(roc_csv)
                if auc is not None: src = roc_csv
            if auc is None:
                if VERBOSE_SCAN:
                    print(f"[WARN] AUC unavailable for: {dirpath}")
                continue
            rows.append({
                "latent": latent, "alpha_min": float(amin), "alpha_max": float(amax), "k": float(kval),
                "auc": float(auc), "roc_path": roc_csv if os.path.isfile(roc_csv) else None,
                "metrics_path": metrics_json if os.path.isfile(metrics_json) else None,
                "ak_dir": dirpath,
            })
    return rows


def plot_best_roc_per_latent():
    entries = _find_entries_roc(ROOTS_ROC_SWEEP)
    if not entries:
        raise SystemExit("[ERROR] No experiments found (neither global_metrics.json nor roc_global.csv).")
    df = pd.DataFrame(entries)
    if LATENTS_TO_PLOT_ROC is not None:
        df = df[df["latent"].isin(LATENTS_TO_PLOT_ROC)].copy()
        if df.empty:
            raise SystemExit("[ERROR] No rows left after LATENTS_TO_PLOT filter.")
    df.sort_values(["latent", "auc"], ascending=[True, False], inplace=True, kind="stable")
    best_idx = df.groupby("latent")["auc"].idxmax()
    best = df.loc[best_idx.values].copy().sort_values("latent")

    if VERBOSE_PICK:
        print("=== Best per latent ===")
        for _, r in best.iterrows():
            print(
                f"lat{int(r['latent'])} | AUC={r['auc']:.3f} "
                f"(αmin={r['alpha_min']}, αmax={r['alpha_max']}, k={r['k']}) "
                f"-> ROC={r['roc_path'] or 'N/A'} | SRC={'JSON' if r['metrics_path'] else 'ROC'}"
            )

    plt.figure(figsize=(FIG_W_ROC, FIG_H_ROC), dpi=DPI_ROC)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Chance")

    n_plotted = 0
    for _, row in best.iterrows():
        if not row["roc_path"] or not os.path.isfile(row["roc_path"]):
            if VERBOSE_SCAN:
                print(f"[WARN] No ROC CSV for plotting: {row['ak_dir']}")
            continue
        fpr, tpr = _read_roc_xy(row["roc_path"])
        if fpr is None or tpr is None:
            continue
        label = (f"lat{int(row['latent'])} | AUC={row['auc']:.3f} "
                 f"(αmin={row['alpha_min']}, αmax={row['alpha_max']}, k={row['k']})")
        plt.plot(fpr, tpr, linewidth=2.0, label=label)
        n_plotted += 1

    plt.title("Best ROC per latent (robust scan)", fontsize=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=10)
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.5)
    if n_plotted > 0:
        plt.legend(loc="lower right", fontsize=8, frameon=True)
    else:
        plt.text(0.5, 0.5, "No ROC to plot (check folders/files)", ha="center", va="center")
    plt.tight_layout()
    plt.show()


# =========================================================================
# Section 4 — Courbes d'entraînement (train/val) superposées par latent
# =========================================================================
ROOT_HIST = "/workspace/venv/vae_final/model_new"
CSV_PATTERNS = ["**/*.csv", "**/*history*.csv", "**/log*.csv"]
SAVE_DIR = os.path.join(ROOT_HIST, "_plots")
EMA_ALPHA = 0.5
SHOW_GRID = True
TITLE_PREFIX = "Courbes d'entraînement par latent"
REQUIRED = {"epoch", "cls_loss", "val_cls_loss", "recon_loss", "val_recon_loss"}


def ema(x, alpha):
    if alpha is None or alpha <= 0: return np.asarray(x, float)
    x = np.asarray(x, float)
    if x.size == 0: return x
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def extract_latent(path):
    m = re.search(r"run_lat(\d+)", path)
    if m:
        return int(m.group(1))
    m = re.search(r"lat(\d+)", path)
    return int(m.group(1)) if m else None


def find_histories(root, patterns):
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(root, pat), recursive=True))
    out = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        cols = set(c.strip() for c in df.columns)
        if "epoch" not in cols:
            df = df.copy()
            df.insert(0, "epoch", np.arange(len(df)))
            cols = set(c.strip() for c in df.columns)
        if REQUIRED.issubset(cols):
            lat = extract_latent(f)
            out.append((lat, f, df))
    by_lat = {}
    for lat, f, df in out:
        if lat is None:
            continue
        mtime = os.path.getmtime(f)
        if lat not in by_lat or mtime > by_lat[lat][0]:
            by_lat[lat] = (mtime, f, df)
    items = sorted([(lat, f, df) for lat, (mt, f, df) in by_lat.items()], key=lambda t: t[0])
    return items


def _idx_from_frac(n, frac):
    return 0 if n <= 0 else max(0, min(n - 1, int(round(frac * (n - 1)))))


def _offset_y(y_val, ylog: bool, sign: int, ax):
    if not np.isfinite(y_val): return 0.0
    if ylog:
        factor = 1.03 if sign > 0 else (1.0 / 1.03)
        return y_val * (factor - 1.0)
    ylo, yhi = ax.get_ylim()
    return sign * 0.02 * (yhi - ylo)


def _offset_x(ep, frac, lat, sign):
    if len(ep) == 0: return 0.0
    xr = float(ep.max() - ep.min()) if ep.max() != ep.min() else 1.0
    base = 0.1 * xr
    jitter = ((hash((int(lat), float(frac))) % 7) - 3) / 3.0
    return sign * base * (0.6 + 0.2 * jitter)


def plot_overlaid(metric_key, val_key, ylabel, title_suffix, ylog=False, ema_alpha=EMA_ALPHA):
    from matplotlib.lines import Line2D
    histories = find_histories(ROOT_HIST, CSV_PATTERNS)
    if not histories:
        raise RuntimeError(
            "Aucun CSV d'historique trouvé avec les colonnes requises "
            f"({', '.join(sorted(REQUIRED))}) sous {ROOT_HIST}"
        )
    os.makedirs(SAVE_DIR, exist_ok=True)

    uniq_lat = sorted({lat for lat, _, _ in histories})
    cmap = plt.get_cmap("tab20")
    COLOR_MAP = {lat: cmap(i % cmap.N) for i, lat in enumerate(uniq_lat)}

    fig, ax = plt.subplots(figsize=(10, 6))

    for lat, f, df in histories:
        if metric_key not in df.columns or val_key not in df.columns:
            continue
        ep = df["epoch"].values
        y  = df[metric_key].values
        yv = df[val_key].values
        if ema_alpha:
            y  = ema(y, ema_alpha)
            yv = ema(yv, ema_alpha)

        c = COLOR_MAP.get(lat, "gray")
        ax.plot(ep, y,  color=c, linewidth=1.8, linestyle="--")
        ax.plot(ep, yv, color=c, linewidth=2.0, linestyle="-")

        LABEL_FRACTIONS = (0.20, 0.80)
        LABEL_FONTSIZE  = 9
        LABEL_BOX = dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5)
        idxs = [_idx_from_frac(len(ep), f) for f in LABEL_FRACTIONS]
        for frac, k in zip(LABEL_FRACTIONS, idxs):
            xk_val = ep[k] + _offset_x(ep, frac, lat, sign=+1)
            yk_val = yv[k] + _offset_y(yv[k], ylog, +1, ax)
            ax.text(xk_val, yk_val, str(lat), color=c, fontsize=LABEL_FONTSIZE,
                    weight="bold", ha="left", va="center", bbox=LABEL_BOX)

            xk_tr  = ep[k] + _offset_x(ep, frac, lat, sign=-1)
            yk_tr  = y[k]  + _offset_y(y[k],  ylog, -1, ax)
            ax.text(xk_tr, yk_tr,  str(lat), color=c, fontsize=LABEL_FONTSIZE,
                    weight="bold", ha="right", va="center", bbox=LABEL_BOX)

    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    if ylog:
        ax.set_yscale("log")
    if SHOW_GRID:
        ax.grid(True, alpha=0.3)
    ax.set_title(f"{TITLE_PREFIX} — {title_suffix}")

    style_legend = [
        Line2D([0],[0], color="black", linestyle="-",  linewidth=2.0, label="validation"),
        Line2D([0],[0], color="black", linestyle="--", linewidth=1.8, label="train"),
    ]
    ax.legend(handles=style_legend, loc="upper right", frameon=True)

    fig.tight_layout()
    out = os.path.join(SAVE_DIR, f"{metric_key}_overlaid.png")
    fig.savefig(out, dpi=160)
    print(f"[saved] {out}")
    plt.show()


# ============================================================
# Entrées de script (utiliser/adapter si besoin)
# ============================================================
if __name__ == "__main__":
    # 1) Exemples 
    #
    # overlay_recon_window(
    #     npz_path="/workspace/venv/vae_final/file_npz/patients_100_clean.npz",
    #     model_paths=[
    #         "/workspace/venv/vae_final/model_new/run_lat16/best_full_NEWAE1D_lat16_bs512_frac25_down3_base16_spe512_20250925-121230.keras",
    #         "/workspace/venv/vae_final/model_new/run_lat64/best_full_NEWAE1D_lat64_bs512_frac25_down3_base16_spe512_20250925-131912.keras",
    #     ],
    #     fs=115.0, t_start_sec=10357, win_s=5, target_channel=0, vis_denorm=True, valid_strict=0.999
    # )
    #
    # plot_auc_vs_latent()
    # plot_best_roc_per_latent()
    #
    # plot_overlaid("cls_loss", "val_cls_loss", ylabel="Classification loss",
    #               title_suffix="Classification (train & val)", ylog=False)
    # plot_overlaid("recon_loss", "val_recon_loss", ylabel="Reconstruction loss",
    #               title_suffix="Reconstruction (train & val)", ylog=False)
    # # si présent:
    # # plot_overlaid("loss", "val_loss", ylabel="Total loss",
    # #               title_suffix="Total (train & val)", ylog=False)
    pass
