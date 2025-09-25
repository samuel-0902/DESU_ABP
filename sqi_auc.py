# sqi_auc.py
# ----------------------------------------
# SQI + préprocessing minimal – évalue UNIQUEMENT sur les fenêtres annotées.
# SQI haut = bonne qualité. AUC calculée sur (1 - SQI) = score d'anomalie.
# ---------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

# ---------------------------- utils modèle ----------------------------
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

def compute_auc_on_annotated_many(model, npz_patient_paths, labels_npz,
                                  *, fs, target_recon_channel=0,
                                  alpha_min=0.2, alpha_max=0.8, k=5.0,
                                  batch_size=256, verbose=0,
                                  return_per_patient=False, sqi_module=None):
    """
    Rôle : calcule l’AUC globale (et par patient en option) en n’utilisant QUE les fenêtres annotées.
    - Agrège sur plusieurs fichiers patients NPZ.
    - Combine classif + reconstruction en score d’anomalie s.
    - Normalise RMSE par percentiles (10/90).
    - Retourne un rapport global et, si demandé, une liste détaillée par patient.
    """
    if sqi_module is None:
        raise ValueError("Passe `sqi_module=sqi` (le module sqi importé).")

    all_scores = []
    all_labels = []
    per_patient = []

    for pth in npz_patient_paths:
        d = np.load(pth, allow_pickle=True)
        X_any = d["X"].astype("float32")
        meta  = list(d["meta"])

        # Fenêtrage + adaptation au modèle
        wins, mask_win, starts, L_win, L_total, t0 = sqi_module.windows_from_meta_any(X_any, meta, fs)
        print("[DBG] N_total_wins        =", wins.shape[0])

        wins, mask_win = sqi_module.adapt_for_model(wins, mask_win, model)

        # Garde uniquement les fenêtres annotées
        wins_a, y_a, kept_idx = sqi_module.filter_to_annotated(wins, meta, labels_npz)
        print("[DBG] N_annotated_overlap =", wins_a.shape[0],
             "| y=0:", int((y_a==0).sum()), " y=1:", int((y_a==1).sum()))

        mask_a = mask_win[kept_idx]

        if len(y_a) == 0:
            per_patient.append({"patient": pth, "n_annotated": 0})
            continue
 

        # Prédictions 
        try:
            recon_pred, p_abn = sqi_module._pick_outputs(model, wins_a, batch_size=batch_size, verbose=verbose)
        except Exception:
            # Modèle AE seul: pas de tête cls -> p_abn=0
            pred = model.predict(wins_a, batch_size=batch_size, verbose=verbose)
            recon_pred = pred if isinstance(pred, (np.ndarray,)) else np.asarray(pred)
            if recon_pred.ndim != 3:
                raise ValueError("Sortie du modèle inattendue: on attend un tenseur (N,L,C) pour la reconstruction.")
            p_abn = np.zeros((recon_pred.shape[0],), dtype=np.float32)

        # Sélectionne le canal de reconstruction ciblé
        if recon_pred.shape[-1] > 1:
            recon_pred = recon_pred[..., target_recon_channel:target_recon_channel+1]
        y_true_recon = wins_a[..., target_recon_channel:target_recon_channel+1]

        # RMSE asquée par fenêtre
        rmse = sqi_module._rmse_per_window_masked(
            y_true_recon, recon_pred, mask_a, min_coverage=0.1
        )
        keep_cov = np.isfinite(rmse)
        print("[DBG] N_after_coverage    =", int(keep_cov.sum()), "/", rmse.size)

        # Filtre fenêtres valides
        keep = np.isfinite(rmse)
        if not np.any(keep):
            per_patient.append({"patient": pth, "n_annotated": 0})
            continue
        rmse = rmse[keep]
        p_abn = p_abn[keep]
        y_a   = y_a[keep]

        # Normalisation robuste du RMSE (percentiles )
        lo = np.nanpercentile(rmse, 10.0)
        hi = np.nanpercentile(rmse, 90.0)
        if not np.isfinite(lo): lo = np.nanmin(rmse)
        if not np.isfinite(hi): hi = np.nanmax(rmse)
        if hi <= lo: hi = lo + 1e-8
        rmse_norm = np.clip((rmse - lo) / (hi - lo), 0.0, 1.0)

        # Score d'anomalie (pas SQI): s = α(p)*p_abn + (1-α)*rmse_norm
        try:
            alpha = sqi_module._alpha_cls(p_abn, alpha_min=alpha_min, alpha_max=alpha_max, k=k)
        except TypeError:
            
            alpha = sqi_module._alpha_curve(p_abn, alpha_min=alpha_min, alpha_max=alpha_max, k=k)

        s = alpha * p_abn + (1.0 - alpha) * rmse_norm
        ok = np.isfinite(s)
        print("[DBG] N_final_for_ROC     =", int(ok.sum()), "/", int(s.size))
        if ok.any():
            print("[DBG] classes finales     | y=0:", int((y_a[ok]==0).sum()),
                " y=1:", int((y_a[ok]==1).sum()))

        # Sécurité NaN
        ok = np.isfinite(s)
        y_a = y_a[ok]; s = s[ok]
        if y_a.size == 0 or len(np.unique(y_a)) < 2:
            per_patient.append({"patient": pth, "n_annotated": int(len(y_a))})
            continue

        # AUC par patient
        fpr_p, tpr_p, thr_p = roc_curve(y_a, s)
        youden_p = int(np.argmax(tpr_p - fpr_p))
        thr_p_y  = float(thr_p[youden_p]) if youden_p < len(thr_p) else float("inf")
        tn, fp, fn, tp = confusion_matrix(y_a, (s >= thr_p_y).astype(int), labels=[0,1]).ravel()

        per_patient.append({
            "patient": pth,
            "n_annotated": int(len(y_a)),
            "auc": float(roc_auc_score(y_a, s)),
            "thr_youden": thr_p_y,
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        })

        
        all_labels.append(y_a)
        all_scores.append(s)

    # ---------------- ROC/AUC GLOBALE-----------
    if len(all_labels) == 0:
        
        raise RuntimeError(
            "Aucune fenêtre annotée valide sur l’ensemble des patients (RMSE NaN ou classes absentes)."
        )

    y_all = np.concatenate(all_labels)
    s_all = np.concatenate(all_scores)

    okg = np.isfinite(s_all)
    y_all = y_all[okg]; s_all = s_all[okg]
    if y_all.size == 0 or len(np.unique(y_all)) < 2:
        raise RuntimeError("Pas assez d’échantillons/classes valides pour la ROC globale.")

    fpr, tpr, thr = roc_curve(y_all, s_all)
    youden = int(np.argmax(tpr - fpr))
    thr_y  = float(thr[youden]) if youden < len(thr) else float("inf")
    auc    = float(roc_auc_score(y_all, s_all))
    tn, fp, fn, tp = confusion_matrix(y_all, (s_all >= thr_y).astype(int), labels=[0,1]).ravel()

    global_report = {
        "auc": auc,
        "thr_youden": thr_y,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "n_annotated": int(len(y_all)),
        "n_patients_used": int(len([p for p in per_patient if p.get("n_annotated",0) > 0])),
    }
    return (global_report, per_patient) if return_per_patient else (global_report, None)


def _alpha_cls(p_abn, alpha_min=0.2, alpha_max=0.8, k=5.0):
    """Rôle : calcule α(p) pour mixer classif et reco (linéaire jusqu’à 0.5 puis croissance exp)."""
    p = np.clip(np.asarray(p_abn, np.float32), 0.0, 1.0)
    a = np.empty_like(p)
    left = p <= 0.5
    a[left] = alpha_max - (alpha_max - alpha_min) * (p[left] / 0.5)
    right = ~left
    x = (p[right] - 0.5) / 0.5
    grow = 1.0 - np.exp(-k * x)
    a[right] = alpha_min + (alpha_max - alpha_min) * grow
    return a

def _rmse_per_window(y_true, y_pred, eps=1e-8):
    """Rôle : calcule la RMSE par fenêtre (NaN-safe, masque implicite sur y_true)."""
    y_true = np.asarray(y_true, np.float32)
    y_pred = np.asarray(y_pred, np.float32)
    mask   = np.isfinite(y_true)
    y0     = np.where(mask, y_true, 0.0)
    diff2  = (y_pred - y0) ** 2 * mask.astype(np.float32)
    num    = diff2.reshape(diff2.shape[0], -1).sum(axis=1)
    den    = mask.reshape(mask.shape[0], -1).sum(axis=1).astype(np.float32) + eps
    return np.sqrt(num / den)

def _norm_rmse(rmse, lo=None, hi=None, p_lo=10.0, p_hi=90.0, eps=1e-8):
    """Rôle : normalise un vecteur de RMSE dans [0,1] via percentiles."""
    r = np.asarray(rmse, np.float32)
    if lo is None or hi is None:
        lo = float(np.percentile(r, p_lo))
        hi = float(np.percentile(r, p_hi))
        if hi <= lo:
            lo = float(np.min(r))
            hi = float(np.max(r) + 1e-6)
    rn = np.clip((r - lo) / (hi - lo + eps), 0.0, 1.0)
    return rn, {"lo": lo, "hi": hi}

def _pick_outputs(model, X, batch_size=512, verbose=0):
    """Rôle : identifie et renvoie (recon, p_abn) parmi les sorties du modèle."""
    pred = model.predict(X, batch_size=batch_size, verbose=verbose)
    outs = pred if isinstance(pred, (list, tuple)) else [pred]
    recon, p_cls = None, None
    for o in outs:
        o = np.asarray(o)
        if o.ndim == 3 and o.shape[1] > 1:
            recon = o
        elif o.ndim in (1, 2) and o.shape[-1] == 1:
            p_cls = o.reshape(-1)
    if recon is None or p_cls is None:
        names = getattr(model, "output_names", [])
        for name, o in zip(names, outs):
            if recon is None and "recon" in name.lower():
                recon = np.asarray(o)
            if p_cls is None and "cls" in name.lower():
                p_cls = np.asarray(o).reshape(-1)
    if recon is None or p_cls is None:
        raise ValueError("Impossible d’identifier les sorties recon/cls du modèle.")
    return recon.astype(np.float32), p_cls.astype(np.float32)

def compute_sqi_from_model(model, X, * ,
                           target_recon_channel=0,
                           alpha_min=0.2, alpha_max=0.8, k=5.0,
                           batch_size=512, rmse_lo=None, rmse_hi=None, verbose=0,
                           mask=None, min_coverage=0.10):
    """
    Rôle  : calcule le SQI par fenêtre à  partir d’un modèle [recon, cls].
    - Supporte un masque de validité par fenêtre (couverture minimale).
    - Retourne `sqi` (N,) avec NaN pour fenêtres invalides et un dict `info`.
    """
    X = np.asarray(X, np.float32)
    recon_pred, p_abn = _pick_outputs(model, X, batch_size=batch_size, verbose=verbose)
    
    if y_a.size > 0:
        try:
            print("[DBG] p_abn mean by class | y=0:",
                float(np.nanmean(p_abn[y_a==0])),
                " y=1:", float(np.nanmean(p_abn[y_a==1])))
        except Exception:
            pass

    # sélection canal
    if recon_pred.shape[-1] > 1:
        recon_pred = recon_pred[..., target_recon_channel:target_recon_channel+1]
    y_true_recon = X[..., target_recon_channel:target_recon_channel+1]

    # RMSE 
    if mask is not None:
        rmse = _rmse_per_window_masked(y_true_recon, recon_pred, mask, min_coverage=min_coverage)
    else:
        rmse = _rmse_per_window(y_true_recon, recon_pred)

    #Normalisation
    fin = np.isfinite(rmse)
    if not np.any(fin):
        # tout invalide -> tout NaN
        rn_full = np.full_like(rmse, np.nan, dtype=np.float32)
    else:
        if rmse_lo is None or rmse_hi is None:
            lo = float(np.percentile(rmse[fin], 10.0))
            hi = float(np.percentile(rmse[fin], 90.0))
            if hi <= lo: hi = lo + 1e-8
        else:
            lo, hi = float(rmse_lo), float(rmse_hi)
            if hi <= lo: hi = lo + 1e-8
        rn = np.clip((rmse[fin] - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        rn_full = np.full_like(rmse, np.nan, dtype=np.float32)
        rn_full[fin] = rn

    # Mixage
    alpha     = _alpha_cls(p_abn, alpha_min=alpha_min, alpha_max=alpha_max, k=k)
    score_cls = 1.0 - p_abn
    score_rec = 1.0 - rn_full

    sqi = alpha * score_cls + (1.0 - alpha) * score_rec
    # Les indices où rn_full est NaN restent NaN dans SQI (fenêtres invalides)
    sqi = np.clip(sqi, 0.0, 1.0)
    sqi[~np.isfinite(rn_full)] = np.nan

    info = {"p_abn": p_abn, "alpha": alpha, "rmse": rmse, "rmse_norm": rn_full,
            "calib": {"lo": None, "hi": None}, "score_cls": score_cls, "score_rec": score_rec}
    return sqi, info


# ------------------------ préprocessing fenêtres ----------------------

# cache global simple pour éviter de recharger à chaque appel
_TRAIN_SCALER = None

def prepare_for_model(X_any):
    """
    Rôle : prépare (N,L,3+) → (N,L,4) avec le 4ᵉ canal = moyenne PAR FENÊTRE du canal 0,
    standardisée par le scaler appris sur le TRAIN, puis répliquée sur L.
    """
    global _TRAIN_SCALER
    if _TRAIN_SCALER is None:
        _TRAIN_SCALER = _build_train_mean_scaler(FINAL_DATASET_PATH)

    if X_any.ndim != 3 or X_any.shape[2] < 3:
        raise ValueError(f"X_any attendu (N,L,3+), obtenu {X_any.shape}")

    abp  = np.nan_to_num(X_any[:, :, 0], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    age_col    = X_any[:, :, 1]
    finite_age = age_col[np.isfinite(age_col)]
    age_val    = float(finite_age[0]) if finite_age.size > 0 else 0.0
    age        = np.full_like(abp, age_val, dtype=np.float32)

    # ffill + clamp 0..1
    state = X_any[:, :, 2].astype(np.float32).copy()
    for i in range(state.shape[0]):
        last, seen = 0.0, False
        for j in range(state.shape[1]):
            v = state[i, j]
            if np.isfinite(v): last, seen = v, True
            else:              state[i, j] = last if seen else 0.0
    state = np.clip(state, 0.0, 1.0)

    # ---- 4ᵉ canal : moyenne PAR FENÊTRE standardisée avec le scaler du TRAIN ----
    m_win = np.nanmean(abp, axis=1).astype(np.float32).reshape(-1, 1)   # (N,1)
    z = _TRAIN_SCALER.transform(m_win).astype(np.float32).reshape(-1)    # (N,)
    canal4 = np.repeat(z[:, None], abp.shape[1], axis=1).astype(np.float32)

    return np.stack([abp, age, state, canal4], axis=-1).astype(np.float32)


def _pad_or_crop(win, L_win):
    """Rôle : ajuste une fenêtre 2D en coupant/paddant à L_win, sans changer les canaux."""
    L0, _ = win.shape
    if L0 == L_win: return win
    if L0 >  L_win: return win[:L_win]
    pad = np.zeros((L_win - L0, win.shape[1]), dtype=win.dtype)
    return np.concatenate([win, pad], axis=0)

def windows_from_meta_any(X_any, meta, fs):
    """
    Rôle : génère les fenêtres (wins), un masque de validité (mask_win),
    les indices de départ (starts), la longueur de fenêtre (L_win),
    la longueur totale estimée (L_total) et le timestamp de référence (t0).
    """
    import numpy as np
    import pandas as pd

    X_any = np.asarray(X_any)
    if len(meta) == 0:
        raise ValueError("meta est vide.")
    t0 = pd.to_datetime(meta[0]['ts'])

    if X_any.ndim == 3:
        N_win, L_win, _ = X_any.shape
        raw_abp  = X_any[:, :, 0]
        mask_win = np.isfinite(raw_abp).astype(np.float32)[..., None]
        wins     = prepare_for_model(X_any)  # (N,L,4)
        starts   = np.array([int(round((pd.to_datetime(m["ts"]) - t0).total_seconds() * fs)) for m in meta], dtype=int)
        te_last  = pd.to_datetime(meta[-1]['te'])
        L_total  = int(np.round((te_last - t0).total_seconds() * fs))
        L_total  = max(L_total, int(starts.max() + L_win))
    elif X_any.ndim == 2:
        win_s   = (pd.to_datetime(meta[0]['te']) - pd.to_datetime(meta[0]['ts'])).total_seconds()
        L_win   = int(round(win_s * fs))
        starts  = np.array([int(round((pd.to_datetime(m["ts"]) - t0).total_seconds() * fs)) for m in meta], dtype=int)
        te_last = pd.to_datetime(meta[-1]['te'])
        L_total = int(np.round((te_last - t0).total_seconds() * fs))
        raw_wins = [X_any[s:s+L_win] if s+L_win<=X_any.shape[0]
                    else np.pad(X_any[s:], ((0, s+L_win-X_any.shape[0]), (0, 0)))
                    for s in starts]
        raw_wins = np.stack(raw_wins, axis=0)
        mask_win = np.isfinite(raw_wins[:, :, 0]).astype(np.float32)[..., None]
        wins     = prepare_for_model(raw_wins)
    else:
        raise ValueError(f"X_any shape inattendue: {X_any.shape}")

    wins     = np.nan_to_num(wins, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    mask_win = (mask_win > 0).astype(np.float32)
    return wins, mask_win, starts, L_win, L_total, t0


def adapt_for_model(wins, mask_win, model):
    """Rôle : adapte fenêtres/masques aux shapes attendues par le modèle (longueur, canaux)."""
    Lm = model.input_shape[1]
    Cm = model.input_shape[-1]
    # temps
    if wins.shape[1] > Lm:
        wins = wins[:, :Lm, :]
        if mask_win is not None: mask_win = mask_win[:, :Lm, :]
    elif wins.shape[1] < Lm:
        dt = Lm - wins.shape[1]
        wins = np.concatenate([wins, np.zeros((wins.shape[0], dt, wins.shape[2]), wins.dtype)], axis=1)
        if mask_win is not None:
            mask_win = np.concatenate([mask_win, np.zeros((mask_win.shape[0], dt, mask_win.shape[2]), mask_win.dtype)], axis=1)
    # canaux
    if wins.shape[2] > Cm:
        wins = wins[:, :, :Cm]
    elif wins.shape[2] < Cm:
        dc = Cm - wins.shape[2]
        wins = np.concatenate([wins, np.zeros((wins.shape[0], wins.shape[1], dc), wins.dtype)], axis=-1)
    # masque safe
    if mask_win is None:
        mask_win = np.ones((wins.shape[0], wins.shape[1], 1), dtype=np.float32)
    elif mask_win.ndim == 2:
        mask_win = mask_win[:, :, None]
    return wins.astype(np.float32), mask_win.astype(np.float32)

from sklearn.preprocessing import StandardScaler

FINAL_DATASET_PATH = "/workspace/venv/vae_final/final_dataset.npz"

def _build_train_mean_scaler(path_final_npz: str) -> StandardScaler:
    """Rôle : apprend un StandardScaler sur la moyenne PAR FENÊTRE du canal 0 du TRAIN."""
    d = np.load(path_final_npz, allow_pickle=True)
    Xtr = d["X_supervised_train"]  # (N, L, C>=1)
    m_win = np.nanmean(Xtr[:, :, 0], axis=1).astype(np.float32)  # moyenne PAR FENÊTRE
    sc = StandardScaler().fit(m_win.reshape(-1, 1))
    return sc


#------------------- n’utiliser QUE les annotées --------

def label_windows_and_mask(meta_patient, labels_npz, *, majority=0.1, bad_value=2):
    """
    Rôle : marque les fenêtres patient qui chevauchent une annotation et assigne y∈{0,1}.
    - mask_annotated[i]=True si chevauche un intervalle labellisé.
    - y_annot[i]=1 si proportion de `bad_value` >= majority (sinon 0).
    """
    if "y" not in labels_npz or "meta" not in labels_npz:
        raise ValueError("labels_npz doit contenir 'y' et 'meta'.")

    def _extract(meta_like):
        rows = []
        for m in meta_like:
            ts = pd.to_datetime(m.get("ts", m.get("start_time", m.get("start"))))
            te = pd.to_datetime(m.get("te", m.get("end_time",   m.get("end"))))
            rows.append({"ts": ts, "te": te})
        return pd.DataFrame(rows)

    df_p = _extract(meta_patient)
    df_l = _extract(labels_npz["meta"])
    y_raw = np.asarray(labels_npz["y"], int)
    n = min(len(df_l), len(y_raw))
    df_l = df_l.iloc[:n].reset_index(drop=True)
    y_bad = (y_raw[:n] == bad_value).astype(int)

    y_out = np.zeros((len(df_p),), dtype=int)
    m_out = np.zeros((len(df_p),), dtype=bool)
    for i, row in df_p.iterrows():
        ts_p, te_p = row.ts, row.te
        overlap = (df_l["te"] > ts_p) & (df_l["ts"] < te_p)
        idx = np.where(overlap.values)[0]
        if len(idx) > 0:
            m_out[i] = True
            y_out[i] = 1 if (y_bad[idx].mean() >= majority) else 0
    return y_out, m_out

def _rmse_per_window_masked(y_true, y_pred, mask_win, min_coverage=0.10, eps=1e-8):
    """
    Rôle : RMSE par fenêtre en respectant un masque (validité avant imputation).
    - Exclut (NaN) les fenêtres dont la couverture valide < min_coverage.
    """
    yt = np.asarray(y_true, np.float32)
    yp = np.asarray(y_pred, np.float32)
    mw = np.asarray(mask_win, np.float32)
    if mw.ndim == 3 and mw.shape[-1] != 1:
        mw = mw[..., :1]
    if mw.ndim == 2:
        mw = mw[:, :, None]

    finite = np.isfinite(yt) & np.isfinite(yp)
    mask = (mw > 0) & finite

    y0 = np.where(mask, yt, 0.0)
    diff2 = (yp - y0) ** 2 * mask.astype(np.float32)

    num = diff2.reshape(diff2.shape[0], -1).sum(axis=1)
    den = mask.reshape(mask.shape[0], -1).sum(axis=1).astype(np.float32)

    rmse = np.sqrt(num / (den + eps)).astype(np.float32)

    L_eff = mask.shape[1] * mask.shape[2]
    cov = den / max(L_eff, 1)
    rmse[cov < float(min_coverage)] = np.nan
    return rmse


def _alpha_curve(p, alpha_min=0.2, alpha_max=0.8, k=5.0):
    """Rôle : alias de compatibilité pour _alpha_cls (mêmes paramètres/comportement)."""
    return _alpha_cls(p, alpha_min=alpha_min, alpha_max=alpha_max, k=k)

# -- Alignement labels (utilisé par la grid) -
def align_labels_by_time(meta_patient, labels_npz, majority=0.5, bad_value=2):
    """
    Rôle : produit un vecteur y_bin (0/1) aligné sur meta_patient via chevauchement temporel.
    1 si proportion de `bad_value` sur l’intervalle ≥ majority, sinon 0.
    """
    if "y" not in labels_npz or "meta" not in labels_npz:
        raise ValueError("npz_labels doit contenir 'y' et 'meta'.")

    import pandas as pd
    import numpy as np

    def _extract(meta_like):
        rows = []
        for m in meta_like:
            ts = pd.to_datetime(m.get("ts", m.get("start_time", m.get("start"))))
            te = pd.to_datetime(m.get("te", m.get("end_time",   m.get("end"))))
            rows.append({"ts": ts, "te": te})
        return pd.DataFrame(rows)

    df_p = _extract(meta_patient)
    df_l = _extract(labels_npz["meta"])
    y_raw = np.asarray(labels_npz["y"], int)

    n = min(len(df_l), len(y_raw))
    df_l = df_l.iloc[:n].reset_index(drop=True)
    y_bad = (y_raw[:n] == bad_value).astype(int)

    y_out = []
    for _, row in df_p.iterrows():
        ts_p, te_p = row.ts, row.te
        overlap = (df_l["te"] > ts_p) & (df_l["ts"] < te_p)
        idx = np.where(overlap.values)[0]
        if len(idx) == 0:
            y_out.append(0)
        else:
            y_out.append(1 if (y_bad[idx].mean() >= majority) else 0)
    return np.asarray(y_out, dtype=int)

def filter_to_annotated(wins, meta_patient, labels_npz, *, majority=0.5, bad_value=2):
    """Rôle : extrait (wins_annotés, y_annotés, indices) selon chevauchement ≥ majority."""
    y_all, m = label_windows_and_mask(meta_patient, labels_npz, majority=majority, bad_value=bad_value)
    idx = np.where(m)[0]
    return wins[idx], y_all[idx], idx

def compute_auc_on_annotated(model, X_any, meta_patient, labels_npz, * ,
                             fs, target_recon_channel=0,
                             alpha_min=0.2, alpha_max=0.8, k=5.0,
                             batch_size=256, verbose=0):
    """
    Rôle : AUC sur un seul patient en n’utilisant que les fenêtres annotées.
    - Construit les fenêtres, adapte au modèle, calcule SQI, puis ROC/AUC.
    """
    wins, mask_win, starts, L_win, L_total, t0 = windows_from_meta_any(X_any, meta_patient, fs)
    wins, mask_win = adapt_for_model(wins, mask_win, model)

    wins_a, y_a, kept_idx = filter_to_annotated(wins, meta_patient, labels_npz)
    mask_a = mask_win[kept_idx]
    if len(y_a) == 0:
        raise RuntimeError("Aucune fenêtre annotée trouvée.")

    sqi_a, info = compute_sqi_from_model(
        model, wins_a,
        target_recon_channel=target_recon_channel,
        alpha_min=alpha_min, alpha_max=alpha_max, k=k,
        batch_size=batch_size, verbose=verbose,
        mask=mask_a, min_coverage=0.10
    )

    # Score d’anomalie = 1 - SQI (et on vire les NaN)
    keep = np.isfinite(sqi_a)
    s = 1.0 - sqi_a[keep]
    y = y_a[keep]

    fpr, tpr, thr = roc_curve(y, s)
    thr_y = float(thr[int(np.argmax(tpr - fpr))]) if len(thr) else float("inf")
    auc   = float(roc_auc_score(y, s))
    tn, fp, fn, tp = confusion_matrix(y, (s >= thr_y).astype(int), labels=[0,1]).ravel()

    return {"auc": auc, "thr_youden": thr_y, "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp), "n_annotated": int(len(y))}, (wins_a, y, s, info)
