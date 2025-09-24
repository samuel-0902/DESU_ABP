"""
Visualisation & post-traitement des fichiers NPZ :
- reconstruction rapide du signal global par moyenne des fenêtres chevauchantes
- génération de NPZ complets à partir des CSV bruts
- détection et masquage des fenêtres artefactées (plateaux, lignes quasi-parfaites)
"""

import numpy as np
import matplotlib.pyplot as plt
from vae_abp.data_io import build_chunk_table
from vae_abp.windowing import windows_from_table_std
from vae_abp.config import SIGNAL_COL, LISTING_PATH, ADD_AGE_CHANNEL, ADD_STATE_CHANNEL
from vae_abp.utils import ensure_dir, log


def reconstruct_signal(
    X=None,
    M=None,
    SD=None,
    meta=None,
    npz_path=None,
    bad_mask=None,
    is_normalized=True,
    max_points=200_000,
    plot=False,
    save=None,
    show_bad_line=True,
):
    """
    Reconstruit le signal complet en moyennant brute les fenêtres chevauchantes.

    Entrées (deux possibilités) :
        - npz_path : str – chemin vers un .npz contenant X, M, SD, meta
        - ou X, M, SD, meta fournis directement

    Paramètres :
        bad_mask : array booléen (N,) – True pour les fenêtres à marquer en rouge
        is_normalized : bool – True si X est z-scoré (on dénormalise avec M/SD)
        max_points : int – sous-échantillonnage du graphe si trop long
        plot : bool – affiche la figure
        save : str/None – sauvegarde le graphe
        show_bad_line : bool – bande rouge en bas pour les fenêtres bad

    Sorties :
        T : ndarray(datetime64[ns]) – grille temporelle
        Y : ndarray(float) – signal reconstruit (mmHg)
        bad_ratio : ndarray(float) – ratio de fenêtres bad par point
    """
    # chargement depuis fichier si besoin
    if npz_path is not None:
        data = np.load(npz_path, allow_pickle=True)
        X, M, SD = data["X"], data.get("M"), data.get("SD")
        meta = data["meta"]

    assert X is not None and meta is not None, "Fournir npz_path ou X(+meta)"
    N, L, C = X.shape

    # dénormalisation ABP
    if is_normalized:
        assert M is not None and SD is not None, "M et SD requis si normalisé"
        y = X[..., 0] * SD[:, None] + M[:, None]
    else:
        y = X[..., 0]

    # grille temporelle complète
    ts_arr = np.array([m["ts"] for m in meta], dtype="datetime64[ns]")
    te_arr = np.array([m["te"] for m in meta], dtype="datetime64[ns]")

    t0, t1 = ts_arr.min(), te_arr.max()
    dt_ns = int(np.median((te_arr.astype("int64") - ts_arr.astype("int64")) / max(L - 1, 1)))
    dt_ns = max(dt_ns, 1)

    n_pts = int(((t1.astype("int64") - t0.astype("int64")) // dt_ns) + 1)
    stride = max(1, int(np.ceil(n_pts / max_points)))
    grid_ns = np.arange(0, n_pts, stride, dtype=np.int64) * dt_ns + t0.astype("int64")
    T = grid_ns.astype("datetime64[ns]")

    val_sum = np.zeros(T.shape[0], dtype=np.float64)
    val_count = np.zeros(T.shape[0], dtype=np.int32)
    bad_count = np.zeros(T.shape[0], dtype=np.int32)

    # accumulation brute
    for i in range(N):
        ti_ns = np.linspace(ts_arr[i].astype("int64"), te_arr[i].astype("int64"), L, dtype=np.int64)
        yi = y[i].astype(np.float64)

        idx = np.searchsorted(grid_ns, ti_ns, side="left")
        idx = np.clip(idx, 0, len(grid_ns) - 1)

        good = np.isfinite(yi)
        np.add.at(val_sum, idx[good], yi[good])
        np.add.at(val_count, idx[good], 1)
        if bad_mask is not None and bad_mask[i]:
            np.add.at(bad_count, idx, 1)

    ok = val_count > 0
    Y = np.full_like(val_sum, np.nan, dtype=np.float64)
    Y[ok] = val_sum[ok] / val_count[ok]

    bad_ratio = np.zeros_like(val_sum, dtype=np.float64)
    bad_ratio[ok] = bad_count[ok] / val_count[ok]

    # visualisation
    if plot or save:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(T, Y, linewidth=0.8, label="Signal reconstruit")
        ax.set_title("Signal reconstruit (moyenne brute)")
        ax.set_xlabel("temps")
        ax.set_ylabel("ABP (mmHg)")
        ax.grid(True, linewidth=0.4, alpha=0.5)

        if bad_mask is not None and show_bad_line:
            bad_indicator = (bad_ratio > 0).astype(float)
            ymin, ymax = np.nanmin(Y), np.nanmax(Y)
            baseline = ymin - 0.05 * (ymax - ymin)
            height = 0.02 * (ymax - ymin)
            ax.fill_between(
                T, baseline, baseline + height,
                where=bad_indicator > 0,
                color="red", alpha=0.6, label="Bad windows"
            )

        plt.tight_layout()
        if save:
            fig.savefig(save, dpi=150)
        if plot:
            plt.show()

    return T, Y, bad_ratio


def make_npz_from_csvs(patients, out_path):
    """
    Chaîne complète : CSV -> chunk_table -> fenêtrage -> NPZ compressé.

    Entrée :
        patients : list[int] – IDs patients à traiter
        out_path : str – chemin de sauvegarde du .npz

    Sortie :
        out_path : str – chemin du fichier créé
    """
    tbl = build_chunk_table(patients, SIGNAL_COL, LISTING_PATH, verbose=1)
    if tbl.empty:
        raise RuntimeError("Aucune donnée disponible (tbl vide).")

    X, M, SD, meta = windows_from_table_std(
        tbl, add_age_channel=ADD_AGE_CHANNEL, add_state_channel=ADD_STATE_CHANNEL
    )

    np.savez_compressed(out_path, X=X, M=M, SD=SD, meta=np.array(meta, dtype=object))
    log(f"[preproc] saved → {out_path}")

    return out_path


def apply_bad_mask(
    npz_path: str,
    out_path: str | None = None,
    sd_thr: float = 0.2,
    range_thr: float | None = None,
    use_denorm: bool = True,
    r2_thr: float = 0.995,
    resid_std_thr: float = 0.5,
    slope_min: float = 0.2,
    range_min: float = 5.0,
):
    """
    Détecte les fenêtres artefactées (plateaux ou lignes quasi-parfaites) et les remplace par NaN.

    Entrée :
        npz_path : str – fichier d'origine (X, M, SD, meta)
        out_path : str/None – si fourni, sauvegarde le nouveau .npz
        sd_thr, range_thr – seuils pour détecter les plateaux
        use_denorm – travaille sur le signal dénormalisé (mmHg)
        r2_thr, resid_std_thr, slope_min, range_min – seuils pour détecter les lignes trop parfaites

    Sortie :
        X_masked, M, SD, meta, bad_mask – versions filtrées
    """
    data = np.load(npz_path, allow_pickle=True)
    X, M, SD = data["X"].copy(), data["M"].copy(), data["SD"].copy()
    meta = data["meta"] if "meta" in data.files else None
    N, L, C = X.shape

    y_std = X[..., 0]
    y = y_std * SD[:, None] + M[:, None] if use_denorm else y_std

    amp = np.nanmax(y, axis=1) - np.nanmin(y, axis=1)
    if range_thr is None:
        bad_plateau = SD < float(sd_thr)
    else:
        bad_plateau = (SD < float(sd_thr)) & (amp < float(range_thr))

    t = np.linspace(0.0, 1.0, L, dtype=np.float32)
    t_mean = float(t.mean())
    var_t = float(t.var()) + 1e-12
    y_mean = y.mean(axis=1)
    cov_ty = (y @ t) / L - y_mean * t_mean
    slope = cov_ty / var_t
    inter = y_mean - slope * t_mean
    y_hat = slope[:, None] * t[None, :] + inter[:, None]
    resid = y - y_hat
    resid_std = resid.std(axis=1)
    var_y = y.var(axis=1) + 1e-12
    r2 = 1.0 - (resid.var(axis=1) / var_y)

    bad_linear = (
        (r2 >= float(r2_thr))
        & (resid_std <= float(resid_std_thr))
        & ((np.abs(slope) >= float(slope_min)) | (amp >= float(range_min)))
    )

    bad_mask = bad_plateau | bad_linear
    print(f"[apply_bad_mask] {npz_path} → {bad_mask.sum()} suspectes / {N} fenêtres")

    X[bad_mask, :, :] = np.nan

    if out_path is not None:
        np.savez_compressed(out_path, X=X, M=M, SD=SD, meta=meta)
        print(f"[apply_bad_mask] Sauvegardé → {out_path}")

    return X, M, SD, meta, bad_mask
