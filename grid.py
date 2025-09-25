
"""
Grid AUC sur annotées — évalue plusieurs modèles et combinaisons (alpha_min, alpha_max,k),
exporte ROC/AUC globales et par patient (CSV/JSON), sans modifier la logique d'inférence.
"""

import os, json, itertools, datetime
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import importlib
import sqi_auc as sqi
import ae as A 

#==============CONFIG ==============
BASE = "/workspace/venv/vae_final"

MODELS = [
    "/workspace/venv/vae_final/model_new/run_lat1/best_full_NEWAE1D_lat1_bs512_frac25_down3_base16_spe512_20250925-123437.keras",
    "/workspace/venv/vae_final/model_new/run_lat2/best_full_NEWAE1D_lat2_bs512_frac25_down3_base16_spe512_20250925-124543.keras",
    "/workspace/venv/vae_final/model_new/run_lat3/best_full_NEWAE1D_lat3_bs512_frac25_down3_base16_spe512_20250925-125642.keras",
    "/workspace/venv/vae_final/model_new/run_lat4/best_full_NEWAE1D_lat4_bs512_frac25_down3_base16_spe512_20250925-130756.keras",
    "/workspace/venv/vae_final/model_new/run_lat8/best_full_NEWAE1D_lat8_bs512_frac25_down3_base16_spe512_20250925-120158.keras",
    "/workspace/venv/vae_final/model_new/run_lat16/best_full_NEWAE1D_lat16_bs512_frac25_down3_base16_spe512_20250925-121230.keras",
    "/workspace/venv/vae_final/model_new/run_lat32/best_full_NEWAE1D_lat32_bs512_frac25_down3_base16_spe512_20250925-122331.keras",
    "/workspace/venv/vae_final/model_new/run_lat64/best_full_NEWAE1D_lat64_bs512_frac25_down3_base16_spe512_20250925-131912.keras",
    "/workspace/venv/vae_final/model_new/run_lat128/best_full_NEWAE1D_lat128_bs512_frac25_down3_base16_spe512_20250925-132945.keras",
]

NPZ_LABELS = f"{BASE}/data_label.npz"
NPZ_PATIENTS = [
    f"{BASE}/file_npz/patients_100_clean.npz",
    f"{BASE}/file_npz/patients_50_clean.npz",
]

FS = 115.0
BATCH_SIZE = 256
TARGET_RECON_CH = 0
VERBOSE = 0

ALPHA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
K_VALUES     = [5.0, 10.0]

RUN_TAG  = "p100_p50_global"
#====================================

def make_run_dir(tag):
    """
    Rôle : crée un répertoire horodaté sous BASE/experiments pour stocker les exports.
    """
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(BASE, "experiments", f"run_{tag}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def collect_y_and_score_for_patients(model, patient_paths, labels_npz,
                                     *, fs, target_recon_channel,
                                     alpha_min, alpha_max, k,
                                     batch_size=256, verbose=0):
    """
    Rôle : reproduit la logique AUC-annotées pour retourner (y, s) par patient et global.
    - Fenêtrage + adaptation au modèle.
    - Filtrage sur fenêtres annotées.
    - RMSE masquée, normalisation (10/90), score s = α(p)*p_abn + (1-α)*rmse_norm.
    - Retourne aussi FPR/TPR/THR par entité pour export ROC.
    """
    per_patient = []
    y_all_list, s_all_list = [], []

    for pth in patient_paths:
        d = np.load(pth, allow_pickle=True)
        X_any = d["X"].astype("float32")
        meta  = list(d["meta"])

        # 1) fenêtres + adapt
        wins, mask_win, starts, L_win, L_total, t0 = sqi.windows_from_meta_any(X_any, meta, fs)
        wins, mask_win = sqi.adapt_for_model(wins, mask_win, model)

        # 2) annotées uniquement
        wins_a, y_a, kept_idx = sqi.filter_to_annotated(wins, meta, labels_npz)
        if len(y_a) == 0:
            per_patient.append({
                "patient": pth, "n_annotated": 0, "auc": float("nan"),
                "thr_youden": float("nan"), "TN": 0, "FP": 0, "FN": 0, "TP": 0,
                "y": np.array([], dtype=int), "s": np.array([], dtype=float)
            })
            continue
        mask_a = mask_win[kept_idx]

        # 3) prédictions
        recon_pred, p_abn = sqi._pick_outputs(model, wins_a, batch_size=batch_size, verbose=verbose)
        if recon_pred.shape[-1] > 1:
            recon_pred = recon_pred[..., target_recon_channel:target_recon_channel+1]
        y_true_recon = wins_a[..., target_recon_channel:target_recon_channel+1]

        # 4) RMSE masquée + filtrage coverage
        rmse = sqi._rmse_per_window_masked(y_true_recon, recon_pred, mask_a, min_coverage=0.10)
        keep = np.isfinite(rmse)
        if not np.any(keep):
            per_patient.append({
                "patient": pth, "n_annotated": 0, "auc": float("nan"),
                "thr_youden": float("nan"), "TN": 0, "FP": 0, "FN": 0, "TP": 0,
                "y": np.array([], dtype=int), "s": np.array([], dtype=float)
            })
            continue

        rmse = rmse[keep].astype(np.float32)
        y_a  = y_a[keep].astype(int)
        p_abn = p_abn[keep].astype(np.float32)

        # 5) normalisation RMSE (10/90)
        lo = np.nanpercentile(rmse, 10.0)
        hi = np.nanpercentile(rmse, 90.0)
        if not np.isfinite(lo): lo = np.nanmin(rmse)
        if not np.isfinite(hi): hi = np.nanmax(rmse)
        if hi <= lo: hi = lo + 1e-8
        rmse_norm = np.clip((rmse - lo) / (hi - lo), 0.0, 1.0)

        # 6) alpha dynamique + score
        alpha = sqi._alpha_cls(p_abn, alpha_min=alpha_min, alpha_max=alpha_max, k=k)
        s = alpha * p_abn + (1.0 - alpha) * rmse_norm
        ok = np.isfinite(s)
        y = y_a[ok]; s = s[ok]

        if y.size >= 2 and len(np.unique(y)) == 2:
            fpr_p, tpr_p, thr_p = roc_curve(y, s)
            youden_p = int(np.argmax(tpr_p - fpr_p))
            thr_y = float(thr_p[youden_p]) if youden_p < len(thr_p) else float("inf")
            tn, fp, fn, tp = confusion_matrix(y, (s >= thr_y).astype(int), labels=[0,1]).ravel()
            auc_p = float(roc_auc_score(y, s))
        else:
            fpr_p, tpr_p, thr_p, thr_y, tn, fp, fn, tp, auc_p = (np.array([0.,1.]), np.array([0.,1.]),
                                                                  np.array([np.nan]), float("nan"),
                                                                  0,0,0,0, float("nan"))

        per_patient.append({
            "patient": pth,
            "n_annotated": int(len(y)),
            "auc": auc_p,
            "thr_youden": thr_y,
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
            "y": y, "s": s,
            "roc": (fpr_p, tpr_p, thr_p),
        })

        y_all_list.append(y); s_all_list.append(s)

    # global concat
    if len(y_all_list) == 0:
        raise RuntimeError("Aucune fenêtre valide pour la ROC globale.")
    y_all = np.concatenate(y_all_list); s_all = np.concatenate(s_all_list)
    fpr, tpr, thr = roc_curve(y_all, s_all)
    youden = int(np.argmax(tpr - fpr))
    thr_y  = float(thr[youden]) if youden < len(thr) else float("inf")
    auc_g  = float(roc_auc_score(y_all, s_all))
    tn, fp, fn, tp = confusion_matrix(y_all, (s_all >= thr_y).astype(int), labels=[0,1]).ravel()

    global_rep = {
        "auc": auc_g,
        "thr_youden": thr_y,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "n_annotated": int(len(y_all)),
        "n_patients_used": int(len([p for p in per_patient if p["n_annotated"] > 0])),
        "roc": (fpr, tpr, thr),
        "y": y_all, "s": s_all,
    }
    return global_rep, per_patient

def main():
    """
    Rôle : boucle d’évaluation multi-modèles et multi-hyperparamètres.
    - Charge labels et crée un répertoire d’exports.
    - Pour chaque modèle, pour chaque (alpha_min, alpha_max, k) :
      * calcule métriques globales et par patient,
      * imprime un résumé,
      * exporte ROC (CSV), métriques (JSON/CSV) et un master CSV par modèle.
    """
    importlib.reload(sqi)
    labels = np.load(NPZ_LABELS, allow_pickle=True)
    run_dir = make_run_dir(RUN_TAG)

    #Sauvegarde une copie de la config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({
            "MODELS": [os.path.basename(m) for m in MODELS],
            "NPZ_PATIENTS": NPZ_PATIENTS,
            "FS": FS,
            "BATCH_SIZE": BATCH_SIZE,
            "TARGET_RECON_CH": TARGET_RECON_CH,
            "ALPHA_VALUES": ALPHA_VALUES,
            "K_VALUES": K_VALUES,
        }, f, indent=2)

    alpha_pairs = [(a_min, a_max) for a_min in ALPHA_VALUES for a_max in ALPHA_VALUES if a_min <= a_max]

    for model_path in MODELS:
        name = os.path.basename(model_path)
        try:
            model = keras.models.load_model(
                model_path, compile=False,
                custom_objects={"FiniteSanitizer": __import__("ae").FiniteSanitizer,
                                "CropToRef": __import__("ae").CropToRef}
            )
        except Exception as e:
            print(f"[MODEL LOAD ERROR] {name}: {e}")
            continue

        model_dir = os.path.join(run_dir, name.replace(".keras",""))
        os.makedirs(model_dir, exist_ok=True)

        # master CSV pour ce moèle
        master_rows = []

        for (a_min, a_max) in alpha_pairs:
            for k in K_VALUES:
                print(f"\n[MODEL] {name} | amin={a_min:.2f} amax={a_max:.2f} k={k:.1f}")

                try:
                    global_rep, per_pat = collect_y_and_score_for_patients(
                        model, NPZ_PATIENTS, labels,
                        fs=FS, target_recon_channel=TARGET_RECON_CH,
                        alpha_min=a_min, alpha_max=a_max, k=k,
                        batch_size=BATCH_SIZE, verbose=VERBOSE
                    )
                except Exception as e:
                    print(f"[ERROR] {name} combo (a_min={a_min}, a_max={a_max}, k={k}): {e}")
                    continue

                # --- Affichge (global) ---
                print("=== GLOBAL (annotées uniquement) ===")
                to_print = {k0:v for k0,v in global_rep.items() if k0 not in ("roc","y","s")}
                print(to_print)

                print("\n--- Par patient ---")
                for r in per_pat:
                    rp = {k0:v for k0,v in r.items() if k0 not in ("roc","y","s")}
                    print(rp)

                # ---Export ---
                combo_tag = f"amin{a_min}_amax{a_max}_k{k}".replace(".","p")
                combo_dir = os.path.join(model_dir, combo_tag)
                os.makedirs(combo_dir, exist_ok=True)

                # a) ROC global CSV
                fpr, tpr, thr = global_rep["roc"]
                pd.DataFrame({"fpr": fpr, "tpr": tpr, "thr": thr}).to_csv(
                    os.path.join(combo_dir, "roc_global.csv"), index=False
                )

                # b) ROC par patient CSV
                for r in per_pat:
                    fpr_p, tpr_p, thr_p = r["roc"]
                    tag = os.path.basename(r["patient"]).replace(".npz","")
                    pd.DataFrame({"fpr": fpr_p, "tpr": tpr_p, "thr": thr_p}).to_csv(
                        os.path.join(combo_dir, f"roc_{tag}.csv"), index=False
                    )

                # c) Metrics globaux JSON
                with open(os.path.join(combo_dir, "global_metrics.json"), "w") as f:
                    json.dump(to_print, f, indent=2)

                # d) Metrics par patient CSV
                rows_pat = []
                for r in per_pat:
                    tag = os.path.basename(r["patient"]).replace(".npz","")
                    rows_pat.append({
                        "patient": tag,
                        "n_annotated": r["n_annotated"],
                        "auc": r["auc"],
                        "thr_youden": r["thr_youden"],
                        "TN": r["TN"], "FP": r["FP"], "FN": r["FN"], "TP": r["TP"]
                    })
                pd.DataFrame(rows_pat).to_csv(os.path.join(combo_dir, "per_patient_metrics.csv"), index=False)

                # e) Master row 
                master_rows.append({
                    "model": name,
                    "alpha_min": a_min, "alpha_max": a_max, "k": k,
                    **to_print,
                    "combo_dir": combo_dir
                })

        # Écrit le  CSV du modèle
        if master_rows:
            pd.DataFrame(master_rows).to_csv(os.path.join(model_dir, "results_master.csv"), index=False)

    print(f"\nDone. Exports dans: {run_dir}")

if __name__ == "__main__":
    main()
