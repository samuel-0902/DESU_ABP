
"""
train.py

Rôle
----
- Parse les arguments CLI (hyperparamètres, chemins).
- Charge le dataset NPZ (supervisé / non supervisé).
- Force la longueur cible des fenêtres.
- Instancie le modèle AE1D (importé depuis `ae`).
- Construit les DataLoaders (tf.data) mixtes sup/uns.
- Configure les callbacks (checkpoint, early stop, TensorBoard, CSV).
- Lance l'entraînement puis sauvegarde le meilleur modèle.
"""

import os
import gc
import argparse
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
from tensorflow.keras import backend as K
from tensorflow.keras import mixed_precision

# GPU : récision mixte 
mixed_precision.set_global_policy('mixed_float16')

# Module modèle 
import ae as A


def main():
    """
    Point d'entrée entraînement.

    Étapes :
    1) Arguments CLI et options runtime (eager/XLA).
    2) Préparation des répertoires (models/, logs/).
    3) Chargement et préformatage des données (force_length).
    4) Construction du modèle AE1D avec hyperparamètres.
    5) Création des DataLoaders (train tf.data + val numpy).
    6) Configuration des callbacks (CKPT, ES, TB, CSV).
    7) Entraînement, chargement des meilleurs poids, sauvegarde finale.
    """
    # ------------------------------- ARGS -------------------------------
    p = argparse.ArgumentParser(description="Train AE1D (ae) + TensorBoard")
    p.add_argument("--latent", type=int, default=int(os.getenv("LAT")) if os.getenv("LAT") else 64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--frac_sup", type=float, default=0.25)
    p.add_argument("--l_target", type=int, default=575)
    p.add_argument("--data", type=str, default="/workspace/venv/vae_final/final_dataset.npz")
    p.add_argument("--base_dir", type=str, default="/workspace/venv/vae_final")
    p.add_argument("--models_subdir", type=str, default="model")
    p.add_argument("--logs_under_models", action="store_true")
    p.add_argument("--steps_cap", type=int, default=1000)
    p.add_argument("--patience", type=int, default=5)

    # options modèle
    p.add_argument("--n_down", type=int, default=3)
    p.add_argument("--base_ch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--w_recon_init", type=float, default=1.0)
    p.add_argument("--w_cls_init", type=float, default=1.0)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--steps_per_execution", type=int, default=512)

    # options runtime
    p.add_argument("--eager", action="store_true", help="exécute en eager mode")
    p.add_argument("--xla", action="store_true", help="active XLA JIT")

    args = p.parse_args()

    # -------------------------- RUNTIME OPTS ---------------------------
    # Rôle : configurer l'exécution (eager vs graph, JIT XLA)
    if args.eager:
        tf.config.run_functions_eagerly(True)
        args.steps_per_execution = 1
    else:
        tf.config.run_functions_eagerly(False)

    if args.xla:
        try:
            tf.config.optimizer.set_jit(True)
        except Exception:
            pass

    # -------------------------- REPERTOIRES ----------------------------
    # Rôle : organiser modèles et logs
    base_dir   = args.base_dir
    models_dir = os.path.join(base_dir, args.models_subdir)
    logs_root  = (os.path.join(models_dir, "logs") if args.logs_under_models
                  else os.path.join(base_dir, "logs"))
    os.makedirs(models_dir, exist_ok=True)

    # ------------------------------ DATA -------------------------------
    # Rôle : charger NPZ et normaliser la longueur
    ds = dict(np.load(args.data, allow_pickle=True))
    L_TARGET = int(args.l_target)
    Xsup = ds["X_supervised_train"]; ysup = ds["Y_supervised_train"]
    Xuns = ds["X_unsupervised_train"]

    Xsup = A.force_length(Xsup, L_TARGET)
    Xuns = A.force_length(Xuns, L_TARGET)
    L_in, C_in = Xsup.shape[1:3]

    # ----------------------- RUN NAME / LOGS ---------------------------
    # Rôle : identifiant unique d'entraînement + répertoire TensorBoard
    run_name = ("NEW"
        f"AE1D_lat{args.latent}_bs{args.batch_size}_frac{int(args.frac_sup*100)}_"
        f"down{args.n_down}_base{args.base_ch}_spe{args.steps_per_execution}"
        f"{'_eager' if args.eager else ''}{'_xla' if args.xla else ''}_"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    log_dir = os.path.join(logs_root, run_name)
    os.makedirs(log_dir, exist_ok=True)

    # ------------------------------ MODEL ------------------------------
    # Rôle : construire et compiler AE1D
    model = A.build_ae1d_vectorlatent(
        L_in, C_in,
        latent_dim=args.latent,
        n_down=args.n_down,
        base_ch=args.base_ch,
        channels_recon=1,
        target_recon_channel=0,
        lr=args.lr,
        w_recon_init=args.w_recon_init,
        w_cls_init=args.w_cls_init,
        label_smoothing=args.label_smoothing,
        steps_per_execution=args.steps_per_execution,
        model_name=run_name
    )
    model._name = run_name

    # --------------------------- DATA LOADERS --------------------------
    # Rôle : val (numpy) + train (tf.data) supervisé/non supervisé
    val_data = A.build_val_data(
        Xsup, ysup, Xuns,
        n_sup=256, n_uns=256,
        target_recon_channel=0, L_target=L_TARGET
    )

    train_ds, steps = A.make_balanced_tfdata(
        Xsup, ysup, Xuns,
        batch_size=args.batch_size,
        frac_sup=args.frac_sup,
        target_recon_channel=0,
        L_target=L_TARGET
    )

    # tf.data non déterministe (débit)
    opts = tf.data.Options()
    opts.experimental_deterministic = False
    train_ds = train_ds.with_options(opts)

    # ----------------------------- CALLBACKS ---------------------------
    # Rôle : checkpoint meilleur val_loss, early stop, logs TB, CSV
    ckpt_best = ModelCheckpoint(
        filepath=os.path.join(models_dir, f"best_{run_name}.weights.h5"),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    earlystop = EarlyStopping(
        monitor="val_loss",
        patience=int(args.patience),
        restore_best_weights=True,
        verbose=1
    )
    tboard = TensorBoard(
        log_dir=log_dir,
        histogram_freq=0,
        write_graph=False,
        update_freq="epoch",
        profile_batch=0
    )
    csv = CSVLogger(os.path.join(models_dir, f"{run_name}_metrics.csv"))
    callbacks = [ckpt_best, earlystop, tboard, csv]

    # ------------------------------- FIT -------------------------------
    # Rôle : entraînement + sauvegarde meilleur modèle
    history = model.fit(
        train_ds,
        steps_per_epoch=min(steps, args.steps_cap),
        epochs=args.epochs,
        validation_data=val_data,
        verbose=1,
        callbacks=callbacks
    )
    model.load_weights(os.path.join(models_dir, f"best_{run_name}.weights.h5"))
    model.save(os.path.join(models_dir, f"best_full_{run_name}.keras"))

    # ------------------------------ CLEANUP ----------------------------
    del history, model, train_ds, val_data
    gc.collect()
    K.clear_session()


if __name__ == "__main__":

    try:
        gpus = tf.config.list_physical_devices("GPU")
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    main()
