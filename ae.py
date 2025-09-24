
# Module de traitement de signaux 1D avec encodage/décodage
# Permet de compresser une séquence en vecteur latent puis
# de reconstruire le signal d'entrée sur un canal cible.
#
# Entrées :
#   - X : tenseur de forme (N, L, C)
#       N = nombre d'exemples
#       L = longueur temporelle (nombre d'échantillons)
#       C = nombre de canaux (signaux parallèles)
#
# Sorties :
#   - reconstruction : (N, L, 1) → canal reconstruit
#   - sortie binaire : (N, 1)    → score associé au vecteur latent
#
# Définition des canaux (C=4) :
#   - Canal 0 = ABP normalisé
#       Signal de pression artérielle (ABP), normalisé.
#   - Canal 1 = Âge normalisé
#       Valeur constante par fenêtre, répétée sur L, normalisée.
#   - Canal 2 = State (état per-opératoire)
#       Variable discrète (CEC, clamp, etc.), forward-fill.
#   - Canal 3 = Moyenne cumulative normalisée
#       Moyenne glissante/cumulative du signal ABP, normalisée.
#
# Les hyperparamètres principaux :
#   - latent_dim : taille du vecteur latent
#   - n_down : nombre de réductions par 2 de la longueur
#   - base_ch : largeur initiale des blocs convolutifs
#   - channels_recon : nb de canaux à reconstruire (ex: 1)
#   - target_recon_channel : indice du canal d'entrée à reconstruire
#   - lr : taux d’optimisation
#   - w_recon_init / w_cls_init : poids initiaux des deux fonctions de coût
#   - steps_per_execution : fréquence des exécutions compilées
#   - label_smoothing : adoucissement optionnel des étiquettes
#
# ================================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as L, Model
from tensorflow.keras.utils import register_keras_serializable


# ---------- Helpers ----------
def _ceil_div2(x):
    """Division entière par 2 avec arrondi au supérieur."""
    return (x + 1) // 2


def force_length(X, L_target):
    """
    Ajuste la longueur temporelle d'une séquence.
    - Coupe si L > L_target
    - Complète par des zéros si L < L_target

    Entrée :
        X : (N, L, C)
    Sortie :
        X_pad : (N, L_target, C)
    """
    X = np.asarray(X)
    N, L, C = X.shape
    if L == L_target:
        return X.astype("float32")
    if L > L_target:
        return X[:, :L_target, :].astype("float32")
    pad = np.zeros((N, L_target - L, C), dtype=X.dtype)
    return np.concatenate([X, pad], axis=1).astype("float32")


# ---------- Couches custom ----------
@register_keras_serializable(package="ae1d")
class FiniteSanitizer(L.Layer):
    """Remplace NaN/Inf par 0 dans le signal."""
    def call(self, x):
        return tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))


@register_keras_serializable(package="ae1d")
class CropToRef(L.Layer):
    """
    Tronque un tenseur à la longueur d'une référence.
    Entrée : [logits, ref]
        - logits : (N, L', C)
        - ref    : (N, L,  C)
    Sortie : (N, L, C) avec L'≥L
    """
    def call(self, inputs):
        logits, ref = inputs
        return logits[:, :tf.shape(ref)[1], :]


# ---------- Construction du modèle ----------
def build_ae1d_vectorlatent(
    L_in, C_in,
    latent_dim=64,              # dimension du vecteur latent
    n_down=3,                   # nombre de réductions successives (x2)
    base_ch=32,                 # largeur initiale des blocs convolutifs
    channels_recon=1,           # nb de canaux à reconstruire
    target_recon_channel=0,     # indice du canal cible
    lr=3e-4,                    # taux d’optimisation
    w_recon_init=1.0,           # poids initial de la reconstruction
    w_cls_init=0.0,             # poids initial du score latent
    steps_per_execution=256,    # fréquence d’exécution compilée
    label_smoothing=0.0,
    model_name="AE1D_VectorLatent"
):
    """Construit un modèle encodeur/décodeur basé sur des blocs 1D."""

    tf.keras.backend.clear_session()

    # Longueurs après compressions/upsamplings
    L_down = L_in
    for _ in range(n_down):
        L_down = _ceil_div2(L_down)
    L_up = L_down * (2 ** n_down)

    # Variables scalaires réglables pendant l’entraînement
    w_recon = tf.Variable(float(w_recon_init), trainable=False, dtype=tf.float32, name="w_recon")
    w_cls   = tf.Variable(float(w_cls_init),   trainable=False, dtype=tf.float32, name="w_cls")

    # -------- blocs de base --------
    def ConvBlk(x, ch, k=7, s=1):
        """Bloc conv1D + normalisation + ReLU."""
        x = L.Conv1D(ch, k, strides=s, padding="same", use_bias=False)(x)
        x = L.LayerNormalization(axis=-1, epsilon=1e-5)(x)
        return L.ReLU()(x)

    def Down(x, ch):
        """Bloc de réduction de dimension (x2)."""
        x = ConvBlk(x, ch, k=7, s=2)
        return ConvBlk(x, ch, k=5, s=1)

    def Up(x, ch):
        """Bloc de sur-échantillonnage (x2)."""
        x = L.UpSampling1D(size=2)(x)
        x = ConvBlk(x, ch, k=5, s=1)
        return ConvBlk(x, ch, k=3, s=1)

    # -------- ENCODEUR --------
    # Entrée : tenseur (N, L_in, C_in)
    # C_in doit être = 4 avec la convention des canaux :
    #   0 : ABP normalisé
    #   1 : Âge normalisé
    #   2 : State (état per-opératoire)
    #   3 : Moyenne cumulative normalisée
    inp = L.Input(shape=(L_in, C_in), name="in")
    x   = FiniteSanitizer(name="finite_sanitize")(inp)

    ch = base_ch
    x = ConvBlk(x, ch, k=7, s=1)
    for _ in range(n_down):
        ch = int(ch * 2)
        x  = Down(x, ch)

    # Projection en vecteur latent
    x = L.GlobalAveragePooling1D()(x)                 # (N, ch)
    z = L.Dense(latent_dim, activation="linear", name="z")(x)  # (N, latent_dim)

    # -------- BRANCHE SCORE --------
    h = L.Dense(64, activation="relu")(z)
    out_cls = L.Dense(1, activation="sigmoid", name="cls")(h)

    # -------- DÉCODEUR --------
    dec_ch0 = max(base_ch, ch // 2)
    x = L.Dense(L_up * dec_ch0, activation="relu")(z)
    x = L.Reshape((L_up, dec_ch0))(x)

    dec_ch = dec_ch0
    for _ in range(n_down):
        dec_ch = max(base_ch, dec_ch // 2)
        x = Up(x, dec_ch)

    recon_logits = L.Conv1D(channels_recon, 3, padding="same", activation=None, name="recon_logits")(x)

    # Recadrage exact à la longueur d'entrée
    out_recon = CropToRef(name="recon")([recon_logits, inp])

    # -------- Optimisation --------
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)

    # -------- Fonctions de coût --------
    def masked_mse(y_true, y_pred):
        """
        Erreur quadratique moyenne avec masque (ignorer NaN).
        y_true : (N, L, 1)
        y_pred : (N, L, 1)
        """
        y_true = tf.cast(y_true, y_pred.dtype)
        mask   = tf.math.is_finite(y_true)
        y0     = tf.where(mask, y_true, 0.0)
        diff2  = tf.square(y_pred - y0) * tf.cast(mask, y_pred.dtype)
        num = tf.reduce_sum(diff2, axis=[1,2])
        den = tf.reduce_sum(tf.cast(mask, y_pred.dtype), axis=[1,2]) + 1e-8
        per_sample = num / den
        return w_recon * per_sample

    bce_none = tf.keras.losses.BinaryCrossentropy(
        from_logits=False,
        label_smoothing=float(label_smoothing),
        reduction=tf.keras.losses.Reduction.NONE
    )
    def weighted_bce(y_true, y_pred):
        """Erreur de classification binaire pondérée."""
        per_sample = bce_none(y_true, y_pred)  # (N,)
        return w_cls * per_sample

    # -------- Métriques --------
    def masked_mae_metric(y_true, y_pred):
        """Erreur absolue moyenne avec masque (ignorer NaN)."""
        y_true = tf.cast(y_true, y_pred.dtype)
        mask   = tf.math.is_finite(y_true)
        y0     = tf.where(mask, y_true, 0.0)
        mae    = tf.abs(y_pred - y0) * tf.cast(mask, y_pred.dtype)
        num = tf.reduce_sum(mae, axis=[1,2])
        den = tf.reduce_sum(tf.cast(mask, y_pred.dtype), axis=[1,2]) + 1e-8
        return num / den

    # -------- Compilation --------
    model = Model(inp, [out_recon, out_cls], name=model_name)
    model.compile(
        optimizer=opt,
        loss=[masked_mse, weighted_bce],
        metrics=[[masked_mae_metric],
                 [tf.keras.metrics.AUC(name="auc"),
                  tf.keras.metrics.BinaryAccuracy(name="acc")]],
        steps_per_execution=int(steps_per_execution),
        run_eagerly=False
    )

    # Exposer quelques attributs utiles
    model.w_recon = w_recon
    model.w_cls   = w_cls
    model.latent_dim = int(latent_dim)
    model.target_recon_channel = int(target_recon_channel)
    return model


# ---------- Génération de données de validation ----------
def build_val_data(X_sup, y_sup, X_uns, n_sup=256, n_uns=256,
                   seed=42, target_recon_channel=0, L_target=None):
    """
    Construit un jeu de validation mélangé (supervisé + non supervisé).
    - X_sup : données étiquetées (N_sup, L, C)
    - y_sup : étiquettes associées
    - X_uns : données non étiquetées
    """
    rng = np.random.default_rng(seed)
    if L_target is not None:
        X_sup = force_length(X_sup, L_target)
        X_uns = force_length(X_uns, L_target)

    sup_idx = rng.choice(len(X_sup), size=min(n_sup, len(X_sup)), replace=False)
    uns_idx = rng.choice(len(X_uns), size=min(n_uns, len(X_uns)), replace=False)

    Xs = X_sup[sup_idx].astype("float32")
    ys = np.asarray(y_sup[sup_idx], "float32").reshape(-1,1)
    Xu = X_uns[uns_idx].astype("float32")

    X_val = np.concatenate([Xs, Xu], axis=0).astype("float32")
    ch = int(target_recon_channel)
    y_recon = X_val[..., ch:ch+1]  # (N,L,1)
    y_cls   = np.concatenate([ys, np.zeros((len(Xu),1), "float32")], axis=0)

    w_recon = np.ones((len(X_val),), "float32")
    w_cls   = np.concatenate([np.ones((len(Xs),), "float32"),
                              np.zeros((len(Xu),), "float32")], axis=0)

    p = rng.permutation(len(X_val))
    return X_val[p], [y_recon[p], y_cls[p]], [w_recon[p], w_cls[p]]


# ---------- Génération de datasets équilibrés ----------
def make_balanced_tfdata(X_sup, y_sup, X_uns,
                         batch_size=128, frac_sup=1.0,
                         seed=42, shuffle_buffer_sup=None,
                         shuffle_buffer_uns=None,
                         target_recon_channel=0, L_target=None):
    """
    Construit un tf.data.Dataset équilibré entre données supervisées et non supervisées.
    Permet de contrôler la proportion de chaque type dans un batch.
    """
    if L_target is not None:
        X_sup = force_length(X_sup, L_target)
        X_uns = force_length(X_uns, L_target)

    X_sup = np.asarray(X_sup, "float32")
    y_sup = np.asarray(y_sup, "float32").reshape(-1, 1)
    X_uns = np.asarray(X_uns, "float32")

    Ns, Nu = len(X_sup), len(X_uns)
    bs = int(batch_size)
    k  = max(1, min(bs, int(round(bs*float(frac_sup)))))  # nb sup par batch
    u  = bs - k
    ch = int(target_recon_channel)

    if shuffle_buffer_sup is None: shuffle_buffer_sup = min(Ns, 50_000)
    if shuffle_buffer_uns is None: shuffle_buffer_uns = min(Nu, 50_000)

    # Dataset supervisé
    ds_sup = (tf.data.Dataset.from_tensor_slices((X_sup, y_sup))
              .shuffle(shuffle_buffer_sup, seed=seed, reshuffle_each_iteration=True)
              .repeat().batch(k, drop_remainder=True))

    if u > 0:
        # Dataset non supervisé
        ds_uns = (tf.data.Dataset.from_tensor_slices(X_uns)
                  .shuffle(shuffle_buffer_uns, seed=seed+1, reshuffle_each_iteration=True)
                  .repeat().batch(u, drop_remainder=True))

        # Fusion des deux
        def _combine(sup_tuple, Xu):
            Xs, ys = sup_tuple
            yu = tf.zeros((tf.shape(Xu)[0], 1), tf.float32)  # étiquettes nulles
            X  = tf.concat([Xs, Xu], axis=0)
            y_recon = X[..., ch:ch+1]
            y_cls   = tf.concat([ys, yu], axis=0)

            w_recon = tf.ones((tf.shape(X)[0],), tf.float32)
            w_cls   = tf.concat([tf.ones((tf.shape(ys)[0],), tf.float32),
                                 tf.zeros((tf.shape(yu)[0],), tf.float32)], axis=0)

            idx = tf.random.shuffle(tf.range(tf.shape(X)[0], dtype=tf.int32), seed=seed)
            X, y_recon, y_cls = (tf.gather(t, idx, axis=0) for t in (X, y_recon, y_cls))
            w_recon, w_cls    = (tf.gather(t, idx, axis=0) for t in (w_recon, w_cls))
            return X, (y_recon, y_cls), (w_recon, w_cls)

        ds = (tf.data.Dataset.zip((ds_sup, ds_uns))
              .map(_combine, num_parallel_calls=tf.data.AUTOTUNE)
              .prefetch(tf.data.AUTOTUNE))
        steps = max(int(np.ceil(Ns / k)), int(np.ceil(Nu / u)))
        return ds, steps
    else:
        def _sup_only(Xs, ys):
            y_recon = Xs[..., ch:ch+1]
            y_cls   = ys
            w_recon = tf.ones((tf.shape(Xs)[0],), tf.float32)
            w_cls   = tf.ones((tf.shape(ys)[0],), tf.float32)
            return Xs, (y_recon, y_cls), (w_recon, w_cls)

        ds = (ds_sup.map(_sup_only, num_parallel_calls=tf.data.AUTOTUNE)
              .prefetch(tf.data.AUTOTUNE))
        steps = int(np.ceil(Ns / k))
        return ds, steps


# ---------- Ajustement des poids dynamiques ----------
def set_loss_weights(model, w_recon=None, w_cls=None):
    """Permet d’ajuster dynamiquement les poids des deux fonctions de coût."""
    if w_recon is not None: model.w_recon.assign(float(w_recon))
    if w_cls   is not None: model.w_cls.assign(float(w_cls))
