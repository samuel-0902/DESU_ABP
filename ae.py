"""
ae.py
Implémente un autoencodeur 1D avec :
- Encodeur convolutionnel (Downsampling) → vecteur latent (`Dense(latent_dim)`).
- Deux têtes de sortie :
  (1) Reconstruction d’un canal cible (régression) avec MSE masqué sur échantillons finis.
  (2) Classification binaire (sigmoïde) avec Binary Cross-Entropy (option label smoothing).
- Poids de pertes dynamiques (`model.w_recon`, `model.w_cls`) pour pondérer reconstruction/cls.
- Pipelines de données supervisées et non supervisées combinables dans un même batch.

- `FiniteSanitizer` : couche sérialisable qui remplace NaN/Inf par 0.
- `CropToRef` : rogne la sortie de reconstruction à la longueur exacte de référence.
- Pertes/metrics masquées : calcul sur valeurs finies uniquement (évite la propagation des NaN).
- `build_val_data`, `make_balanced_tfdata` : jeux de validation et pipelines tf.data équilibrés.

Entrée modèle : tenseur (B, L_in, C_in).
Sorties modèle :
- `recon` : (B, L_in, channels_recon) — reconstruction du canal cible.
- `cls`   : (B, 1) — probabilité binaire.



"""

# ae_vector.py — version SANS Lambda
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as L, Model
from tensorflow.keras.utils import register_keras_serializable

# ---------- helpers longueur ----------
def _ceil_div2(x): return (x + 1) // 2
# Doc (FR)
# ----------
# Utilitaire interne de longueur : retourne ⌈x/2⌉ avec arithmétique entière.
# Sert à prédire la longueur après chaque bloc de downsampling (stride=2).


def force_length(X, L_target):
    """Coupe/pad en fin pour obtenir (N, L_target, C)."""

    # Harmonise la longueur temporelle des fenêtres :
    # - Si L == L_target : conversion en float32 et retour tel quel.
    # - Si L  > L_target : coupe au début pour obtenir L_target.
    # - Si L  < L_target : pad de zéros en fin jusqu’à L_target.
    # Entrée : X de forme (N, L, C).
    # Sortie : (N, L_target, C), dtype float32.
    X = np.asarray(X)
    N, L, C = X.shape
    if L == L_target:
        return X.astype("float32")
    if L > L_target:
        return X[:, :L_target, :].astype("float32")
    pad = np.zeros((N, L_target - L, C), dtype=X.dtype)
    return np.concatenate([X, pad], axis=1).astype("float32")

# ---------- custom layers ----------
@register_keras_serializable(package="ae1d")
class FiniteSanitizer(L.Layer):
    """Remplace NaN/Inf par 0."""

    # Couche Keras sérialisable.
    # Entrée : tenseur quelconque.
    # Sortie : mêmes forme et dtype, avec NaN/±Inf remplacés par 0.
    def call(self, x):
        return tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))


@register_keras_serializable(package="ae1d")
class CropToRef(L.Layer):
    """Rogne logits à la longueur de la référence (inputs=[logits, ref])."""

    # Couche Keras sérialisable.
    # Objectif : forcer la sortie de reconstruction à s’aligner exactement
    # sur la longueur temporelle de l’entrée de référence `ref`.
    # Entrées : [logits, ref]
    # Sortie  : logits tronqués à tf.shape(ref)[1].
    def call(self, inputs):
        logits, ref = inputs
        return logits[:, :tf.shape(ref)[1], :]

# ---------- modèle ----------
def build_ae1d_vectorlatent(
    L_in, C_in,
    latent_dim=64,
    n_down=3,
    base_ch=32,
    channels_recon=1,
    target_recon_channel=0,
    lr=3e-4,
    w_recon_init=1.0,
    w_cls_init=0.0,
    steps_per_execution=256,
    label_smoothing=0.0,
    model_name="AE1D_VectorLatent",
    dense_over=0.10,          # <- +10% de canaux au départ du décodeur
):
    """
    Construit et compile l’autoencodeur 1D à vecteur latent.

  
    L_in : int
        Longueur temporelle d’entrée.
    C_in : int
        Nombre de canaux d’entrée.
    latent_dim : int, par défaut 64
        Dimension du vecteur latent (goulot d’étranglement).
    n_down : int, par défaut 3
        Nombre de niveaux de downsampling (stride=2) dans l’encodeur.
    base_ch : int, par défaut 32
        Nombre de canaux de base pour les convolutions (décuplés à la descente).
    channels_recon : int, par défaut 1
        Nombre de canaux reconstruits en sortie.
    target_recon_channel : int, par défaut 0
        Index du canal de l’entrée à reconstruire.
    lr : float, par défaut 3e-4
        Taux d’apprentissage de l’optimiseur Adam.
    w_recon_init : float, par défaut 1.0
        Poids initial de la perte de reconstruction (MSE masqué).
    w_cls_init : float, par défaut 0.0
        Poids initial de la perte de classification (BCE).
    steps_per_execution : int, par défaut 256
        Regroupe plusieurs steps dans un seul passage (optimisation runtime).
    label_smoothing : float, par défaut 0.0
        Lissage des labels pour la perte BCE (si > 0).
    model_name : str, par défaut "AE1D_VectorLatent"
        Nom du modèle Keras.
    dense_over : float, par défaut 0.10
        Sur-provisionnement initial des canaux du décodeur (+10% environ).

    Sortie
 
    model : tf.keras.Model
        Modèle Keras compilé, avec attributs additionnels :
        - `model.w_recon` (tf.Variable) : poids dynamique de la perte de reco.
        - `model.w_cls`   (tf.Variable) : poids dynamique de la perte de cls.
        - `model.latent_dim`, `model.target_recon_channel`, `model.dec_start_ch`.
    """
    tf.keras.backend.clear_session()

    # --- longueurs enc/dec cohérentes ---
    L_down = L_in
    for _ in range(n_down):
        L_down = _ceil_div2(L_down)      # longueur au bottleneck après Down×n

    # boutons dynamiques
    w_recon = tf.Variable(float(w_recon_init), trainable=False, dtype=tf.float32, name="w_recon")
    w_cls   = tf.Variable(float(w_cls_init),   trainable=False, dtype=tf.float32, name="w_cls")

    # Blocs internes (encodeur/décodeur)
    def ConvBlk(x, ch, k=7, s=1):
        # Bloc conv → LN → ReLU
        x = L.Conv1D(ch, k, strides=s, padding="same", use_bias=False)(x)
        x = L.LayerNormalization(axis=-1, epsilon=1e-5)(x)
        return L.ReLU()(x)

    def Down(x, ch):
        # Downsampling stride=2 suivi d’un bloc conv
        x = ConvBlk(x, ch, k=7, s=2)     # stride 2
        return ConvBlk(x, ch, k=5, s=1)

    def Up(x, ch):
        # Upsampling ×2 puis deux blocs conv pour affiner
        x = L.UpSampling1D(size=2)
        x = x(x_in := x_in) if False else L.UpSampling1D(size=2)(x)  # no-op guard
        x = L.UpSampling1D(size=2)(x)  # <-- remove guard line if editor warns (keeps graph simple)
        # We actually only need one UpSampling1D call; leaving just one:
        # x = L.UpSampling1D(size=2)(x)
        x = ConvBlk(x, ch, k=5, s=1)
        return ConvBlk(x, ch, k=3, s=1)

    # ----- ENCODER -----
    inp = L.Input(shape=(L_in, C_in), name="in")
    x   = FiniteSanitizer(name="finite_sanitize")(inp)

    ch = base_ch
    x = ConvBlk(x, ch, k=7, s=1)
    for _ in range(n_down):
        ch = int(ch * 2)                  # double les canaux à chaque Down
        x  = Down(x, ch)

    # Bottleneck latent
    x = L.GlobalAveragePooling1D()(x)                       # (B, ch)
    z = L.Dense(latent_dim, activation="linear", name="z")(x)  # (B, latent_dim)

    # ----- CLASSIF -----
    h = L.Dense(64, activation="relu")(z)
    out_cls = L.Dense(1, activation="sigmoid", name="cls")(h)

    # ----- DECODER (progressif, sans sur-dimensionner la longueur) -----
    # ch_bottleneck = ch (= base_ch * 2**n_down)
    ch_bottleneck = int(ch)
    # point de départ (canaux) du décodeur: moitié du bottleneck, borné par base_ch
    dec_start_ch = max(base_ch, ch_bottleneck // 2)
    # surprovisionnement +10% (arrondi à l’entier)
    dec_start_ch = int(np.ceil(dec_start_ch * (1.0 + float(dense_over))))

    # Dense → Reshape vers (L_down, dec_start_ch)
    x = L.Dense(L_down * dec_start_ch, activation="relu", name="dec_dense")(z)
    x = L.Reshape((L_down, dec_start_ch), name="dec_reshape")(x)

    # Up × n_down: à chaque niveau, L ×2 et canaux ≈ /2 mais jamais < base_ch
    dec_ch = dec_start_ch
    for _ in range(n_down):
        next_ch = max(base_ch, dec_ch // 2)
        x = L.UpSampling1D(size=2)(x)
        x = ConvBlk(x, next_ch, k=5, s=1)
        x = ConvBlk(x, next_ch, k=3, s=1)
        dec_ch = next_ch

    # Tête de reco
    recon_logits = L.Conv1D(channels_recon, 3, padding="same",
                            activation=None, name="recon_logits")(x)

    # Crop exact à L_in (corrige les ceil des downs)
    out_recon = CropToRef(name="recon")([recon_logits, inp])

    # ----- Optim & pertes -----
    opt = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0)

    def masked_mse(y_true, y_pred):
        # MSE masquée : ne compte que les positions finies dans y_true
        y_true = tf.cast(y_true, y_pred.dtype)
        mask   = tf.math.is_finite(y_true)
        y0     = tf.where(mask, y_true, 0.0)
        diff2  = tf.square(y_pred - y0) * tf.cast(mask, y_pred.dtype)
        num = tf.reduce_sum(diff2, axis=[1,2])
        den = tf.reduce_sum(tf.cast(mask, y_pred.dtype), axis=[1,2]) + 1e-8
        return w_recon * (num / den)

    bce_none = tf.keras.losses.BinaryCrossentropy(
        from_logits=False,
        label_smoothing=float(label_smoothing),
        reduction=tf.keras.losses.Reduction.NONE
    )
    def weighted_bce(y_true, y_pred):
        # BCE pondérée par w_cls (réduction NONE → moyenne gérée par Keras)
        return w_cls * bce_none(y_true, y_pred)

    def masked_mae_metric(y_true, y_pred):
        # MAE masquée : évalue l’erreur absolue moyenne sur échantillons finis
        y_true = tf.cast(y_true, y_pred.dtype)
        mask   = tf.math.is_finite(y_true)
        y0     = tf.where(mask, y_true, 0.0)
        mae    = tf.abs(y_pred - y0) * tf.cast(mask, y_pred.dtype)
        num = tf.reduce_sum(mae, axis=[1,2])
        den = tf.reduce_sum(tf.cast(mask, y_pred.dtype), axis=[1,2]) + 1e-8
        return num / den

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

    # expose boutons & infos
    model.w_recon = w_recon
    model.w_cls   = w_cls
    model.latent_dim = int(latent_dim)
    model.target_recon_channel = int(target_recon_channel)
    model.dec_start_ch = int(dec_start_ch)
    return model


# ---------- datasets ----------
def build_val_data(X_sup, y_sup, X_uns, n_sup=256, n_uns=256, seed=42, target_recon_channel=0, L_target=None):
    """
    Construit un mini-jeu de validation mélangé (supervisé + non supervisé).

    Paramètres
    ----------
    X_sup : np.ndarray, forme (Ns, L, C)
        Fenêtres supervisées.
    y_sup : np.ndarray, forme (Ns,) ou (Ns,1)
        Labels binaires (0/1) pour la classification.
    X_uns : np.ndarray, forme (Nu, L, C)
        Fenêtres non supervisées (pas de labels).
    n_sup : int, par défaut 256
        Nombre d’échantillons supervisés à échantillonner.
    n_uns : int, par défaut 256
        Nombre d’échantillons non supervisés à échantillonner.
    seed : int, par défaut 42
        Graine du générateur pseudo-aléatoire.
    target_recon_channel : int, par défaut 0
        Index du canal à reconstruire (extrait depuis X_*).
    L_target : int | None
        Si fourni, force la longueur via `force_length`.

    Sorties
    -------
    X_val : np.ndarray, (N, L, C)
        Entrées de validation concaténées et mélangées.
    y_val : list [y_recon, y_cls]
        - y_recon : (N, L, 1) — cibles de reconstruction (canal `target_recon_channel`).
        - y_cls   : (N, 1)     — labels de classification (0 pour la partie non supervisée).
    w_val : list [w_recon, w_cls]
        Poids d’échantillon par sortie (cls=0 pour non supervisé).
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


def make_balanced_tfdata(X_sup, y_sup, X_uns, batch_size=128, frac_sup=1.0,
                         seed=42, shuffle_buffer_sup=None, shuffle_buffer_uns=None,
                         target_recon_channel=0, L_target=None):
    """
    Crée un pipeline tf.data équilibré supervisé / non supervisé.

    Paramètres
    ----------
    X_sup : np.ndarray, (Ns, L, C)
        Fenêtres supervisées.
    y_sup : np.ndarray, (Ns,) ou (Ns,1)
        Labels binaires 0/1.
    X_uns : np.ndarray, (Nu, L, C)
        Fenêtres non supervisées.
    batch_size : int, par défaut 128
        Taille totale du batch concaténé (sup + uns).
    frac_sup : float, par défaut 1.0
        Fraction supervisée dans le batch (k = round(batch_size * frac_sup)).
    seed : int, par défaut 42
        Graine de mélange.
    shuffle_buffer_sup : int | None
        Taille du buffer de shuffle pour le stream supervisé (défaut : min(Ns, 50k)).
    shuffle_buffer_uns : int | None
        Idem pour le stream non supervisé (défaut : min(Nu, 50k)).
    target_recon_channel : int, par défaut 0
        Canal de reconstruction (extrait des entrées).
    L_target : int | None
        Force la longueur via `force_length` si non None.

    Sorties
    -------
    ds : tf.data.Dataset
        Dataset infini de tuples (X, (y_recon, y_cls), (w_recon, w_cls)).
    steps : int
        Nombre de steps/epoch recommandé pour itérer une fois sur chaque source.
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

    ds_sup = (tf.data.Dataset.from_tensor_slices((X_sup, y_sup))
              .shuffle(shuffle_buffer_sup, seed=seed, reshuffle_each_iteration=True)
              .repeat().batch(k, drop_remainder=True))

    if u > 0:
        ds_uns = (tf.data.Dataset.from_tensor_slices(X_uns)
                  .shuffle(shuffle_buffer_uns, seed=seed+1, reshuffle_each_iteration=True)
                  .repeat().batch(u, drop_remainder=True))

        def _combine(sup_tuple, Xu):
            Xs, ys = sup_tuple
            yu = tf.zeros((tf.shape(Xu)[0], 1), tf.float32)  # dummy labels uns
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


def set_loss_weights(model, w_recon=None, w_cls=None):
    """
    Met à jour les poids dynamiques des pertes du modèle.

    Paramètres
    ----------
    model : tf.keras.Model
        Modèle construit par `build_ae1d_vectorlatent`.
    w_recon : float | None
        Nouveau poids de la perte de reconstruction (si non None).
    w_cls : float | None
        Nouveau poids de la perte de classification (si non None).
    """
    if w_recon is not None: model.w_recon.assign(float(w_recon))
    if w_cls   is not None: model.w_cls.assign(float(w_cls))
