
# Outil d’annotation manuelle de signaux de pression artérielle (ABP) dans un notebook.
# - Navigation temporelle fluide (fenêtre réglable, déplacement par pas/fenêtre, centrage HH:MM:SS).
# - Sélection temporelle précise (pas au choix, overlay en direct).
# - Annotation par classe principale (normal, pas_normal, mauvaise_qualite) et type d’artefact.
# - Détection/affichage contextuel de périodes CEC/Clampage si colonnes disponibles.
# - Visualisation interactive avec surcouches (sélection en cours, annotations déjà posées).
# - Export final des annotations via AnnotHandle().dataframe().


import datetime as dt
import time as _time
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import ipywidgets as w
from IPython.display import display, clear_output

# --------- Config ---------
ABP_COLS = ["abp_filtered", "abp[mmHg]", "abp", "ABP", "abp_mmHg"]
ARTIFACT_TYPES = ["artefact_indetermine", "artefact_purge", "artefact_prelevement", "cec", "cpb", "pas_cec"]

MIN_VIEW_S = 2
MIN_SEL_S = 1
MAX_WINDOW_POINTS = 12000

COLOR_MAIN = {"normal": "#43A047", "pas_normal": "#E53935", "mauvaise_qualite": "#757575"}
COLOR_TYPE = {
    "artefact_indetermine": "#E53935",
    "artefact_purge": "#AD1457",
    "artefact_prelevement": "#6D4C41",
    "cec": "#FB8C00",
    "cpb": "#8E24AA",
    "pas_cec": "#00897B",
}
BG_CEC = "#FB8C00"
BG_CLAMP = "#7E57C2"
PENDING = "rgba(30,136,229,0.25)"


# --------- Utils ---------
# Rôle : extraire une série temporelle exploitable à partir du DataFrame (col "datetime"/"DateTime" ou index datetime).
# Entrée : df (pd.DataFrame)
# Sortie : pd.Series de timestamps
# Erreurs : ValueError si aucune info temporelle valide n’est trouvée.
def _ensure_datetime(df: pd.DataFrame) -> pd.Series:
    if "datetime" in df:
        t = pd.to_datetime(df["datetime"], errors="coerce")
        if t.notna().any():
            return t
    if "DateTime" in df:
        col = df["DateTime"]
        if np.issubdtype(np.asarray(col).dtype, np.number):
            t = pd.to_datetime(col, unit="s", errors="coerce")
        else:
            t = pd.to_datetime(col, errors="coerce")
        if t.notna().any():
            return t
    if isinstance(df.index, pd.DatetimeIndex):
        return pd.to_datetime(df.index, errors="coerce").to_series(index=df.index)
    raise ValueError("Pas d'info temporelle ('datetime'/'DateTime'/index datetime).")


# Rôle : trouver la colonne ABP parmi la liste de colonnes candidates ABP_COLS.
# Entrée : df (pd.DataFrame)
# Sortie : nom de colonne (str)
# Erreurs : ValueError si aucune colonne valide n’est trouvée.
def _find_abp_col(df: pd.DataFrame) -> str:
    for c in ABP_COLS:
        if c in df.columns:
            return c
    raise ValueError(f"Colonne ABP introuvable. Cherché: {', '.join(ABP_COLS)}")


# Rôle : formatage utilitaire en HH:MM:SS d’un entier de secondes.
# Entrée : sec (int)
# Sortie : str "HH:MM:SS"
def _fmt_hms(sec: int) -> str:
    return pd.to_datetime(sec, unit="s").strftime("%H:%M:%S")


# Rôle : renvoyer l’index du timestamp le plus proche dans un tableau trié.
# Entrées : t_sorted (array-like de datetime64), ts (timestamp cible)
# Sortie : int (index)
def _nearest_idx(t_sorted, ts):
    x = t_sorted
    i = int(np.searchsorted(x, np.datetime64(ts), side="left"))
    if i == 0:
        return 0
    if i >= len(x):
        return len(x) - 1
    l = abs((x[i - 1] - np.datetime64(ts)).astype("timedelta64[ms]").astype(int))
    r = abs((x[i] - np.datetime64(ts)).astype("timedelta64[ms]").astype(int))
    return i - 1 if l <= r else i


# Rôle : convertir une série binaire (0/1) en intervals (start, end) au format ISO, pour dessiner des bandes (ex. CEC/Clamp).
# Entrées : x_ts (timestamps), series01 (binaire 0/1)
# Sortie : liste de tuples (start_iso, end_iso)
def _runs_to_intervals(x_ts, series01):
    arr = pd.Series(series01).fillna(0).astype(bool).to_numpy()
    if not arr.any():
        return []
    x_iso = (
        pd.Series(pd.to_datetime(x_ts).astype("datetime64[us]"))
        .dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        .tolist()
    )
    out, start = [], None
    for i, on in enumerate(arr):
        if on and start is None:
            start = x_iso[i]
        elif (not on) and (start is not None):
            out.append((start, x_iso[i]))
            start = None
    if start is not None:
        out.append((start, x_iso[-1]))
    return out


# Rôle : downsample pour l’affichage : conserve min et max de blocs, afin de limiter à MAX_WINDOW_POINTS (perf/lecture).
# Entrées : x_ts (timestamps), y (valeurs), max_pts (int)
# Sortie : (x_ds, y_ds) numpy arrays
def _downsample_minmax(x_ts, y, max_pts=MAX_WINDOW_POINTS):
    n = len(y)
    if n <= max_pts:
        return x_ts, y
    buckets = max(1, max_pts // 2)
    step = int(np.ceil(n / buckets))
    xs, ys = [], []
    for s in range(0, n, step):
        e = min(n, s + step)
        if e - s <= 0:
            break
        ych = y[s:e]
        i_min = int(np.argmin(ych)) + s
        i_max = int(np.argmax(ych)) + s
        for idx in sorted([i_min, i_max]):
            xs.append(x_ts[idx])
            ys.append(y[idx])
    return np.array(xs), np.array(ys)


# Rôle : conteneur simple pour récupérer les annotations saisies par l’utilisateur.
# Méthode : dataframe() -> pd.DataFrame avec les colonnes documentées ci-dessous.
class AnnotHandle:
    def __init__(self, getter):
        self._getter = getter

    def dataframe(self):
        cols = [
            "patient_id",
            "start_time",
            "end_time",
            "duration_s",
            "start_idx",
            "end_idx",
            "n_samples",
            "label_main",
            "artefact_type",
            "cec",
            "cpb",
            "notes",
            "created_at_iso",
        ]
        return pd.DataFrame(self._getter(), columns=cols)


# --------- Main ---------
# Rôle : lance le widget d’annotation complet (UI + logique) et renvoie un AnnotHandle pour récupérer les annotations.
# Entrées :
#   - df_raw : DataFrame des données (avec datetime/DateTime ou index datetime + une colonne ABP)
#   - patient_id : identifiant du patient (str)
#   - window_seconds : durée initiale de la fenêtre affichée (int)
# Sortie :
#   - AnnotHandle (utiliser .dataframe() pour récupérer les annotations sous forme de DataFrame)
def launch_abp_annotator(df_raw: pd.DataFrame, patient_id="unknown", window_seconds=120):
    df = df_raw.copy()

    # temps + signal
    t_ns = _ensure_datetime(df)
    if t_ns.isna().all():
        raise ValueError("'datetime' illisible.")
    abp_col = _find_abp_col(df)
    y_abp = pd.to_numeric(df[abp_col], errors="coerce").astype(float)

    # binaires éventuels
    for c in ("clampage", "cec"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # tri temporel
    t_us = t_ns.astype("datetime64[us]")
    order = np.argsort(t_us.values.astype("datetime64[ns]"))
    t_sorted = t_us.values[order]
    y_sorted = y_abp.values[order]
    clamp_sorted = df["clampage"].values[order] if "clampage" in df else None
    cec_sorted = df["cec"].values[order] if "cec" in df else None
    x_all = pd.to_datetime(t_sorted)

    # bornes (sec)
    times_sec = t_ns.values.astype("datetime64[s]").astype("int64")
    tmin_s, tmax_s = int(times_sec.min()), int(times_sec.max())
    center0 = tmin_s + (tmax_s - tmin_s) // 2

    # ===== NAV (HAUT) =====
    nav_win_val = w.BoundedIntText(
        value=int(window_seconds), min=MIN_VIEW_S, max=24 * 3600, step=2, description="Durée vue:"
    )
    nav_win_unit = w.ToggleButtons(options=[("s", "s"), ("min", "min")], value="s", description="Unit:")
    nav_center = w.IntSlider(
        min=tmin_s,
        max=tmax_s,
        value=center0,
        step=1,
        description="Centre vue:",
        layout=w.Layout(width="95%"),
        continuous_update=False,
    )
    nav_center_hms = w.Text(value=_fmt_hms(center0), description="HH:MM:SS:")
    btn_left = w.Button(description="◀︎ -10 s")
    btn_right = w.Button(description="+10 s ▶︎")
    btn_prev = w.Button(description="⟸ -Fenêtre")
    btn_next = w.Button(description="+Fenêtre ⟹")
    btn_autoY = w.Button(description="↻ Auto Y")
    show_clamp = w.Checkbox(value=("clampage" in df), description="Clampage")
    show_cec = w.Checkbox(value=("cec" in df), description="CEC")

    # ===== SÉLECTION (BAS) =====
    sel_step_dd = w.Dropdown(
        options=[("1 s", 1), ("2 s", 2), ("5 s", 5), ("10 s", 10), ("30 s", 30)],
        value=5,
        description="Pas sélection:",
    )
    live_overlay = w.Checkbox(value=True, description="Live overlay (drag)")
    sel_slider = w.IntRangeSlider(
        min=tmin_s,
        max=tmax_s,
        step=sel_step_dd.value,
        value=(center0 - window_seconds // 2, center0 + window_seconds // 2),
        description="Sélection:",
        layout=w.Layout(width="95%"),
        continuous_update=live_overlay.value,
        readout=False,
    )
    sel_readout = w.HTML()

    # nudges rapides
    nudge_left = w.Button(description="◀ pas")
    nudge_right = w.Button(description="pas ▶")
    shrink = w.Button(description="⟲ -pas")
    expand = w.Button(description="⟲⟲ +pas")
    take_view = w.Button(description="🧲 = Vue")

    # ===== Annotation =====
    main_label = w.ToggleButtons(
        options=[("Normal", "normal"), ("Pas normal", "pas_normal"), ("Mauvaise qualité", "mauvaise_qualite")],
        value="pas_normal",
        description="Classe:",
    )
    type_dd = w.Dropdown(options=ARTIFACT_TYPES, value="artefact_indetermine", description="Type:")
    type_dd.disabled = main_label.value != "pas_normal"

    cec_badge = w.HTML()
    cpb_badge = w.HTML()
    notes = w.Textarea(value="", placeholder="Notes (optionnel)", description="Notes:", layout=w.Layout(width="100%", height="60px"))

    btn_add = w.Button(description="➕ Ajouter", button_style="primary")
    btn_undo = w.Button(description="↶ Retirer dernière", button_style="warning")
    btn_show = w.Button(description="👀 DataFrame")

    status = w.HTML()
    out_df = w.Output()

    # ===== Figure =====
    fig = go.FigureWidget()
    fig.update_layout(
        height=540,
        hovermode="x unified",
        xaxis=dict(type="date"),
        yaxis=dict(),
        uirevision="keep",
    )

    # état
    rows = []
    ann_shapes = []
    pending = {"x0": None, "x1": None}
    state_view = {"xmin": None, "xmax": None}
    _sel_guard = {"on": False}
    _throttle = {"t": 0.0}

    # ------- Helpers -------
    def _nav_win_seconds() -> int:
        v = max(MIN_VIEW_S, int(nav_win_val.value))
        return v * 60 if nav_win_unit.value == "min" else v

    def _nav_bounds():
        half = _nav_win_seconds() // 2
        c = int(nav_center.value)
        a = max(tmin_s, min(tmax_s, c - half))
        b = max(tmin_s, min(tmax_s, c + half))
        if b <= a:
            b = min(tmax_s, a + MIN_VIEW_S)
        return pd.to_datetime(a, unit="s"), pd.to_datetime(b, unit="s")

    def _bg_shapes(x_ts, series01, color, op):
        shapes = []
        for x0, x1 in _runs_to_intervals(x_ts, series01):
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=x0,
                    x1=x1,
                    y0=0,
                    y1=1,
                    fillcolor=color,
                    opacity=op,
                    line=dict(width=0),
                    layer="below",
                )
            )
        return shapes

    def _refresh_shapes():
        shapes = []
        xw_min, xw_max = state_view["xmin"], state_view["xmax"]
        if xw_min is not None and xw_max is not None:
            mask = (x_all >= xw_min) & (x_all <= xw_max)
            if show_clamp.value and (clamp_sorted is not None) and mask.any():
                shapes += _bg_shapes(x_all[mask], clamp_sorted[mask], BG_CLAMP, 0.14)
            if show_cec.value and (cec_sorted is not None) and mask.any():
                shapes += _bg_shapes(x_all[mask], cec_sorted[mask], BG_CEC, 0.12)
        for s in ann_shapes:
            if xw_min is None or xw_max is None or (pd.to_datetime(s["x1"]) >= xw_min and pd.to_datetime(s["x0"]) <= xw_max):
                shapes.append(
                    dict(
                        type="rect",
                        xref="x",
                        yref="paper",
                        x0=s["x0"],
                        x1=s["x1"],
                        y0=0,
                        y1=1,
                        fillcolor=s["color"],
                        opacity=0.22,
                        line=dict(width=0),
                        layer="above",
                    )
                )
        if pending["x0"] is not None and pending["x1"] is not None:
            if xw_min is None or xw_max is None or (pending["x1"] >= xw_min and pending["x0"] <= xw_max):
                shapes.append(
                    dict(
                        type="rect",
                        xref="x",
                        yref="paper",
                        x0=pending["x0"],
                        x1=pending["x1"],
                        y0=0,
                        y1=1,
                        fillcolor=PENDING,
                        opacity=0.25,
                        line=dict(width=0),
                        layer="above",
                    )
                )
        fig.layout.shapes = tuple(shapes)

    def _reframe_selection_slider():
        if state_view["xmin"] is None or state_view["xmax"] is None:
            return
        new_min = int(pd.Timestamp(state_view["xmin"]).value // 1_000_000_000)
        new_max = int(pd.Timestamp(state_view["xmax"]).value // 1_000_000_000)
        if new_max - new_min < MIN_VIEW_S:
            new_max = new_min + MIN_VIEW_S

        step = int(sel_step_dd.value)
        try:
            a_cur, b_cur = sel_slider.value
        except Exception:
            a_cur, b_cur = new_min, new_max

        def _snap(v):
            return new_min + ((max(new_min, min(new_max, v)) - new_min) // step) * step

        a_new = _snap(a_cur)
        b_new = _snap(b_cur)
        if b_new <= a_new:
            b_new = min(new_max, a_new + step)

        if _sel_guard["on"]:
            return
        _sel_guard["on"] = True
        try:
            sel_slider.min = min(sel_slider.min, new_min)
            sel_slider.max = max(sel_slider.max, new_max)
            sel_slider.step = step
            sel_slider.value = (a_new, b_new)
            sel_slider.min = new_min
            sel_slider.max = new_max
            sel_slider.continuous_update = bool(live_overlay.value)

            pending["x0"] = pd.to_datetime(a_new, unit="s")
            pending["x1"] = pd.to_datetime(b_new, unit="s")
            _update_badges(pending["x0"], pending["x1"])
            _update_readout()
        finally:
            _sel_guard["on"] = False

    def _update_readout():
        if pending["x0"] is None or pending["x1"] is None:
            sel_readout.value = "<i>Aucune sélection</i>"
            return
        t0, t1 = pending["x0"], pending["x1"]
        if t1 < t0:
            t0, t1 = t1, t0
        sel_readout.value = f"<b>Sélection:</b> {t0.strftime('%H:%M:%S')} → {t1.strftime('%H:%M:%S')} (durée: {(t1 - t0).total_seconds():.1f} s)"

    def _auto_flags(t0, t1):
        if t1 < t0:
            t0, t1 = t1, t0
        i0 = _nearest_idx(t_sorted, t0)
        i1 = _nearest_idx(t_sorted, t1)
        if i1 < i0:
            i0, i1 = i1, i0
        cec_auto = int((cec_sorted is not None) and (np.nanmax(cec_sorted[i0 : i1 + 1]) > 0))
        cpb_auto = int((clamp_sorted is not None) and (np.nanmax(clamp_sorted[i0 : i1 + 1]) > 0))
        return cec_auto, cpb_auto

    def _badge(label, active):
        col = "#2E7D32" if active else "#9E9E9E"
        txt = "Oui" if active else "Non"
        return f"<span style='border-radius:8px;padding:3px 8px;background:{col};color:white;margin-right:8px'>{label}: {txt}</span>"

    def _update_badges(t0, t1):
        cec_a, cpb_a = _auto_flags(t0, t1)
        cec_badge.value = _badge("CEC", cec_a) if cec_sorted is not None else "<i>CEC: N/A</i>"
        cpb_badge.value = _badge("CPB", cpb_a) if clamp_sorted is not None else "<i>CPB: N/A</i>"

    def _draw_view():
        t0, t1 = _nav_bounds()
        mask = (x_all >= t0) & (x_all <= t1)
        with fig.batch_update():
            fig.data = ()
            if not mask.any():
                fig.update_xaxes(range=[t0, t1])
                fig.layout.shapes = ()
                state_view["xmin"], state_view["xmax"] = t0, t1
            else:
                xw = x_all[mask]
                yw = y_sorted[mask]
                if len(xw) > MAX_WINDOW_POINTS:
                    xw_ds, yw_ds = _downsample_minmax(np.asarray(xw), np.asarray(yw), MAX_WINDOW_POINTS)
                    xw = pd.to_datetime(xw_ds)
                    yw = yw_ds
                fig.add_trace(go.Scattergl(x=xw, y=yw, mode="lines", name="ABP"))
                fig.update_xaxes(range=[xw.min(), xw.max()])
                state_view["xmin"], state_view["xmax"] = xw.min(), xw.max()
            _reframe_selection_slider()
            _refresh_shapes()

    # ------- Actions sélection -------
    def _set_selection_from_secs(a_s, b_s, snap=True):
        new_min = int(pd.Timestamp(state_view["xmin"]).value // 1_000_000_000)
        new_max = int(pd.Timestamp(state_view["xmax"]).value // 1_000_000_000)
        step = int(sel_step_dd.value)
        if snap:
            a_s = new_min + ((a_s - new_min) // step) * step
            b_s = new_min + ((b_s - new_min) // step) * step
        a_s = max(new_min, min(new_max, a_s))
        b_s = max(new_min, min(new_max, b_s))
        if b_s <= a_s:
            b_s = min(new_max, a_s + step)
        if _sel_guard["on"]:
            return
        _sel_guard["on"] = True
        try:
            sel_slider.value = (a_s, b_s)
            pending["x0"] = pd.to_datetime(a_s, unit="s")
            pending["x1"] = pd.to_datetime(b_s, unit="s")
            _update_badges(pending["x0"], pending["x1"])
            _update_readout()
            _refresh_shapes()
        finally:
            _sel_guard["on"] = False

    def _nudge(delta):
        a_s, b_s = sel_slider.value
        _set_selection_from_secs(a_s + delta, b_s + delta)

    def _resize(delta):
        a_s, b_s = sel_slider.value
        c = (a_s + b_s) // 2
        half = max(1, (b_s - a_s) // 2 + delta // 2)
        _set_selection_from_secs(c - half, c + half)

    def _take_view():
        new_min = int(pd.Timestamp(state_view["xmin"]).value // 1_000_000_000)
        new_max = int(pd.Timestamp(state_view["xmax"]).value // 1_000_000_000)
        _set_selection_from_secs(new_min, new_max)

    # ------- Append annotation -------
    def _append():
        t0, t1 = pending["x0"], pending["x1"]
        if t0 is None or t1 is None:
            status.value = "<i>Aucune sélection.</i>"
            return
        if (t1 - t0).total_seconds() < MIN_SEL_S:
            status.value = f"<i>Sélection trop courte (≥ {MIN_SEL_S}s).</i>"
            return
        label = main_label.value
        a_type = type_dd.value if label == "pas_normal" else ""
        cec_a, cpb_a = _auto_flags(t0, t1)
        i0 = _nearest_idx(t_sorted, t0)
        i1 = _nearest_idx(t_sorted, t1)
        if i1 < i0:
            i0, i1 = i1, i0
        rows.append(
            dict(
                patient_id=patient_id,
                start_time=t0.isoformat(),
                end_time=t1.isoformat(),
                duration_s=float((t1 - t0).total_seconds()),
                start_idx=int(i0),
                end_idx=int(i1),
                n_samples=int(i1 - i0 + 1),
                label_main=label,
                artefact_type=a_type,
                cec=int(cec_a),
                cpb=int(cpb_a),
                notes=notes.value or "",
                created_at_iso=dt.datetime.utcnow().isoformat(),
            )
        )
        color = COLOR_TYPE.get(a_type) if label == "pas_normal" else COLOR_MAIN.get(label, "#607D8B")
        ann_shapes.append({"x0": t0, "x1": t1, "color": color})
        status.value = f"<span style='color:#070'>Ajouté (total: {len(rows)}) — {t0.strftime('%H:%M:%S')} → {t1.strftime('%H:%M:%S')}</span>"
        _refresh_shapes()

    # ------- Callbacks NAV -------
    def _nav_shift(delta_s):
        nav_center.value = int(np.clip(nav_center.value + delta_s, tmin_s, tmax_s))

    def _on_nav_center(change):
        nav_center_hms.value = _fmt_hms(change["new"])
        _draw_view()

    def _on_nav_hms_submit(_):
        try:
            h, m, s = map(int, nav_center_hms.value.split(":"))
            base = pd.to_datetime(x_all.min()).normalize()
            sec = int((base + pd.Timedelta(hours=h, minutes=m, seconds=s)).timestamp())
            nav_center.value = int(np.clip(sec, tmin_s, tmax_s))
        except Exception:
            pass

    def _on_nav_win_change(_):
        _draw_view()

    btn_left.on_click(lambda _x: _nav_shift(-10))
    btn_right.on_click(lambda _x: _nav_shift(+10))
    btn_prev.on_click(lambda _x: _nav_shift(-_nav_win_seconds()))
    btn_next.on_click(lambda _x: _nav_shift(+_nav_win_seconds()))
    nav_center.observe(_on_nav_center, names="value")
    nav_center_hms.on_submit(_on_nav_hms_submit)
    nav_win_val.observe(_on_nav_win_change, names="value")
    nav_win_unit.observe(_on_nav_win_change, names="value")
    show_clamp.observe(lambda _x: _refresh_shapes(), names="value")
    show_cec.observe(lambda _x: _refresh_shapes(), names="value")
    btn_autoY.on_click(lambda _x: fig.update_yaxes(autorange=True))

    # ------- Callbacks sélection -------
    def _on_sel_change(_):
        if live_overlay.value:
            now = _time.time()
            if now - _throttle["t"] < 0.05:
                return
            _throttle["t"] = now
        if _sel_guard["on"]:
            return
        a_s, b_s = sel_slider.value
        step = int(sel_step_dd.value)
        a_s = sel_slider.min + ((a_s - sel_slider.min) // step) * step
        b_s = sel_slider.min + ((b_s - sel_slider.min) // step) * step
        if b_s <= a_s:
            b_s = min(sel_slider.max, a_s + step)
        if (a_s, b_s) != sel_slider.value:
            _set_selection_from_secs(a_s, b_s)
            return
        pending["x0"] = pd.to_datetime(a_s, unit="s")
        pending["x1"] = pd.to_datetime(b_s, unit="s")
        _update_badges(pending["x0"], pending["x1"])
        _update_readout()
        _refresh_shapes()

    sel_slider.observe(_on_sel_change, names="value")

    def _on_step_change(change):
        step = int(change["new"])
        sel_slider.step = step
        a_s, b_s = sel_slider.value
        _set_selection_from_secs(a_s, b_s)

    sel_step_dd.observe(_on_step_change, names="value")

    def _on_live_toggle(_):
        sel_slider.continuous_update = bool(live_overlay.value)

    live_overlay.observe(_on_live_toggle, names="value")

    nudge_left.on_click(lambda _x: _nudge(-int(sel_step_dd.value)))
    nudge_right.on_click(lambda _x: _nudge(+int(sel_step_dd.value)))
    shrink.on_click(lambda _x: _resize(-int(sel_step_dd.value)))
    expand.on_click(lambda _x: _resize(+int(sel_step_dd.value)))
    take_view.on_click(lambda _x: _take_view())

    # ------- Callbacks form -------
    def _on_label_change(chg):
        type_dd.disabled = chg["new"] != "pas_normal"

    main_label.observe(_on_label_change, names="value")
    btn_add.on_click(lambda _x: _append())

    def _on_undo(_):
        if rows:
            rows.pop()
            if ann_shapes:
                ann_shapes.pop()
            status.value = "<i>Dernière annotation retirée.</i>"
            _refresh_shapes()
        else:
            status.value = "<i>Rien à retirer.</i>"

    btn_undo.on_click(_on_undo)

    def _on_show_df(_):
        with out_df:
            clear_output(wait=True)
            display(pd.DataFrame(rows))

    btn_show.on_click(_on_show_df)

    # ------- Layout -------
    header = w.HTML(
        f"<h3>Annotateur ABP — Patient: <code>{patient_id}</code> — Source: <code>{abp_col}</code></h3>"
    )
    nav_row1 = w.HBox([nav_win_val, nav_win_unit])
    nav_row2 = w.HBox([btn_prev, btn_left, nav_center, btn_right, btn_next])
    nav_row3 = w.HBox([w.Label("Centre:"), nav_center_hms, show_clamp, show_cec, btn_autoY])

    sel_row0 = w.HBox([sel_step_dd, live_overlay, nudge_left, nudge_right, shrink, expand, take_view])
    sel_row = w.HBox([sel_slider])
    sel_row2 = w.HBox([sel_readout, w.HTML("&nbsp;"), w.HTML("<b>Contexte:</b>"), cec_badge, cpb_badge])

    controls = w.HBox([main_label, type_dd])
    actions = w.HBox([btn_add, btn_undo, btn_show])

    panel = w.VBox(
        [header, nav_row1, nav_row2, nav_row3, fig, sel_row0, sel_row, sel_row2, controls, notes, actions, status, out_df]
    )
    display(panel)

    # ------- Init -------
    def _init_selection_to_view():
        t0, t1 = _nav_bounds()
        state_view["xmin"], state_view["xmax"] = t0, t1
        _reframe_selection_slider()

    _draw_view()
    _init_selection_to_view()

    return AnnotHandle(lambda: list(rows))
