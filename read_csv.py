"""
Module de lecture et pré-traitement des fichiers CSV ABP pédiatriques.

Fonctions principales :
- read_csv_same_output : lecture robuste des CSV avec détection automatique du format date/heure
- get_patient_caracteristics_aligned : extraction des métadonnées patient depuis l'Excel listings
- mark_periods : création de colonnes binaires 0/1 pour les phases CEC/clampage

Formats d'entrée :
- CSV : colonne DateTime (ou variantes) + abp[mmHg] + éventuellement cec/clampage
- Excel listings : fichier .xlsx avec colonnes 'N_doss', 'birth date', 'H CEC', etc.

Formats de sortie :
- DataFrame pandas indexé par datetime naive, colonnes normalisées
- Series pandas (1 ligne) pour les caractéristiques patient
- DataFrame augmenté des colonnes binaires cec/clampage
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, savgol_filter
from typing import Union


# ---------- lecture du CSV ----------
def read_csv_same_output(path: str, sep: str = ';') -> pd.DataFrame:
    """
    Lit un fichier CSV ABP avec gestion robuste de l'encodage et du format datetime.

    Args:
        path: chemin vers le fichier CSV
        sep: séparateur de colonnes (défaut: ';')

    Returns:
        DataFrame pandas avec :
        - index : DatetimeIndex naive (UTC removed)
        - colonnes : 'abp[mmHg]' normalisée, 'DataSource' si présente
    """
    # lecture avec essais multiples d'encodage
    for enc in ('utf-8', 'cp1252', 'latin1', 'utf-8-sig'):
        try:
            df = pd.read_csv(path, sep=sep, encoding=enc, engine='python', on_bad_lines='skip')
            break
        except UnicodeDecodeError:
            continue
    else:
        df = pd.read_csv(path, sep=sep, encoding='utf-8', engine='python',
                         on_bad_lines='skip', encoding_errors='replace')

    # normalisation des noms de colonnes
    df.columns = df.columns.astype(str).str.strip().str.replace('\ufeff', '', regex=False)

    # recherche des colonnes datetime et datasource
    def find_col(cols, target):
        tgt = target.lower().replace(' ', '')
        for c in cols:
            if str(c).lower().replace(' ', '') == tgt:
                return c
        return None

    dt_col = find_col(df.columns, 'DateTime') or find_col(df.columns, 'datetime')
    ds_col = find_col(df.columns, 'DataSource') or find_col(df.columns, 'datasource')

    if dt_col is None:
        raise KeyError(f"Colonne DateTime introuvable. Colonnes: {list(df.columns)}")

    # parsing robuste des dates
    s = df[dt_col].astype(str).str.strip()
    dt = pd.to_datetime(s, errors='coerce', utc=False, dayfirst=False)

    if dt.isna().mean() > 0.2:
        s_num = pd.to_numeric(s.str.replace(",", ".", regex=False), errors='coerce')
        if s_num.notna().sum() == 0:
            raise ValueError("Datetime column could not be parsed as string nor numeric.")

        m = s_num.max(skipna=True)
        if m > 1e12:
            dt = pd.to_datetime(s_num, unit='ms', origin='unix', errors='coerce')
        elif m > 1e10:
            dt = pd.to_datetime(s_num, unit='us', origin='unix', errors='coerce')
        elif m > 1e9:
            dt = pd.to_datetime(s_num, unit='s', origin='unix', errors='coerce')
        else:
            base = pd.to_datetime('1899-12-30')
            dt = base + pd.to_timedelta(s_num, unit='D')

    if dt.isna().any():
        n_bad = int(dt.isna().sum())
        raise ValueError(f"{n_bad} datetime values could not be parsed; check the source format.")

    # finalisation
    df['datetime'] = dt
    df = df.set_index('datetime').sort_index()

    if ds_col is not None and ds_col != 'DataSource':
        df = df.rename(columns={ds_col: 'DataSource'})

    return df


# ---------- caractéristiques patient ----------
def get_patient_caracteristics_aligned(
    df_signal: pd.DataFrame,
    excel: Union[str, pd.DataFrame],
    number: int,
    id_col: str = "N_doss",
) -> pd.DataFrame:
    """
    Extrait les métadonnées patient depuis l'Excel listings et les aligne sur le jour du signal.

    Args:
        df_signal: DataFrame du signal (doit avoir DatetimeIndex)
        excel: chemin .xlsx/.xls OU DataFrame déjà chargé
        number: identifiant patient (int)
        id_col: nom de la colonne ID dans l'Excel

    Returns:
        DataFrame (1 ligne) avec :
        birth_date, age_cec, age (jours), taille, poids, SC,
        heures CEC/clampage, durées CEC/clampage (min)
    """
    if not isinstance(df_signal.index, pd.DatetimeIndex):
        dt_col = None
        for cand in ["DateTime", "datetime", "time", "Time", "timestamp"]:
            if cand in df_signal.columns:
                dt_col = cand
                break
        if dt_col is None:
            raise ValueError("df_signal doit avoir un DatetimeIndex ou une colonne DateTime.")
        df_signal = df_signal.copy()
        df_signal[dt_col] = pd.to_datetime(df_signal[dt_col], errors="coerce")
        df_signal = df_signal.set_index(dt_col).sort_index()
    if df_signal.index.tz is not None:
        df_signal = df_signal.tz_convert(None)

    if df_signal.index.size == 0 or df_signal.index[0] is pd.NaT:
        raise ValueError("Index temporel du signal vide ou invalide.")
    ref_day = df_signal.index.normalize()[0]

    if isinstance(excel, str):
        df_excel = pd.read_excel(excel)
    elif isinstance(excel, pd.DataFrame):
        df_excel = excel
    else:
        raise TypeError("Paramètre 'excel' doit être un chemin .xlsx/.xls ou un DataFrame.")

    if id_col not in df_excel.columns:
        raise KeyError(f"Colonne ID '{id_col}' introuvable dans l'Excel.")

    id_series = pd.to_numeric(df_excel[id_col], errors="coerce").astype('Int64')
    df_pat = df_excel[id_series == number]
    if df_pat.empty:
        raise KeyError(f"Aucune donnée pour patient {number} (col {id_col}).")

    row = df_pat.iloc[0]

    def _to_dt_from_hms(val):
        if pd.isna(val):
            return pd.NaT
        if isinstance(val, pd.Timestamp):
            t = val.time()
        elif hasattr(val, "to_pydatetime"):
            t = pd.to_datetime(val, errors="coerce").time()
        else:
            s = str(val)
            ts = pd.to_datetime(s, format="%H:%M:%S", errors="coerce")
            if pd.isna(ts):
                ts = pd.to_datetime(s, errors="coerce")
            if pd.isna(ts):
                return pd.NaT
            t = ts.time()
        return pd.to_datetime(f"{ref_day.date()} {t}")

    excel_origin = pd.Timestamp("1899-12-30")

    def _excel_days_to_ts(x):
        v = pd.to_numeric(x, errors="coerce")
        if pd.isna(v):
            return pd.NaT
        return excel_origin + pd.to_timedelta(float(v), unit="D")

    birth_date = _excel_days_to_ts(row.get("birth date"))
    age_cec_dt = _excel_days_to_ts(row.get("CEC date"))

    def _num(col):
        return pd.to_numeric(row[col], errors="coerce") if col in df_excel.columns else np.nan

    taille = _num("Taille")
    poids = _num("Poids")
    sc = _num("SC")

    H_CEC_full = _to_dt_from_hms(row.get("H CEC"))
    H_Fin_CEC_full = _to_dt_from_hms(row.get("H Fin CEC"))
    H_X_clamp_full = _to_dt_from_hms(row.get("H X clamp"))
    H_Fin_X_Clamp_full = _to_dt_from_hms(row.get("H Fin X Clamp"))

    def _dur(a, b):
        if pd.isna(a) or pd.isna(b):
            return np.nan
        return (b - a).total_seconds() / 60.0

    cec_duration = _dur(H_CEC_full, H_Fin_CEC_full)
    cpb_duration = _dur(H_X_clamp_full, H_Fin_X_Clamp_full)

    out = pd.Series({
        "birth_date": birth_date,
        "age_cec": age_cec_dt,
        "age": (age_cec_dt - birth_date).days if (pd.notna(age_cec_dt) and pd.notna(birth_date)) else np.nan,
        "taille": taille,
        "poids": poids,
        "SC": sc,
        "H CEC_full": H_CEC_full,
        "H Fin CEC_full": H_Fin_CEC_full,
        "H X clamp_full": H_X_clamp_full,
        "H Fin X Clamp_full": H_Fin_X_Clamp_full,
        "cec_duration": cec_duration,
        "cpb_duration": cpb_duration,
    })

    return out.to_frame().T


# ---------- marquage des périodes ----------
def mark_periods(df_signal: pd.DataFrame, df_caract: pd.DataFrame, start_col: str, stop_col: str, col_name: str) -> pd.DataFrame:
    """
    Ajoute une colonne binaire 0/1 dans df_signal indiquant la période définie par [start_col, stop_col].

    Args:
        df_signal: DataFrame du signal (index DatetimeIndex)
        df_caract: DataFrame (1 ligne) avec les colonnes start_col et stop_col
        start_col: nom de la colonne début dans df_caract
        stop_col: nom de la colonne fin dans df_caract
        col_name: nom de la nouvelle colonne à créer dans df_signal

    Returns:
        DataFrame augmenté de la colonne binaire col_name
    """
    t0 = df_caract[start_col].iloc[0]
    t1 = df_caract[stop_col].iloc[0]
    if pd.isna(t0) or pd.isna(t1):
        df_signal[col_name] = 0
    else:
        mask = (df_signal.index >= t0) & (df_signal.index <= t1)
        df_signal[col_name] = mask.astype(int)
    return df_signal
