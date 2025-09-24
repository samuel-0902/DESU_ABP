'''
Configuration globale du projet : chemins Windows,
fonctions utilitaires dépendantes.
'''

import os

BASE_DIR     = r"D:\Nouveau dossier\list\csv_file"
LISTING_PATH = r"D:\Nouveau dossier\Sauvegarde listing 26112022.xlsx"
ANNOT_PATH   = r"C:\Users\dahan\OneDrive\Bureau\autoreg\annotations\annotations_merged.parquet"

SAVE_ROOT    = r"D:\Nouveau dossier\conda_autoreg\models"
RUNS_DIR     = r"D:\Nouveau dossier\conda_autoreg\runs"

SIGNAL_COL   = "abp[mmHg]"


USE_GLOBAL_ANN_POOL = True

WIN_S      = 5.0
STRIDE_S   = 2.5
TARGET_FS  = 115.0
DOWNSAMPLES = 3
MIN_POINTS = 5


ADD_AGE_CHANNEL   = True
ADD_STATE_CHANNEL = True
AGE_MAX_YEARS     = 18.0


def compute_target_len(win_s: float, target_fs: float = TARGET_FS, downsamples: int = DOWNSAMPLES) -> int:
    '''
    Calcule la longueur de fenêtre compatible avec le facteur de sous-échantillonnage.
    '''
    mult = 2 ** downsamples
    base = int(round(win_s * target_fs))
    L = max(mult, int(round(base / mult)) * mult)
    return L

TARGET_LEN = compute_target_len(WIN_S, TARGET_FS, DOWNSAMPLES)
