"""
Utilitaires système  logging, dossiers
"""

import os
import json

import datetime
import numpy as np




def log(msg: str):
    '''
    Affiche un message horodaté sur la console (format HH:MM:SS).
    '''
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def ensure_dir(p):
    '''
    Crée le dossier (et ses parents) s’il n’existe pas.
    '''
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

