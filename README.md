# DESU_ABP — Autoencodeur 1D & SQI pour la qualité des signaux de pression artérielle

Pipeline et modèles pour (i) détecter/filtrer les artéfacts dans les signaux de pression artérielle (ABP) pédiatriques, (ii) construire un **Signal Quality Index (SQI)** semi-supervisé à partir d’un **autoencodeur 1D** avec tête de classification.

Projet associé au mémoire **DESU – Data Science Appliquée aux Neurosciences** (Samuel Dahan).

---

## 🗂️ Structure du dépôt

DESU_ABP/
├─ ae.py # Autoencodeur 1D, FiniteSanitizer, CropToRef, masked_mse
├─ sqi_auc.py # SQI (alpha dynamique), normalisation RMSE, AUC, helpers
├─ grid.py # Évaluation multi-modèles & (αmin, αmax, k), exports ROC/JSON/CSV
├─ recon_viewer.py # Reconstruction overlap-add + projection SQI continu (EMA)
├─ data_io.py # Prétraitements, canaux contextuels, NPZ, datasets finaux
├─ windowing.py # Fenêtrage & z-score par fenêtre
├─ window_mask.py # Masque “bad” (plateaux / linéarité)
├─ preprocessing_labeled_window.py # Dataset supervisé depuis annotations
├─ annotation.py # Outil d’annotation ABP
├─ train.py # Entraînement AE1D (reconstruction + classification)
├─ utils.py, config.py
├─ requirements.txt
└─ model/ # ⬅️ Modèles pré-entraînés (.keras) présents dans la branche main
├─ NEWAE1D_lat4_...best_full.keras
├─ NEWAE1D_lat8_...best_full.keras
└─ ...



## 🧰 Prérequis & installation

- **Python** : 3.11 recommandé  
- **GPU (optionnel)** : CUDA compatible TensorFlow 2.20  
- **Graphviz** (optionnel, pour schémas) : `sudo apt-get install graphviz`

```bash
git clone https://github.com/samuel-0902/DESU_ABP.git
cd DESU_ABP
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
```


Données attendues

Brut : CSV haute fréquence (ABP ~115 Hz effectif), synchronisés ECG (non versionnés).

Intermédiaires par patient : file_npz/patients_<PID>_clean.npz (X normalisé + M/SD + meta).

Labels supervisés : data_label.npz (fenêtres annotées normal/anormal).

Deep learning et analyse post-opératoire des signaux physiologiques : détection d’artéfacts et proposition d’un indice de qualité (ABP).
Samuel Dahan — DESU Data Science Appliquée aux Neurosciences, 2025.
