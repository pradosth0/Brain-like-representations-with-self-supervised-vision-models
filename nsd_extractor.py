#!/usr/bin/env python3
"""
NSD fMRI Extractor — Natural Scenes Dataset (Allen et al. 2022)
================================================================
Télécharge et extrait les volumes 3D préprocessés (1.8mm) pour chaque trial,
en faisant le lien avec l'image COCO correspondante.

S3 bucket : s3://natural-scenes-dataset (--region us-east-1)

POURQUOI LES BETAS = "IRM À 5 SECONDES" :
    Les fichiers betas_fithrf_GLMdenoise_RR contiennent une estimation GLM
    de l'amplitude de la réponse BOLD pour chaque trial. La fonction de réponse
    hémodynamique (HRF) fittée atteint son pic à ~5-6s après le début du stimulus
    (3s ON, 1s OFF dans NSD, TR=1.6s). Ces betas SONT la réponse cérébrale
    préprocessée à l'image, ce qui est équivalent (et scientifiquement supérieur)
    à extraire un volume brut à t+5s.

STRUCTURE DE SORTIE :
    nsd_output/
    ├── nsd_expdesign.mat               (téléchargé une fois)
    ├── trial_index.csv                  (index global : subject/session/trial → stim)
    └── subj{XX}/
        └── session{SS}/
            ├── trials_metadata.json     (infos complètes du batch)
            └── trial{TTTT}_stim{SSSSS}.nii.gz  (un volume 3D par trial)

USAGE :
    # Tous les sujets, toutes les sessions
    python nsd_extractor.py

    # Sujets 1 et 2, sessions 1 à 5
    python nsd_extractor.py --subjects 1 2 --sessions 1 2 3 4 5

    # Vérifier ce qui sera téléchargé sans télécharger
    python nsd_extractor.py --subjects 1 --sessions 1 --dry-run

DÉPENDANCES :
    pip install nibabel scipy h5py numpy
    # + AWS CLI configuré (aws configure) ou variables AWS_ACCESS_KEY_ID etc.
"""

import os
import gc
import csv
import json
import argparse
import subprocess
from pathlib import Path

import numpy as np
import nibabel as nib

try:
    import h5py
except ImportError:
    raise ImportError("Installer h5py : pip install h5py")
try:
    import scipy.io
except ImportError:
    raise ImportError("Installer scipy : pip install scipy")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BUCKET = "natural-scenes-dataset"
REGION = "us-east-1"

# Chemin S3 du fichier de design expérimental (commun à tous les sujets)
S3_EXPDESIGN = "nsddata/experiments/nsd/nsd_expdesign.mat"

# Template S3 pour les betas 1.8mm (GLMdenoise + ridge regression)
S3_BETAS_TEMPLATE = (
    "nsddata_betas/ppdata/{subject}/func1pt8mm/"
    "betas_fithrf_GLMdenoise_RR/betas_session{session:02d}.nii.gz"
)

# Constantes NSD
TR_SECONDS          = 1.6   # Repetition Time (s)
N_SESSIONS_MAX      = 40    # Maximum de sessions par sujet
TRIALS_PER_SESSION  = 750   # Estimation (la valeur réelle vient du NIfTI)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES S3
# ═══════════════════════════════════════════════════════════════════════════════

def s3_download(s3_key: str, local_path: Path) -> None:
    """Télécharge un fichier depuis S3 via AWS CLI."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aws", "s3", "cp",
        f"s3://{BUCKET}/{s3_key}",
        str(local_path),
        "--region", REGION,
        "--no-progress",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Échec du téléchargement S3 :\n"
            f"  clé    : {s3_key}\n"
            f"  erreur : {result.stderr.strip()}"
        )


def s3_exists(s3_key: str) -> bool:
    """Vérifie si une clé S3 existe."""
    cmd = ["aws", "s3", "ls", f"s3://{BUCKET}/{s3_key}", "--region", REGION]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and bool(result.stdout.strip())


def s3_list_sessions(subject: str) -> list[int]:
    """Liste les sessions disponibles pour un sujet sur S3."""
    prefix = (
        f"nsddata_betas/ppdata/{subject}/func1pt8mm/"
        "betas_fithrf_GLMdenoise_RR/"
    )
    cmd = ["aws", "s3", "ls", f"s3://{BUCKET}/{prefix}", "--region", REGION]
    result = subprocess.run(cmd, capture_output=True, text=True)
    sessions = []
    for line in result.stdout.strip().splitlines():
        fname = line.strip().split()[-1]  # ex: "betas_session03.nii.gz"
        if fname.startswith("betas_session") and fname.endswith(".nii.gz"):
            try:
                n = int(fname.replace("betas_session", "").replace(".nii.gz", ""))
                sessions.append(n)
            except ValueError:
                pass
    return sorted(sessions)


# ═══════════════════════════════════════════════════════════════════════════════
# CHARGEMENT DU DESIGN EXPÉRIMENTAL
# ═══════════════════════════════════════════════════════════════════════════════

def load_expdesign(mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Charge nsd_expdesign.mat et retourne les deux tableaux clés.

    masterordering : (n_total_trials,), dtype int, valeurs 1–10000
        masterordering[t] = k  →  le slot k de l'image unique montrée au trial t.
        (Partagé entre les 8 sujets : même ordre de présentation)

    subjectim : (8, 10000), dtype int, valeurs 1–73000
        subjectim[subject_idx, k-1] = nsd_stim_id  (1-indexé)
        → Index dans nsd_stimuli.hdf5 pour le sujet s, slot k.
        Accès image : nsd_stimuli[nsd_stim_id - 1, ...]
        (Les 1000 premiers slots k=1..1000 sont les images partagées entre sujets)
    """
    print(f"\nChargement de {mat_path.name} ...")

    def _load_hdf5():
        """Format MATLAB v7.3 (HDF5)."""
        with h5py.File(mat_path, "r") as f:
            print("  Format HDF5 — clés :", list(f.keys()))
            for key in f.keys():
                arr = np.array(f[key])
                print(f"    {key}: shape={arr.shape}, dtype={arr.dtype}")

            mo = np.array(f["masterordering"]).flatten().astype(int)

            si = np.array(f["subjectim"]).astype(int)
            # HDF5 transpose les matrices MATLAB : (10000, 8) → (8, 10000)
            if si.ndim == 2 and si.shape[0] > si.shape[1]:
                si = si.T
        return mo, si

    def _load_scipy():
        """Format MATLAB v5/v6."""
        mat = scipy.io.loadmat(str(mat_path))
        keys = [k for k in mat if not k.startswith("_")]
        print("  Format .mat — clés :", keys)
        for k in keys:
            try:
                print(f"    {k}: shape={mat[k].shape}, dtype={mat[k].dtype}")
            except Exception:
                pass

        mo = mat["masterordering"].flatten().astype(int)
        si = mat["subjectim"].astype(int)
        if si.ndim == 2 and si.shape[0] > si.shape[1]:
            si = si.T
        return mo, si

    try:
        masterordering, subjectim = _load_hdf5()
    except Exception as e_hdf5:
        print(f"  HDF5 échoué ({e_hdf5}), tentative scipy.io ...")
        try:
            masterordering, subjectim = _load_scipy()
        except Exception as e_scipy:
            raise RuntimeError(
                f"Impossible de charger {mat_path}.\n"
                f"  Erreur HDF5   : {e_hdf5}\n"
                f"  Erreur scipy  : {e_scipy}"
            )

    print(
        f"  masterordering : shape={masterordering.shape}, "
        f"range=[{masterordering.min()}, {masterordering.max()}]"
    )
    print(
        f"  subjectim      : shape={subjectim.shape}, "
        f"range=[{subjectim.min()}, {subjectim.max()}]"
    )
    return masterordering, subjectim


# ═══════════════════════════════════════════════════════════════════════════════
# TRAITEMENT D'UNE SESSION
# ═══════════════════════════════════════════════════════════════════════════════

def process_session(
    subject_name: str,
    subject_idx: int,
    session_num: int,
    masterordering: np.ndarray,
    subjectim: np.ndarray,
    trial_offset: int,
    out_root: Path,
    dry_run: bool = False,
) -> int:
    """
    Télécharge les betas d'une session, extrait un volume 3D par trial,
    sauvegarde, puis supprime le fichier brut pour économiser l'espace.

    Retourne le nombre de trials traités (pour calculer l'offset cumulatif).
    """
    session_dir = out_root / subject_name / f"session{session_num:02d}"
    done_flag   = session_dir / ".done"

    # Session déjà traitée ?
    if done_flag.exists():
        meta_file = session_dir / "trials_metadata.json"
        n_done = TRIALS_PER_SESSION
        if meta_file.exists():
            with open(meta_file) as f:
                n_done = json.load(f).get("n_trials", TRIALS_PER_SESSION)
        print(
            f"  [{subject_name}] Session {session_num:02d}: "
            f"déjà traitée ({n_done} trials), ignorée."
        )
        return n_done

    s3_key = S3_BETAS_TEMPLATE.format(subject=subject_name, session=session_num)

    if dry_run:
        exists = s3_exists(s3_key)
        print(
            f"  [DRY-RUN] [{subject_name}] Session {session_num:02d}: "
            f"{'✓ trouvée' if exists else '✗ absente'} — {s3_key}"
        )
        return TRIALS_PER_SESSION if exists else 0

    if not s3_exists(s3_key):
        print(
            f"  [{subject_name}] Session {session_num:02d}: "
            f"non trouvée sur S3, ignorée."
        )
        return 0

    session_dir.mkdir(parents=True, exist_ok=True)
    tmp_nii = session_dir / f"_tmp_betas_session{session_num:02d}.nii.gz"

    # ── Téléchargement ────────────────────────────────────────────────────────
    print(f"\n  [{subject_name}] Session {session_num:02d}: téléchargement ...")
    s3_download(s3_key, tmp_nii)
    size_mb = tmp_nii.stat().st_size / 1e6
    print(f"    → {size_mb:.1f} MB téléchargés")

    # ── Chargement du NIfTI 4D (X, Y, Z, n_trials) ───────────────────────────
    print(f"  [{subject_name}] Session {session_num:02d}: chargement NIfTI ...")
    img     = nib.load(str(tmp_nii))
    data    = img.get_fdata(dtype=np.float32)  # float32 pour économiser RAM
    affine  = img.affine
    header  = img.header.copy()
    n_trials = data.shape[3]

    print(
        f"  [{subject_name}] Session {session_num:02d}: "
        f"shape={data.shape}, trial_offset={trial_offset}"
    )

    # ── Extraction trial par trial ────────────────────────────────────────────
    records = []
    for t_local in range(n_trials):
        t_global = trial_offset + t_local

        if t_global >= len(masterordering):
            print(
                f"    Avertissement : trial global {t_global} dépasse "
                f"masterordering ({len(masterordering)}). Arrêt."
            )
            break

        # masterordering[t] = k (1-indexé, 1–10000)
        k_1indexed = int(masterordering[t_global])
        k_0indexed = k_1indexed - 1

        # subjectim[subject, k-1] = NSD stim ID (1-indexé, 1–73000)
        nsd_stim_id_1indexed = int(subjectim[subject_idx, k_0indexed])
        nsd_stim_id_0indexed = nsd_stim_id_1indexed - 1  # pour nsd_stimuli.hdf5

        # Sauvegarde du volume 3D
        fname   = f"trial{t_local:04d}_stim{nsd_stim_id_0indexed:05d}.nii.gz"
        out_nii = session_dir / fname
        nib.save(
            nib.Nifti1Image(data[:, :, :, t_local].copy(), affine, header),
            str(out_nii),
        )

        records.append({
            "session"          : session_num,
            "trial_in_session" : t_local,
            "global_trial"     : t_global,
            "image_slot_k"     : k_1indexed,          # 1-indexé (1–10000)
            "nsd_stim_id"      : nsd_stim_id_0indexed, # 0-indexé pour hdf5
            "is_shared_1000"   : k_1indexed <= 1000,   # images communes à tous
            "file"             : fname,
        })

    # ── Sauvegarde des métadonnées ────────────────────────────────────────────
    meta = {
        "subject"            : subject_name,
        "session"            : session_num,
        "n_trials"           : len(records),
        "trial_offset"       : trial_offset,
        "brain_volume_shape" : list(data.shape[:3]),
        "voxel_size_mm"      : 1.8,
        "TR_seconds"         : TR_SECONDS,
        "beta_version"       : "betas_fithrf_GLMdenoise_RR",
        "note" : (
            "Chaque volume 3D est un beta GLM single-trial (amplitude estimée). "
            "La HRF fittée pic à ~5-6s après début stimulus (3s ON, 1s OFF, TR=1.6s). "
            "nsd_stim_id : index 0-basé dans nsd_stimuli.hdf5 (shape 73000×425×425×3). "
            "is_shared_1000=true → image vue par les 8 sujets (test set partagé)."
        ),
        "trials": records,
    }
    with open(session_dir / "trials_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── Nettoyage : suppression du fichier brut ───────────────────────────────
    del data
    gc.collect()
    tmp_nii.unlink()
    done_flag.touch()

    print(
        f"  [{subject_name}] Session {session_num:02d}: "
        f"✓ {len(records)} trials sauvegardés, fichier brut supprimé."
    )
    return len(records)


# ═══════════════════════════════════════════════════════════════════════════════
# INDEX GLOBAL CSV
# ═══════════════════════════════════════════════════════════════════════════════

def build_global_index(out_root: Path) -> None:
    """
    Parcourt tous les trials_metadata.json et génère un CSV global :
    subject, session, trial_in_session, global_trial, nsd_stim_id,
    is_shared_1000, file_path
    """
    rows = []
    for meta_file in sorted(out_root.rglob("trials_metadata.json")):
        with open(meta_file) as f:
            meta = json.load(f)
        subj = meta["subject"]
        sess = meta["session"]
        for t in meta["trials"]:
            rows.append({
                "subject"          : subj,
                "session"          : sess,
                "trial_in_session" : t["trial_in_session"],
                "global_trial"     : t["global_trial"],
                "nsd_stim_id"      : t["nsd_stim_id"],   # 0-indexé pour hdf5
                "is_shared_1000"   : t["is_shared_1000"],
                "file_path"        : str(
                    Path(subj) / f"session{sess:02d}" / t["file"]
                ),
            })

    if not rows:
        print("Aucune métadonnée trouvée, index CSV non généré.")
        return

    index_path = out_root / "trial_index.csv"
    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n📋 Index global sauvegardé → {index_path}  ({len(rows)} lignes)")
    print(
        "   Colonnes : subject | session | trial_in_session | global_trial | "
        "nsd_stim_id | is_shared_1000 | file_path"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NSD fMRI Extractor – volumes 3D préprocessés 1.8mm par trial",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--subjects", nargs="+", type=int, default=list(range(1, 9)),
        metavar="N",
        help="Numéros de sujets à traiter (1–8). Défaut : les 8.",
    )
    parser.add_argument(
        "--sessions", nargs="+", type=str, default=["all"],
        metavar="S",
        help=(
            "Numéros de sessions (1–40) ou 'all'. "
            "Ex: --sessions 1 2 3  ou  --sessions all"
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, default="./nsd_output",
        help="Répertoire de sortie (défaut: ./nsd_output).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Vérifie l'existence des fichiers S3 sans télécharger.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Résolution de la liste de sessions
    if args.sessions == ["all"]:
        sessions = list(range(1, N_SESSIONS_MAX + 1))
    else:
        sessions = [int(s) for s in args.sessions]

    print("=" * 65)
    print("  NSD fMRI Extractor")
    print(f"  Sujets   : {args.subjects}")
    print(f"  Sessions : {sessions[:5]}{'...' if len(sessions) > 5 else ''}")
    print(f"  Sortie   : {out_root.resolve()}")
    print(f"  Mode     : {'DRY-RUN' if args.dry_run else 'EXTRACTION'}")
    print("=" * 65)

    # ── Étape 1 : design expérimental ─────────────────────────────────────────
    expdesign_path = out_root / "nsd_expdesign.mat"
    if not expdesign_path.exists():
        print("\nTéléchargement de nsd_expdesign.mat ...")
        s3_download(S3_EXPDESIGN, expdesign_path)
    masterordering, subjectim = load_expdesign(expdesign_path)

    print(f"\nTrials dans masterordering : {len(masterordering)}")
    trials_per_session_estimated = len(masterordering) // N_SESSIONS_MAX
    print(f"Estimation trials/session  : {trials_per_session_estimated}")

    # ── Étape 2 : traitement par sujet × session ──────────────────────────────
    for subj_num in args.subjects:
        subject_name = f"subj{subj_num:02d}"
        subject_idx  = subj_num - 1

        print(f"\n{'─'*65}")
        print(f"  Sujet : {subject_name}  (index={subject_idx})")
        print(f"{'─'*65}")

        # Optionnel : auto-détecter les sessions disponibles
        # available = s3_list_sessions(subject_name)
        # sessions_to_run = [s for s in sessions if s in available]

        trial_offset = 0  # compteur cumulatif de trials pour ce sujet
        for session_num in sessions:
            n = process_session(
                subject_name  = subject_name,
                subject_idx   = subject_idx,
                session_num   = session_num,
                masterordering= masterordering,
                subjectim     = subjectim,
                trial_offset  = trial_offset,
                out_root      = out_root,
                dry_run       = args.dry_run,
            )
            trial_offset += n

        print(f"  → {trial_offset} trials traités pour {subject_name}")

    # ── Étape 3 : index CSV global ────────────────────────────────────────────
    if not args.dry_run:
        build_global_index(out_root)

    print(f"\n✅  Terminé. Résultats dans : {out_root.resolve()}")


if __name__ == "__main__":
    main()
