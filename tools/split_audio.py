# tools/split_audio.py
import argparse, random, shutil
from pathlib import Path

def split_class(in_dir, out_train, out_val, val_ratio, move):
    """
    Divise les fichiers d'une classe en train/val selon val_ratio.
    Copie (par défaut) ou déplace (--move) les fichiers.
    """
    # On ne garde que les extensions audio
    exts = {'.wav', '.mp3', '.flac', '.ogg', '.opus'}
    files = sorted([p for p in in_dir.glob("*") if p.is_file() and p.suffix.lower() in exts])
    n = len(files)
    if n == 0:
        print(f"[WARN] Vide: {in_dir}")
        return

    # Calcul du nombre de fichiers validation
    k = int(round(n * val_ratio))
    if n >= 2:
        # au moins 1 en val, au moins 1 en train
        k = max(1, min(k, n - 1))
    else:
        # si un seul fichier -> tout en train
        k = 0

    random.shuffle(files)
    val_files = set(files[:k])

    # Création des dossiers de sortie
    out_train.mkdir(parents=True, exist_ok=True)
    out_val.mkdir(parents=True, exist_ok=True)

    # Choix de l’opération (copie ou déplacement)
    op = shutil.move if move else shutil.copy2

    # Copie/Déplacement des fichiers
    for f in files:
        dst = (out_val if f in val_files else out_train) / f.name
        op(str(f), str(dst))

    print(f"{in_dir.name}: {n-k} train / {k} val")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", default="data/audio/all", help="Dossier contenant toutes les classes")
    ap.add_argument("--out_root", default="data/audio", help="Destination du split")
    ap.add_argument("--val_ratio", type=float, default=0.2, help="Ratio validation ex: 0.2 = 80/20")
    ap.add_argument("--seed", type=int, default=42, help="Graine aléatoire")
    ap.add_argument("--move", action="store_true", help="Déplacer au lieu de copier")
    args = ap.parse_args()

    random.seed(args.seed)
    in_root = Path(args.in_root)
    out_train = Path(args.out_root) / "train"
    out_val   = Path(args.out_root) / "val"

    classes = [d for d in in_root.iterdir() if d.is_dir()]
    for cls in sorted(classes):
        split_class(cls, out_train/cls.name, out_val/cls.name, args.val_ratio, args.move)

if __name__ == "__main__":
    main()
