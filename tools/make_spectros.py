# tools/make_spectros.py
import argparse
from pathlib import Path
import numpy as np
import librosa, librosa.display
import soundfile as sf  # assure la lecture de nombreux formats
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------- Paramètres par défaut (peuvent être changés en CLI) ----------------
DEFAULT_SR = 16000
DEFAULT_DUR = 1.0         # secondes (pad/trim pour uniformiser)
DEFAULT_OUT_SIZE = 224    # PNG 224x224
DEFAULT_N_MELS = 64
DEFAULT_N_FFT = 512
DEFAULT_HOP = 160         # ~10ms
DEFAULT_WIN = 400         # ~25ms
DEFAULT_CMAP = "magma"    # joli, produit du RGB

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}

def load_mono_fixed(path, sr, dur):
    """Charge un fichier audio, convertit en mono, resample à sr, puis pad/trim à dur secondes."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    tgt_len = int(sr * dur)
    if len(y) > tgt_len:
        y = y[:tgt_len]
    elif len(y) < tgt_len:
        y = np.pad(y, (0, tgt_len - len(y)))
    return y

def save_png_no_axes(img_array_db, out_png, sr, hop_length, cmap, out_size):
    """Affiche un spectre propre, sans axes, et sauvegarde en PNG."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    dpi = 100
    fig_size = (out_size/100.0, out_size/100.0)  # ex: 2.24x2.24 pouces pour 224px @100dpi
    plt.figure(figsize=fig_size, dpi=dpi)
    plt.axis("off")
    librosa.display.specshow(img_array_db, sr=sr, hop_length=hop_length, cmap=cmap)
    plt.tight_layout(pad=0)
    plt.savefig(str(out_png), bbox_inches="tight", pad_inches=0)
    plt.close()

def to_logmel_png(y, sr, out_png, n_mels, n_fft, hop, win, cmap, out_size):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, hop_length=hop, win_length=win, n_fft=n_fft, power=2.0
    )
    logS = librosa.power_to_db(S, ref=np.max)
    save_png_no_axes(logS, out_png, sr, hop, cmap, out_size)

def to_stft_png(y, sr, out_png, n_fft, hop, win, cmap, out_size):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=win))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    save_png_no_axes(S_db, out_png, sr, hop, cmap, out_size)

def convert_split(audio_split_dir: Path, spectro_split_dir: Path, args):
    """
    Parcourt data/audio/{train|val}/<classe>/*.ext et
    écrit dans data/spectros/{train|val}/<classe>/*.png
    """
    if not audio_split_dir.exists():
        print(f"[WARN] Dossier introuvable: {audio_split_dir}")
        return

    classes = sorted([d for d in audio_split_dir.iterdir() if d.is_dir()])
    if not classes:
        print(f"[WARN] Aucune classe sous: {audio_split_dir}")
        return

    for cls_dir in classes:
        files = sorted([p for p in cls_dir.rglob("*") if p.suffix.lower() in AUDIO_EXTS])
        if not files:
            print(f"[WARN] Classe vide: {cls_dir.name}")
            continue

        print(f"[{audio_split_dir.name}] {cls_dir.name}: {len(files)} fichiers")
        for f in tqdm(files, leave=False):
            rel = f.relative_to(cls_dir)
            out_png = (spectro_split_dir / cls_dir.name / rel).with_suffix(".png")

            try:
                y = load_mono_fixed(str(f), sr=args.sr, dur=args.dur)
                if args.mode == "logmel":
                    to_logmel_png(
                        y, args.sr, out_png,
                        n_mels=args.n_mels, n_fft=args.n_fft, hop=args.hop, win=args.win,
                        cmap=args.cmap, out_size=args.out_size
                    )
                else:
                    to_stft_png(
                        y, args.sr, out_png,
                        n_fft=args.n_fft, hop=args.hop, win=args.win,
                        cmap=args.cmap, out_size=args.out_size
                    )
            except Exception as e:
                print(f"[SKIP] {f} -> {e}")

def main():
    ap = argparse.ArgumentParser(description="Convertit data/audio/train|val en data/spectros/train|val (PNG)")
    ap.add_argument("--in_root", default="data/audio", help="racine des audios (contient train/ et val/)")
    ap.add_argument("--out_root", default="data/spectros", help="racine de sortie (contiendra train/ et val/)")
    ap.add_argument("--mode", choices=["logmel", "stft"], default="logmel", help="type de spectre")
    ap.add_argument("--sr", type=int, default=DEFAULT_SR, help="taux d'échantillonnage cible")
    ap.add_argument("--dur", type=float, default=DEFAULT_DUR, help="durée cible en secondes (pad/trim)")
    ap.add_argument("--out_size", type=int, default=DEFAULT_OUT_SIZE, help="taille PNG (carré, ex 224)")
    ap.add_argument("--n_mels", type=int, default=DEFAULT_N_MELS, help="nb de bandes mel (logmel)")
    ap.add_argument("--n_fft", type=int, default=DEFAULT_N_FFT, help="taille FFT")
    ap.add_argument("--hop", type=int, default=DEFAULT_HOP, help="hop length")
    ap.add_argument("--win", type=int, default=DEFAULT_WIN, help="win length")
    ap.add_argument("--cmap", default=DEFAULT_CMAP, help="colormap matplotlib")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)

    # Train
    convert_split(in_root / "train", out_root / "train", args)
    # Val
    convert_split(in_root / "val", out_root / "val", args)

    print("\n✅ Conversion terminée.")
    print("Exemples attendus :")
    print("  data/spectros/train/go/xxx.png")
    print("  data/spectros/val/stop/yyy.png")

if __name__ == "__main__":
    main()
