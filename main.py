"""
Traite tous les transcripts du dossier backlog/ :

    uv run main.py                     # traite tout le backlog
    uv run main.py --seed 3            # autre tirage des pauses, bruits et glitches
    uv run main.py --allow-partial     # accepte un audio où certaines répliques ont échoué
    uv run main.py --redo INC-48770    # renvoie un incident de done/ dans backlog/ et le régénère

Pour chaque transcript backlog/<nom>.txt, le résultat est rangé dans done/<incident>/ :
    done/INC-48770_double_debit/
        transcript_incident_double_debit.txt
        INC-48770_double_debit.mp3
        INC-48770_double_debit.srt
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKLOG = ROOT / "backlog"
DONE = ROOT / "done"
CACHE = ROOT / ".tts_cache"


def incident_name(transcript: Path) -> str:
    """INC-xxxxx (lu dans l'en-tête) + nom du fichier, ex. INC-48770_double_debit."""
    stem = re.sub(r"^(transcript_)?(incident_|reunion_)?", "", transcript.stem) or transcript.stem
    head = "\n".join(transcript.read_text(encoding="utf-8-sig").splitlines()[:15])
    m = re.search(r"\bINC-\d+\b", head)
    name = f"{m.group(0)}_{stem}" if m and m.group(0) not in stem else stem
    return re.sub(r"[^\w\-]+", "_", name).strip("_")


def free_folder(name: str) -> Path:
    target = DONE / name
    n = 2
    while target.exists():
        target = DONE / f"{name}_{n}"
        n += 1
    return target


def redo(key: str) -> None:
    hits = sorted(p for p in DONE.iterdir() if p.is_dir() and key.lower() in p.name.lower()) if DONE.exists() else []
    if not hits:
        sys.exit(f"Aucun dossier de done/ ne correspond à « {key} ».")
    for folder in hits:
        for t in folder.glob("*.txt"):
            shutil.move(str(t), BACKLOG / t.name)
            print(f"↩ {folder.name}/{t.name} remis dans backlog/")
        shutil.rmtree(folder)


def main() -> int:
    ap = argparse.ArgumentParser(description="Simulateur de réunion Teams : backlog/ → done/")
    ap.add_argument("--seed", type=int, default=7, help="change les pauses, bruits et glitches")
    ap.add_argument("--allow-partial", action="store_true",
                    help="range quand même l'incident si certaines répliques n'ont pas pu être générées")
    ap.add_argument("--redo", metavar="INCIDENT", help="régénère un incident déjà dans done/")
    args = ap.parse_args()

    BACKLOG.mkdir(exist_ok=True)
    DONE.mkdir(exist_ok=True)
    if args.redo:
        redo(args.redo)

    todo = sorted(BACKLOG.glob("*.txt"))
    if not todo:
        print("Rien à faire : déposez un transcript (.txt) dans backlog/ puis relancez.")
        return 0

    from simulator import generate  # import ici : le message « rien à faire » reste instantané

    print(f"{len(todo)} transcript(s) dans backlog/.\n")
    ok, kept, errors = [], [], []
    for i, transcript in enumerate(todo, 1):
        name = incident_name(transcript)
        print(f"[{i}/{len(todo)}] {transcript.name}  →  {name}")
        t0 = time.time()
        work = BACKLOG / ".en_cours" / name
        try:
            res = generate(transcript, work / name, seed=args.seed, cache=CACHE)
        except KeyboardInterrupt:
            print("\nInterrompu. Le transcript reste dans backlog/ ; les voix déjà générées sont en cache.")
            shutil.rmtree(work, ignore_errors=True)
            return 130
        except Exception as e:
            print(f"  ✗ Échec : {e.__class__.__name__}: {e}")
            traceback.print_exc(limit=3)
            shutil.rmtree(work, ignore_errors=True)
            errors.append(transcript.name)
            print()
            continue

        if res.failed and not args.allow_partial:
            print(f"  ⚠ {len(res.failed)} réplique(s) manquante(s) : le transcript reste dans backlog/.")
            print("    Relancez plus tard (seules ces répliques seront refaites) ou utilisez --allow-partial.")
            shutil.rmtree(work, ignore_errors=True)
            kept.append(transcript.name)
            print()
            continue

        target = free_folder(name)
        target.mkdir(parents=True)
        for f in (res.audio, res.subtitles):
            shutil.move(str(f), target / f.name)
        shutil.move(str(transcript), target / transcript.name)
        shutil.rmtree(work, ignore_errors=True)
        mins, secs = divmod(int(res.seconds), 60)
        print(f"  ✓ {mins} min {secs:02d} s d'audio, en {time.time() - t0:.0f} s  →  done/{target.name}/\n")
        ok.append(target.name)

    shutil.rmtree(BACKLOG / ".en_cours", ignore_errors=True)
    print("─" * 60)
    print(f"Terminés : {len(ok)}" + (f"  ({', '.join(ok)})" if ok else ""))
    if kept:
        print(f"Restés dans backlog/ (répliques manquantes) : {', '.join(kept)}")
    if errors:
        print(f"En erreur (restés dans backlog/) : {', '.join(errors)}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
