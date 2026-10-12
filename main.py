#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["edge-tts>=7"]
# ///
"""
Génère l'enregistrement audio d'une réunion Teams à partir d'un transcript
("Prénom: réplique"), avec les voix neuronales françaises de Microsoft
(via le paquet gratuit edge-tts).

Utilisation (connexion internet requise) :
    uv run main.py

Traite chaque transcript (.txt) du dossier backlog/ : le .mp3 et le .srt sont
écrits dans done/<id>-<nom>/, puis le transcript d'origine y est déplacé.
L'id est celui du nom du fichier ("012-incident.txt") ; à défaut, le prochain
numéro libre dans done/ est attribué.
"""
import argparse
import asyncio
import base64
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import edge_tts
except ImportError:
    print("Installation de edge-tts…")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "edge-tts"])
    import edge_tts

HERE = Path(__file__).resolve().parent
BACKLOG = HERE / "backlog"
DONE = HERE / "done"

# Voix et "personnalité" vocale de chaque participant.
# rate = débit en %, pitch = hauteur en Hz (petits écarts pour bien distinguer les voix).
SPEAKERS = {
    "Claire": {"voice": "fr-FR-DeniseNeural",               "gender": "Female", "rate": 4,  "pitch": 0},
    "Julien": {"voice": "fr-FR-HenriNeural",                "gender": "Male",   "rate": 2,  "pitch": -2},
    "Sophie": {"voice": "fr-FR-VivienneMultilingualNeural", "gender": "Female", "rate": 7,  "pitch": 2},
    "Thomas": {"voice": "fr-FR-RemyMultilingualNeural",     "gender": "Male",   "rate": 5,  "pitch": 0},
    "Marc":   {"voice": "fr-BE-GerardNeural",               "gender": "Male",   "rate": -2, "pitch": -3},
}

# Une trame MP3 silencieuse (24 kHz, mono, 48 kbit/s = même format que edge-tts), 24 ms.
SILENT_FRAME = base64.b64decode(
    "//NkxAAAAANIAAAAAExBTUUzLjEwMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
)
FRAME_SEC = 0.024
BYTES_PER_SEC = 6000  # 48 kbit/s


def silence(seconds: float) -> bytes:
    return SILENT_FRAME * max(0, round(seconds / FRAME_SEC))


def parse(path: Path):
    items = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"\[pause\s+([\d.,]+)\]", line, re.I)
        if m:
            items.append(("pause", float(m.group(1).replace(",", "."))))
            continue
        if line.startswith("["):
            continue  # autre indication de mise en scène
        m = re.match(r"^([A-Za-zÀ-ÿ]+)\s*:\s*(.+)$", line)
        if m and m.group(1) in SPEAKERS:
            items.append(("say", m.group(1), m.group(2)))
        else:
            print(f"Ligne ignorée : {line[:60]}")
    return items


async def check_voices():
    """Remplace une voix si Microsoft l'a retirée, par une autre voix française du même genre."""
    try:
        available = await edge_tts.list_voices()
    except Exception as e:
        print(f"(Impossible de vérifier la liste des voix : {e})")
        return
    names = {v["ShortName"] for v in available}
    used = {s["voice"] for s in SPEAKERS.values()}
    for who, s in SPEAKERS.items():
        if s["voice"] in names:
            continue
        alt = next((v["ShortName"] for v in available
                    if v["Locale"].startswith("fr-") and v["Gender"] == s["gender"]
                    and v["ShortName"] not in used), None)
        if alt:
            print(f"Voix {s['voice']} indisponible pour {who} → {alt}")
            used.add(alt)
            s["voice"] = alt


async def synth(text, spk, rng_seed, sem):
    rng = random.Random(rng_seed)
    s = SPEAKERS[spk]
    rate = s["rate"] + rng.randint(-3, 3)        # légère variation d'une réplique à l'autre
    pitch = s["pitch"] + rng.randint(-1, 1)
    async with sem:
        for attempt in range(4):
            try:
                comm = edge_tts.Communicate(text, s["voice"],
                                            rate=f"{rate:+d}%", pitch=f"{pitch:+d}Hz")
                data = bytearray()
                async for chunk in comm.stream():
                    if chunk["type"] == "audio":
                        data += chunk["data"]
                if data:
                    return bytes(data)
            except Exception as e:
                if attempt == 3:
                    raise
                print(f"  nouvel essai ({e.__class__.__name__})…")
            await asyncio.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Aucun audio reçu pour : {text[:50]}")


def ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


async def main(items, seed):
    rng = random.Random(seed)

    lines = [it for it in items if it[0] == "say"]
    print(f"{len(lines)} répliques, 5 voix. Synthèse en cours…")
    await check_voices()

    sem = asyncio.Semaphore(4)
    done = 0

    async def one(i, it):
        nonlocal done
        clip = await synth(it[2], it[1], seed * 1000 + i, sem)
        done += 1
        print(f"\r  {done}/{len(lines)}", end="", flush=True)
        return clip

    audio = await asyncio.gather(*[one(i, it) for i, it in enumerate(lines)])
    print()
    return audio, rng


def assemble(items, audio, rng):
    out = bytearray(silence(1.0))
    t = len(out) / BYTES_PER_SEC
    srt, idx, k = [], 0, 0
    prev_text = ""
    extra = 0.0
    for it in items:
        if it[0] == "pause":
            extra += it[1]
            continue
        _, spk, text = it
        if k > 0:
            # Pause naturelle entre deux interventions
            if prev_text.rstrip().endswith("?"):
                gap = rng.uniform(0.25, 0.55)     # on répond vite à une question
            elif len(text.split()) <= 4:
                gap = rng.uniform(0.2, 0.45)      # réponses courtes ("Oui.", "Ça marche.")
            else:
                gap = rng.uniform(0.4, 0.85)
            gap += extra
            out += silence(gap)
            t += round(gap / FRAME_SEC) * FRAME_SEC
        extra = 0.0
        clip = audio[k]
        dur = len(clip) / BYTES_PER_SEC
        idx += 1
        srt.append(f"{idx}\n{ts(t)} --> {ts(t + dur)}\n{spk} : {text}\n")
        out += clip
        t += dur
        prev_text = text
        k += 1
    out += silence(1.0)
    return bytes(out), "\n".join(srt), t + 1.0


def meeting_dir(transcript: Path) -> Path:
    """done/<id>-<nom> : l'id vient du nom du fichier, sinon c'est le prochain numéro libre."""
    name = transcript.stem.removeprefix("transcript_")
    if not re.match(r"\d+([-_ ]|$)", name):
        ids = [int(m.group()) for d in DONE.iterdir() if (m := re.match(r"\d+", d.name))]
        name = f"{max(ids, default=0) + 1:03d}-{name}"
    return DONE / name


def process(transcript: Path, seed: int) -> bool:
    """Traite un transcript du backlog ; renvoie False s'il est laissé dans le backlog."""
    dest = meeting_dir(transcript)
    print(f"\n{transcript.name} → done/{dest.name}/")
    if dest.exists():
        print("  Ce dossier existe déjà : transcript laissé dans le backlog.")
        return False
    items = parse(transcript)
    if not any(it[0] == "say" for it in items):
        print("  Aucune réplique reconnue : transcript laissé dans le backlog.")
        return False

    audio, rng = asyncio.run(main(items, seed))
    mp3, srt, total = assemble(items, audio, rng)
    dest.mkdir()
    (dest / f"{dest.name}.mp3").write_bytes(mp3)
    (dest / f"{dest.name}.srt").write_text(srt, encoding="utf-8")
    shutil.move(transcript, dest / transcript.name)
    print(f"Terminé : {dest.name}.mp3 ({int(total // 60)} min {int(total % 60)} s)")
    print(f"Sous-titres : {dest.name}.srt")
    return True


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7, help="change le rythme des pauses")
    args = ap.parse_args()

    DONE.mkdir(exist_ok=True)
    transcripts = sorted(BACKLOG.glob("*.txt"))
    if not transcripts:
        print("Aucun transcript (.txt) dans backlog/.")
        sys.exit(0)

    failed = 0
    for transcript in transcripts:
        try:
            ok = process(transcript, args.seed)
        except Exception as e:
            print(f"  Échec ({e.__class__.__name__} : {e}) : transcript laissé dans le backlog.")
            ok = False
        failed += not ok
    print(f"\n{len(transcripts) - failed}/{len(transcripts)} transcript(s) traité(s).")
    sys.exit(1 if failed else 0)