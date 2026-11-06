"""
Moteur du simulateur de réunion Teams.

Transforme un transcript ("Nom [balises]: texte") en enregistrement audio réaliste :
  - voix neuronales françaises (Microsoft, via edge-tts)
  - prises de parole qui se chevauchent, interruptions, acquiescements en fond
  - micro et environnement propres à chaque participant, bruit de fond, clavier, téléphone
  - problèmes réseau : voix hachée, voix métallique, coupures, latence
  - sons d'arrivée / de départ

Ce module ne se lance pas seul : utilisez main.py (voir README.md).
Le format du transcript est décrit dans FORMAT.md.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import random
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

SR = 24000
SPEECH_RMS = 0.1  # niveau de référence de la parole (-20 dBFS)

# ───────────────────────── Participants ─────────────────────────
# voice      : voix edge-tts (remplacée automatiquement si indisponible)
# rate/pitch : débit (%) et hauteur (Hz) de base
# mic        : headset | earbuds | laptop
# room       : quantité de réverbération de la pièce (0 = sec)
# bg / bg_db : environnement sonore (roomtone, home, fan, openspace, callcenter, train)
#              et niveau relatif à la voix (dB)
# Tout ceci peut être modifié depuis le transcript :
#   [setup Thomas bg=home bg_db=-44 mic=headset net=good]
# net        : qualité réseau de départ : good | medium | poor
# route/fx   : pour les voix de fond (entendues dans le micro d'un participant)
SPEAKERS = {
    "Claire": dict(voice="fr-FR-DeniseNeural", gender="Female", rate=4, pitch=0, gain_db=0,
                   mic="headset", room=0.04, rt60=0.25, bg="roomtone", bg_db=-48, net="good"),
    "Julien": dict(voice="fr-FR-HenriNeural", gender="Male", rate=2, pitch=-2, gain_db=-1,
                   mic="laptop", room=0.30, rt60=0.45, bg="fan", bg_db=-34, net="good"),
    "Sophie": dict(voice="fr-FR-VivienneMultilingualNeural", gender="Female", rate=7, pitch=2, gain_db=0,
                   mic="headset", room=0.10, rt60=0.6, bg="openspace", bg_db=-24, net="good"),
    "Thomas": dict(voice="fr-FR-RemyMultilingualNeural", gender="Male", rate=5, pitch=0, gain_db=0,
                   mic="earbuds", room=0.06, rt60=0.3, bg="train", bg_db=-23, net="medium"),
    "Marc":   dict(voice="fr-BE-GerardNeural", gender="Male", rate=-2, pitch=-3, gain_db=0,
                   mic="headset", room=0.08, rt60=0.5, bg="callcenter", bg_db=-25, net="good"),
    # Voix de fond
    "Kevin":   dict(voice="fr-CH-FabriceNeural", gender="Male", rate=12, pitch=2, gain_db=-7,
                    route="Sophie", fx="distant"),
    "Annonce": dict(voice="fr-CH-ArianeNeural", gender="Female", rate=-6, pitch=0, gain_db=-5,
                    route="Thomas", fx="pa"),
}

DEFAULT_SPEAKERS = copy.deepcopy(SPEAKERS)

MICS = {"headset": (110, 7200), "earbuds": (170, 6800), "laptop": (230, 6000)}

# balise -> (débit %, hauteur Hz, gain dB)
PROSODY = {"stress": (10, 3, 1), "fast": (12, 0, 0), "slow": (-10, 0, 0),
           "calm": (-4, -1, -1), "soft": (-3, -1, -5), "loud": (5, 2, 3)}

# probabilités d'artefacts aléatoires par réplique : (perte légère, perte forte, bande étroite)
NET = {"good": (0.03, 0.0, 0.0), "medium": (0.25, 0.05, 0.05), "poor": (0.45, 0.30, 0.20)}

BABBLE = [
    "Tu as vu le mail de ce matin ? Il paraît que tout est en panne.",
    "Non mais je t'assure, j'ai redémarré trois fois et ça ne marche toujours pas.",
    "Quelqu'un a le numéro du support ? Le client est au téléphone.",
    "On mange où ce midi ? Il y a le nouveau restaurant en bas.",
    "Attends, je te rappelle, je suis en réunion, là.",
    "Oui bonjour madame, je comprends, nous sommes en train de regarder le problème.",
    "C'est la troisième fois cette semaine, franchement.",
    "Tu peux m'envoyer le fichier ? Je ne le trouve pas sur le partage.",
    "Bonjour monsieur, pouvez-vous me donner votre numéro de client, s'il vous plaît ?",
    "D'accord, je note, et je reviens vers vous dès que possible.",
    "Il fait froid ici, quelqu'un a touché à la clim ?",
    "Non, non, c'est pas chez nous, c'est côté serveur, je crois.",
]


def db(x: float) -> float:
    return 10 ** (x / 20)


# ───────────────────────── Synthèse vocale ─────────────────────────
FALLBACK_VOICE = {"Female": "fr-FR-DeniseNeural", "Male": "fr-FR-HenriNeural"}


def simplify(text: str) -> str:
    """Version « sûre » du texte, si le service refuse l'original."""
    t = re.sub(r"[«»“”\"]", "", text).replace("…", ",").replace("—", ".")
    t = re.sub(r"\s*,\s*([,.!?])", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip(" ,")
    return t if re.search(r"\w", t) else "Euh."


class TTS:
    """Synthèse edge-tts avec cache, débit limité et repli automatique en cas d'échec."""

    def __init__(self, cache: Path, concurrency: int = 2):
        import edge_tts  # noqa: F401
        self.cache = cache
        cache.mkdir(exist_ok=True)
        self.sem = asyncio.Semaphore(concurrency)
        self.failed: list[str] = []
        self.degraded: list[str] = []

    async def _fetch(self, text, voice, rate, pitch) -> bytes:
        import edge_tts
        data = bytearray()
        comm = edge_tts.Communicate(text, voice, rate=f"{rate:+d}%", pitch=f"{pitch:+d}Hz")
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                data += chunk["data"]
        if not data:
            raise RuntimeError("audio vide")
        return bytes(data)

    async def say(self, text: str, voice: str, rate: int, pitch: int, gender: str = "Female") -> np.ndarray:
        key = hashlib.sha1(f"{voice}|{rate}|{pitch}|{text}".encode()).hexdigest()[:24]
        f = self.cache / f"{key}.mp3"
        if f.exists() and f.stat().st_size > 0:
            return trim(load_audio(f))

        # Essais successifs : original (x3, avec attente croissante), puis sans prosodie,
        # puis texte simplifié, puis voix de secours.
        alt = FALLBACK_VOICE.get(gender, "fr-FR-DeniseNeural")
        plans = [(text, voice, rate, pitch)] * 3 + [
            (text, voice, 0, 0),
            (simplify(text), voice, 0, 0),
            (simplify(text), alt, rate, pitch),
            (simplify(text), alt, 0, 0),
        ]
        async with self.sem:
            for i, (t, v, r, p) in enumerate(plans):
                try:
                    data = await self._fetch(t, v, r, p)
                    f.write_bytes(data)
                    if i >= 3:
                        self.degraded.append(f"{text[:50]} (repli n°{i - 2})")
                    await asyncio.sleep(0.15)  # on ménage le service
                    return trim(load_audio(f))
                except Exception as e:
                    wait = min(20, 1.5 * 2 ** min(i, 3)) + random.uniform(0, 1)
                    print(f"\n    échec ({e.__class__.__name__}) sur « {text[:40]} » – nouvel essai dans {wait:.0f} s…")
                    await asyncio.sleep(wait)
        # Abandon : on ne plante pas, on laisse un court silence et on le signale.
        self.failed.append(text)
        return np.zeros(int(0.4 * SR), dtype=np.float32)


async def resolve_voices():
    import edge_tts
    try:
        available = await edge_tts.list_voices()
    except Exception as e:
        print(f"(Impossible de vérifier la liste des voix : {e})")
        return []
    names = {v["ShortName"] for v in available}
    used = {s["voice"] for s in SPEAKERS.values()}
    fr = [v for v in available if v["Locale"].startswith("fr-")]
    for who, s in SPEAKERS.items():
        if s["voice"] in names:
            continue
        alt = next((v["ShortName"] for v in fr if v["Gender"] == s["gender"] and v["ShortName"] not in used), None)
        if alt:
            print(f"Voix {s['voice']} indisponible pour {who} → {alt}")
            used.add(alt)
            s["voice"] = alt
    return [v["ShortName"] for v in fr if v["ShortName"] not in used]


def load_audio(path: Path) -> np.ndarray:
    try:
        x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        if not shutil.which("ffmpeg"):
            raise SystemExit("Impossible de décoder le MP3 : mettez à jour soundfile (uv le fait) ou installez ffmpeg.")
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1",
                              "-ar", str(SR), "-"], capture_output=True, check=True).stdout
        return np.frombuffer(raw, dtype=np.float32).copy()
    x = x.mean(axis=1)
    if sr != SR:
        g = gcd(SR, sr)
        x = signal.resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def trim(x: np.ndarray, thresh_db: float = -40, margin: float = 0.04) -> np.ndarray:
    if len(x) == 0:
        return x
    win = int(0.01 * SR)
    env = np.sqrt(np.convolve(x ** 2, np.ones(win) / win, mode="same"))
    idx = np.nonzero(env > db(thresh_db) * max(env.max(), 1e-9))[0]
    if len(idx) == 0:
        return x
    m = int(margin * SR)
    return x[max(0, idx[0] - m): min(len(x), idx[-1] + m)]


# ───────────────────────── Traitement du signal ─────────────────────────
def band(x, lo=None, hi=None, order=2):
    if lo and hi:
        sos = signal.butter(order, [lo, hi], btype="band", fs=SR, output="sos")
    elif lo:
        sos = signal.butter(order, lo, btype="high", fs=SR, output="sos")
    else:
        sos = signal.butter(order, hi, btype="low", fs=SR, output="sos")
    return signal.sosfilt(sos, x).astype(np.float32)


def make_ir(rt60: float, rng: np.random.Generator) -> np.ndarray:
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-6.91 * t / rt60)
    ir = band(ir, hi=5500)
    ir[: int(0.004 * SR)] = 0  # pré-délai
    return (ir / np.sqrt(np.sum(ir ** 2))).astype(np.float32)


def reverb(x, ir, wet):
    if wet <= 0:
        return x
    y = signal.fftconvolve(x, ir)[: len(x) + len(ir)].astype(np.float32)
    out = np.zeros(len(y), dtype=np.float32)
    out[: len(x)] = x * (1 - 0.5 * wet)
    return out + y * wet


def rms(x):
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0


def fade(x, fin=0.0, fout=0.0):
    x = x.copy()
    a, b = int(fin * SR), int(fout * SR)
    if a:
        x[:a] *= np.linspace(0, 1, min(a, len(x)))[: len(x[:a])]
    if b and len(x) > b:
        x[-b:] *= np.linspace(1, 0, b)
    return x


def pink(n, rng):
    f = np.fft.rfft(rng.standard_normal(n))
    k = np.arange(len(f)); k[0] = 1
    y = np.fft.irfft(f / np.sqrt(k), n)
    return (y / (rms(y) + 1e-9)).astype(np.float32)


def brown(n, rng):
    y = np.cumsum(rng.standard_normal(n)).astype(np.float32)
    y = band(y, lo=20)
    return y / (rms(y) + 1e-9)


def loop_to(x, n, rng, xf=0.5):
    """Répète x jusqu'à n échantillons, avec fondus enchaînés."""
    out = np.zeros(n, dtype=np.float32)
    k = int(xf * SR)
    pos = -int(rng.uniform(0, len(x) - 2 * k))
    while pos < n:
        seg = fade(x, xf, xf)
        a, b = max(pos, 0), min(pos + len(seg), n)
        out[a:b] += seg[a - pos: b - pos]
        pos += len(seg) - k
    return out


# ── Artefacts réseau ──
def packet_loss(x, rng, p_enter, p_exit, frame_ms=20):
    """Pertes de paquets en rafales (modèle de Gilbert) + masquage par répétition (son « robot »)."""
    f = int(frame_ms * SR / 1000)
    out = x.copy()
    bad, last, reps = False, None, 0
    edge = np.hanning(2 * int(0.002 * SR))
    h = len(edge) // 2
    for i in range(len(x) // f):
        seg = x[i * f:(i + 1) * f]
        bad = (rng.random() > p_exit) if bad else (rng.random() < p_enter)
        if bad:
            rep = last * (0.65 ** (reps + 1)) if (last is not None and reps < 3) else np.zeros(f, np.float32)
            rep = rep.copy()
            rep[:h] *= edge[:h]; rep[-h:] *= edge[h:]
            out[i * f:(i + 1) * f] = rep
            reps += 1
        else:
            last, reps = seg.copy(), 0
    return out


def stretch_glitch(x, rng):
    """Voix « bloquée » : un petit grain répété (comme quand Teams étire l'audio)."""
    if len(x) < SR:
        return x
    g = int(rng.uniform(0.03, 0.06) * SR)
    p = int(rng.uniform(0.2, 0.8) * (len(x) - SR))
    grain = fade(x[p:p + g], 0.004, 0.004)
    reps = int(rng.integers(5, 10))
    out = x.copy()
    for r in range(reps):
        a = p + r * g
        if a + g > len(out):
            break
        out[a:a + g] = grain * (0.9 ** r)
    return out


def low_bandwidth(x):
    y = signal.resample_poly(x, 1, 3)
    y = np.sign(y) * np.log1p(255 * np.abs(np.clip(y * 2, -1, 1))) / np.log1p(255)  # µ-law
    y = np.round(y * 24) / 24
    y = np.sign(y) * (np.expm1(np.abs(y) * np.log1p(255)) / 255) / 2
    y = signal.resample_poly(y, 3, 1)[: len(x)]
    y = y + 0.35 * np.concatenate([np.zeros(36), y[:-36]])  # léger effet de peigne métallique
    return band(y.astype(np.float32), 300, 3300)


# ── Bruits d'environnement et effets ──
def keyboard(seconds, rng):
    n = int(seconds * SR)
    out = np.zeros(n, dtype=np.float32)
    t = rng.uniform(0, 0.2)
    while t < seconds:
        space = rng.random() < 0.15
        ln = int((0.03 if space else 0.012) * SR)
        click = rng.standard_normal(ln) * np.exp(-np.arange(ln) / ((0.006 if space else 0.002) * SR))
        click = band(click, 300, 1500) * 2 if space else band(click, 1500, 7000)
        a = int(t * SR)
        b = min(n, a + ln)
        out[a:b] += click[: b - a] * rng.uniform(0.5, 1.0)
        t += rng.exponential(0.09) + 0.04
        if rng.random() < 0.07:
            t += rng.uniform(0.3, 1.0)
    return out / (rms(out) + 1e-9)


def desk_phone(rng):
    """Sonnerie électronique de téléphone de bureau (deux salves)."""
    out = []
    for _ in range(2):
        t = np.arange(int(1.0 * SR)) / SR
        f = np.where((t * 20).astype(int) % 2 == 0, 1000, 1250)
        burst = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.6
        out += [fade(burst.astype(np.float32), 0.01, 0.02), np.zeros(int(1.6 * SR), np.float32)]
    return np.concatenate(out)


def chime(kind):
    notes = [659.25, 880.0] if kind == "join" else [880.0, 659.25]
    out = np.zeros(int(0.9 * SR), dtype=np.float32)
    for i, fr in enumerate(notes):
        t = np.arange(int(0.6 * SR)) / SR
        tone = (np.sin(2 * np.pi * fr * t) + 0.25 * np.sin(4 * np.pi * fr * t)) * np.exp(-t / 0.18)
        a = int(i * 0.13 * SR)
        out[a:a + len(tone)] += fade(tone.astype(np.float32), 0.005, 0)
    return out * 0.08


def build_babble(clips, seconds, streams, rng):
    n = int(seconds * SR)
    out = np.zeros(n, dtype=np.float32)
    for _ in range(streams):
        s = np.zeros(n, dtype=np.float32)
        pos = int(rng.uniform(0, 3) * SR)
        while pos < n:
            c = clips[int(rng.integers(len(clips)))] * rng.uniform(0.5, 1.0)
            b = min(n, pos + len(c))
            s[pos:b] += c[: b - pos]
            pos = b + int(rng.uniform(0.3, 2.5) * SR)
        out += s
    out = band(out, 150, 3000)
    out = reverb(out, make_ir(0.9, rng), 0.8)[:n]
    return out / (rms(out) + 1e-9)


def environment(kind, n, rng, babble_clips):
    loop = 60.0
    m = int(loop * SR)
    if kind == "roomtone":
        base = band(pink(m, rng), 60, 4000)
    elif kind == "home":  # bureau à la maison : souffle + ronronnement d'appareil lointain
        t = np.arange(m) / SR
        hum = 0.05 * (np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 200 * t))
        base = band(pink(m, rng), 60, 3500) + 0.4 * band(brown(m, rng), hi=250) + hum.astype(np.float32)
    elif kind == "fan":
        whine = np.sin(2 * np.pi * (1180 + 6 * np.sin(2 * np.pi * 0.3 * np.arange(m) / SR)) * np.arange(m) / SR)
        base = band(pink(m, rng), 150, 2500) + 0.08 * whine.astype(np.float32)
    elif kind == "train":
        t = np.arange(m) / SR
        rumble = band(brown(m, rng), hi=350) * (0.8 + 0.2 * np.sin(2 * np.pi * 0.07 * t))
        wind = band(pink(m, rng), 1500, 6000) * 0.25
        base = (rumble + wind).astype(np.float32)
    elif kind in ("openspace", "callcenter") and babble_clips:
        bab = build_babble(babble_clips, loop * 1.5, 4 if kind == "openspace" else 7, rng)
        base = bab + 0.3 * band(pink(len(bab), rng), 60, 3000)
        if kind == "callcenter":  # téléphones lointains
            for _ in range(3):
                ph = band(desk_phone(rng), hi=2500) * 0.25
                p = int(rng.uniform(0, len(base) - len(ph)))
                base[p:p + len(ph)] += reverb(ph, make_ir(0.8, rng), 0.8)[: len(ph)]
    else:
        base = band(pink(m, rng), 60, 3000)
    base = base / (rms(base) + 1e-9)
    return loop_to(base.astype(np.float32), n, rng)


# ───────────────────────── Transcript ─────────────────────────
@dataclass
class Line:
    idx: int
    spk: str
    text: str
    tags: dict
    clip: np.ndarray | None = None
    start: float = 0.0
    dur: float = 0.0
    trunc: bool = False
    full: float = 0.0
    net: str = "good"
    effects: list = field(default_factory=list)

    @property
    def cfg(self):
        return SPEAKERS[self.spk]

    @property
    def route(self):
        return self.cfg.get("route", self.spk)

    @property
    def end(self):
        return self.start + self.dur


def parse_tags(raw: str) -> dict:
    tags = {}
    for t in [s.strip() for s in raw.split(",") if s.strip()]:
        m = re.fullmatch(r"(>>|&)\s*([\d.]*)", t)
        if m:
            tags["mode"] = m.group(1)
            tags["x"] = float(m.group(2)) if m.group(2) else (0.7 if m.group(1) == ">>" else 0.4)
        elif t == "~":
            tags["mode"] = "~"
        elif m := re.fullmatch(r"(drop|delay)\s+([\d.]+)", t):
            tags[m.group(1)] = float(m.group(2))
        elif t in ("lag", "lowbw") or t in PROSODY:
            tags[t] = True
        else:
            print(f"  balise inconnue ignorée : [{t}]")
    return tags


SETUP_NUM = {"bg_db", "room", "rt60", "rate", "pitch", "gain_db"}
SETUP_KEYS = SETUP_NUM | {"mic", "bg", "net", "voice", "gender"}


def apply_setup(args):
    """[setup Nom cle=valeur ...] : règle micro, environnement, réseau, voix… d'un participant."""
    if not args or args[0] not in SPEAKERS:
        print(f"  [setup] participant inconnu : {' '.join(args)}")
        return
    cfg = SPEAKERS[args[0]]
    for kv in args[1:]:
        k, _, v = kv.partition("=")
        if k not in SETUP_KEYS or not v:
            print(f"  [setup] réglage ignoré : {kv}")
            continue
        cfg[k] = float(v) if k in SETUP_NUM else v
        if k in ("rate", "pitch"):
            cfg[k] = int(cfg[k])


def parse(path: Path):
    events = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if m := re.fullmatch(r"\[(\w+)\s*([^\]]*)\]", line):
            if m.group(1).lower() == "setup":
                apply_setup(m.group(2).split())
                continue
            events.append(("dir", m.group(1).lower(), m.group(2).split()))
            continue
        if m := re.match(r"^([A-Za-zÀ-ÿ]+)\s*(?:\[([^\]]*)\])?\s*:\s*(.+)$", line):
            if m.group(1) in SPEAKERS:
                events.append(("say", Line(len(events), m.group(1), m.group(3), parse_tags(m.group(2) or ""))))
                continue
        print(f"  ligne ignorée : {line[:60]}")
    return events


# ───────────────────────── Mise en place temporelle ─────────────────────────
def layout(events, rng: random.Random):
    cursor, pending = 1.0, 0.0
    last: Line | None = None
    prev_text = ""
    bc_count: dict[int, int] = {}
    net_state = {k: v.get("net", "good") for k, v in SPEAKERS.items()}
    lines, chimes, sfx, open_mic = [], [], [], {}
    open_start = {}

    for ev in events:
        if ev[0] == "dir":
            _, name, args = ev
            t = cursor + pending
            if name == "pause" and args:
                pending += float(args[0].replace(",", "."))
            elif name in ("join", "leave"):
                chimes.append((t + 0.2, name))
                pending += 0.9 if name == "join" else 0.6
            elif name == "mic" and len(args) == 2:
                if args[1] == "open":
                    open_start[args[0]] = t
                elif args[0] in open_start:
                    open_mic.setdefault(args[0], []).append((open_start.pop(args[0]), t))
            elif name == "typing" and len(args) >= 1:
                sfx.append(("typing", args[0], t, float(args[1]) if len(args) > 1 else 4.0))
            elif name == "phone" and args:
                sfx.append(("phone", args[0], t + 0.3, 3.6))
            elif name == "net" and len(args) == 2:
                net_state[args[0]] = args[1]
            else:
                print(f"  indication ignorée : [{name} {' '.join(args)}]")
            continue

        L: Line = ev[1]
        full = len(L.clip) / SR
        L.dur = L.full = full
        if "drop" in L.tags:
            L.dur = min(full, L.tags["drop"])
            L.trunc = True
        delay = L.tags.get("delay", 0.0)
        mode = L.tags.get("mode")

        if mode == "~" and last is not None:
            k = bc_count.get(last.idx, 0)
            bc_count[last.idx] = k + 1
            frac = min(0.85, 0.35 + 0.22 * k + rng.uniform(0, 0.12))
            L.start = last.start + last.dur * frac + delay
            cursor = max(cursor, L.end)
        elif mode in (">>", "&") and last is not None:
            L.start = max(last.start + 0.25, last.end - L.tags["x"]) + pending + delay
            if mode == ">>":
                cut = L.start + rng.uniform(0.25, 0.45)
                if cut < last.end:
                    last.dur = cut - last.start
                    last.trunc = True
            cursor = max(last.end, L.end)
            pending = 0.0
            last = L
        else:
            if prev_text.rstrip().endswith("?"):
                gap = rng.uniform(0.2, 0.5)
            elif len(L.text.split()) <= 4:
                gap = rng.uniform(0.15, 0.4)
            else:
                gap = rng.uniform(0.35, 0.8)
            L.start = cursor + gap + pending + delay
            cursor = L.end
            pending = 0.0
            last = L

        if mode != "~":
            prev_text = L.text
        L.net = net_state.get(L.route, "good") if "route" not in L.cfg else "good"
        lines.append(L)

    end = cursor + pending + 1.5
    for who, t0 in open_start.items():
        open_mic.setdefault(who, []).append((t0, end))
    return lines, chimes, sfx, open_mic, end


# ───────────────────────── Rendu ─────────────────────────
def render(lines, chimes, sfx, open_mic, total, seed, babble_clips):
    nrng = np.random.default_rng(seed)
    n = int(total * SR) + SR
    master = np.zeros(n, dtype=np.float32)
    irs = {k: make_ir(v.get("rt60", 0.4), nrng) for k, v in SPEAKERS.items() if "route" not in v}

    def add(buf, x, t):
        a = int(t * SR)
        if a >= len(buf):
            return
        b = min(len(buf), a + len(x))
        buf[max(a, 0):b] += x[max(0, -a): b - a]

    for spk, cfg in SPEAKERS.items():
        if "route" in cfg:
            continue
        mine = [L for L in lines if L.route == spk]
        my_sfx = [s for s in sfx if s[1] == spk]
        if not mine and not my_sfx:
            continue
        print(f"    piste {spk}…")
        track = np.zeros(n, dtype=np.float32)
        lo, hi = MICS[cfg["mic"]]
        active = []

        for L in mine:
            y = L.clip / (rms(L.clip) + 1e-9) * SPEECH_RMS
            gain = L.cfg.get("gain_db", 0) + sum(PROSODY[t][2] for t in PROSODY if t in L.tags)
            if L.tags.get("mode") == "~":
                gain -= 5
            y = y * db(gain)
            if L.trunc:
                y = fade(y[: int(L.dur * SR)], 0, 0.06 if "drop" not in L.tags else 0.01)
            fx = L.cfg.get("fx")
            if fx == "distant":
                y = reverb(band(y, hi=2800), make_ir(0.9, nrng), 0.75)
            elif fx == "pa":
                y = band(y, 350, 3500)
                y = np.tanh(y * 4) / 4
                y = reverb(y, make_ir(1.6, nrng), 0.85)
            y = band(y, lo, hi)
            y = reverb(y, irs[spk], cfg.get("room", 0))
            add(track, y, L.start)
            active.append((L.start - 0.15, L.end + 0.45))

        for kind, _, t, dur in my_sfx:
            if kind == "typing":
                k = keyboard(dur, nrng) * SPEECH_RMS * db(-14 if cfg["mic"] == "laptop" else -22)
                add(track, reverb(band(k, lo, hi), irs[spk], cfg.get("room", 0)), t)
            else:
                ph = band(desk_phone(nrng), hi=3500) * SPEECH_RMS * db(-8)
                add(track, reverb(band(ph, lo, hi), make_ir(0.7, nrng), 0.7), t)
            active.append((t, t + dur + 0.3))
        active += open_mic.get(spk, [])

        # bruit de fond, transmis seulement quand le micro « s'ouvre »
        mask = np.zeros(n, dtype=np.float32)
        for a, b in active:
            mask[max(0, int(a * SR)): max(0, int(b * SR))] = 1
        w = np.hanning(int(0.12 * SR)); w /= w.sum()
        mask = signal.fftconvolve(mask, w, mode="same").astype(np.float32)
        bg = environment(cfg["bg"], n, nrng, babble_clips) * SPEECH_RMS * db(cfg["bg_db"])
        track += band(bg, lo, hi) * mask

        # problèmes réseau sur les répliques du participant
        for L in mine:
            if "route" in L.cfg:
                continue
            light, heavy, narrow = NET.get(L.net, NET["good"])
            a = max(0, int((L.start - 0.05) * SR))
            b = min(n, int((L.end + 0.35) * SR))
            seg = track[a:b]
            if "lowbw" in L.tags or nrng.random() < narrow:
                seg = low_bandwidth(seg)
            if "lag" in L.tags or nrng.random() < heavy:
                seg = packet_loss(seg, nrng, 0.05, 0.22)
                seg = stretch_glitch(seg, nrng)
            elif nrng.random() < light:
                seg = packet_loss(seg, nrng, 0.012, 0.45)
            if "drop" in L.tags:
                c = int((L.dur - 0.25) * SR)
                if c > 0:
                    seg[c:] = packet_loss(seg[c:], nrng, 0.3, 0.1)
                seg[int(L.dur * SR):] = 0
                track[b: min(n, b + int(0.6 * SR))] = 0
            track[a:b] = seg[: b - a]

        master += track
        del track, mask, bg

    for t, kind in chimes:
        add(master, chime(kind) * 0.9, t)

    # léger souffle de fond (évite le silence numérique absolu)
    floor = band(pink(30 * SR, nrng), 80, 6000)
    master += loop_to(floor / (rms(floor) + 1e-9), len(master), nrng) * SPEECH_RMS * db(-56)

    # limiteur doux
    th = 0.8
    over = np.abs(master) > th
    master[over] = np.sign(master[over]) * (th + (1 - th) * np.tanh((np.abs(master[over]) - th) / (1 - th)))
    return master[: int(total * SR)] * 0.92


def ts(sec: float) -> str:
    ms = int(round(max(sec, 0) * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(lines, path: Path):
    out = []
    for i, L in enumerate(sorted(lines, key=lambda l: l.start), 1):
        who = L.spk
        if "route" in L.cfg:
            who = f"({L.spk}, en fond chez {L.route})"
        elif L.tags.get("mode") == "~":
            who += " (en fond)"
        txt = L.text + (" [coupé]" if L.full - L.dur > 0.5 else "")
        out.append(f"{i}\n{ts(L.start)} --> {ts(L.end)}\n{who} : {txt}\n")
    path.write_text("\n".join(out), encoding="utf-8")


def save_audio(y, base: Path) -> Path:
    mp3 = base.with_suffix(".mp3")
    try:
        sf.write(str(mp3), y, SR, format="MP3")
        return mp3
    except Exception:
        pass
    wav = base.with_suffix(".wav")
    sf.write(str(wav), y, SR, subtype="PCM_16")
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-b:a", "96k", str(mp3)], check=True)
        wav.unlink()
        return mp3
    return wav


async def synthesize_all(lines, tts: TTS, seed: int, spare_voices):
    done = 0

    async def one(L: Line):
        nonlocal done
        r = random.Random(seed * 1000 + L.idx)
        cfg = L.cfg
        rate = cfg["rate"] + r.randint(-3, 3) + sum(PROSODY[t][0] for t in PROSODY if t in L.tags)
        pitch = cfg["pitch"] + r.randint(-1, 1) + sum(PROSODY[t][1] for t in PROSODY if t in L.tags)
        L.clip = await tts.say(L.text, cfg["voice"], rate, pitch, cfg.get("gender", "Female"))
        done += 1
        print(f"\r    voix {done}/{len(lines)}", end="", flush=True)

    await asyncio.gather(*[one(L) for L in lines])
    print()

    babble = []
    need = any(SPEAKERS[s].get("bg") in ("openspace", "callcenter")
               for s in {L.route for L in lines})
    if need and spare_voices:
        print("    bruit de fond (conversations)…")
        vs = spare_voices
        clips = await asyncio.gather(*[tts.say(t, vs[i % len(vs)], random.Random(i).randint(-5, 10), 0)
                                       for i, t in enumerate(BABBLE)])
        babble = [c / (rms(c) + 1e-9) for c in clips if len(c)]
    return babble


@dataclass
class Result:
    audio: Path
    subtitles: Path
    seconds: float
    lines: int
    failed: list
    degraded: list


async def _synthesize(transcript: Path, seed: int, cache: Path):
    events = parse(transcript)
    lines = [e[1] for e in events if e[0] == "say"]
    print(f"  {len(lines)} répliques. Vérification des voix…")
    spare = await resolve_voices()
    tts = TTS(cache)
    babble = await synthesize_all(lines, tts, seed, spare)
    return events, lines, babble, tts


def generate(transcript: Path, out_base: Path, seed: int = 7, cache: Path | None = None) -> Result:
    """Génère <out_base>.mp3 (ou .wav) et <out_base>.srt à partir d'un transcript."""
    # chaque transcript repart des réglages par défaut ([setup] ne déborde pas sur le suivant)
    SPEAKERS.clear()
    SPEAKERS.update(copy.deepcopy(DEFAULT_SPEAKERS))
    cache = cache or transcript.parent / ".tts_cache"

    events, lines, babble, tts = asyncio.run(_synthesize(transcript, seed, cache))
    if tts.degraded:
        print(f"  {len(tts.degraded)} réplique(s) en mode dégradé (prosodie, texte ou voix simplifiés) :")
        for d in tts.degraded:
            print(f"    - {d}")
    if tts.failed:
        print(f"  ⚠ {len(tts.failed)} réplique(s) impossible(s) à générer (remplacées par un silence) :")
        for t in tts.failed:
            print(f"    - {t[:70]}")

    print("  Mise en place et mixage…")
    placed, chimes, sfx, open_mic, total = layout(events, random.Random(seed))
    audio = render(placed, chimes, sfx, open_mic, total, seed, babble)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    out = save_audio(audio, out_base)
    srt = out_base.with_suffix(".srt")
    write_srt(placed, srt)
    return Result(out, srt, total, len(lines), list(tts.failed), list(tts.degraded))
