#!/usr/bin/env python3
"""
AutoStepperPy – modern Python DDR/ITG-style chart generator for StepMania.

Key behavior:

- Uses librosa for BPM & beat detection + onset strength.
- Always generates up to 5 charts (Beginner, Easy, Medium, Hard, Challenge),
  with optional difficulty selection from the CLI.
- Difficulty (meter) is dynamic per song:
    * Estimated from steps-per-minute and then clamped
      into ranges per tier, then made monotonic
      (Beginner ≤ Easy ≤ Medium ≤ Hard ≤ Challenge).
- Beat grid is based on beat INDEX (0,1,2,...) assuming steady 4/4,
  which matches how StepMania uses BPM + OFFSET and tends to feel
  more “on-beat” when dancing than noisy timestamps.

Rhythm grid:

- Uses a full 16-row grid per measure:
    * 4ths (on-beats): rows 0,4,8,12
    * 8ths: rows 2,6,10,14
    * 16ths: everything else
  Beat strengths are propagated to 8ths/16ths so higher difficulties
  can form DDR-style streams and patterns.

Gimmicks:

- Holds:
    * Present on ALL difficulties (can be disabled via CLI).
    * Very rare & short on Beginner/Easy.
    * More frequent and longer on Medium/Hard/Challenge.
- Mines:
    * Only on Hard and Challenge (can be disabled via CLI).

I/O behavior:

- Supports .mp3/.wav/.ogg/.flac/.mp4.
- For audio files:
    <output>/<SongTitle>/
        <SongTitle>.<ext>   (trimmed or copied audio)
        <SongTitle>.sm      (charts)
- For .mp4 files:
    <output>/<SongTitle>/
        <SongTitle>.avi or <SongTitle>.mp4 (BACKGROUND video, configurable)
        <SongTitle>_bg.png (static 640x480 screenshot used as #BACKGROUND)
        <SongTitle>_bn.png (static 256x80 screenshot used as #BANNER)
        <SongTitle>.ogg     (extracted audio, used as MUSIC)
        <SongTitle>.sm

Long songs:

- Default analysis length is 90 seconds (classic DDR style).
- If a song is longer than 90s:
    * You are prompted:
        - [y]es  → use full length for THIS song
        - [a]ll  → use full length for ALL long songs this run
        - [n]o   → keep 90s clip for this song
- The actual used length is stored in #MUSICLENGTH and #SONGTYPE.
- For "Long"/"Marathon" songs (>=150s), patterns include periodic
  “rest windows” with fewer 8ths/16ths to give the player micro breaks.
  Very long songs (>=240s) get slightly more frequent rest windows.

Video behavior:

- For .mp4 input you can choose the background video format:
    * bg_format="avi" (default): transcode MP4 → AVI via ffmpeg for BACKGROUND
      which can reduce lag in some StepMania builds.
    * bg_format="mp4": copy MP4 directly as BACKGROUND.
- In both cases, the script also grabs a centered screenshot from a
  musically meaningful time (near the first chorus / "heart" of the song),
  using beat detection to pick the frame. This screenshot is used as:
    * #BACKGROUND (640x480)
    * #BANNER (256x80)

Timing & BPM:

- Beat timing is locked to librosa’s beat detection.
- Beat strengths use beat-synchronous energy between beats rather than a
  single onset frame sample, for more stable “strong/weak beat” detection.
- A smoothed instantaneous tempo estimate is used to generate multi-segment
  #BPMS (e.g., "0.000=140.000,64.000=150.000").  Short jittery changes are
  ignored so only real tempo shifts become segments.
- Optional CLI overrides allow manual BPM and OFFSET if you want full control.

Parallelism:

- New CLI arg --workers lets you use multiple CPU cores:
    * --workers 1   → sequential (default)
    * --workers 4   → up to 4 processes in parallel
    * --workers -1  → use all available cores

CLI via Python Fire.

Typical usage:

    pip install fire librosa soundfile numpy
    # optional (for real trimming & mp4 audio extraction):
    pip install pydub  # requires ffmpeg installed on your system

    # Process all supported audio/video files in current folder
    python AutoStepper.py

    # Use 4 workers for faster batch processing
    python AutoStepper.py generate --workers=4

    # Use all cores
    python AutoStepper.py generate --workers=-1
"""

import os
import sys
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import fire
import builtins

try:
    import librosa
except ImportError:
    print(
        "This script requires the 'librosa' package. Install it with:\n"
        "    pip install librosa soundfile numpy"
    )
    sys.exit(1)

# Optional: used to trim audio to `limit_seconds` and extract audio from mp4.
try:
    from pydub import AudioSegment
    HAVE_PYDUB = True
except ImportError:
    HAVE_PYDUB = False

SUPPORTED_EXTS = (".mp3", ".wav", ".ogg", ".flac", ".mp4")


@dataclass
class ChartMetadata:
    title: str
    artist: str
    music_filename: str
    offset: float
    bpm: float
    length_seconds: float = 0.0
    background_filename: str = ""   # static BG image (or video fallback)
    banner_filename: str = ""       # BANNER image


@dataclass
class StepChart:
    description: str           # e.g. "AutoStepperPy"
    difficulty: str            # Beginner, Easy, Medium, Hard, Challenge
    meter: int                 # numeric difficulty / footer
    notes: List[str]           # list of measures (each 16 lines of "0000"/"1M23"...)


def find_audio_files(input_path: str) -> List[str]:
    """Return list of audio/video file paths from a file or directory (non-recursive)."""
    if os.path.isfile(input_path):
        lower = input_path.lower()
        return [input_path] if lower.endswith(SUPPORTED_EXTS) else []

    files: List[str] = []
    for entry in os.listdir(input_path):
        full = os.path.join(input_path, entry)
        if os.path.isfile(full) and entry.lower().endswith(SUPPORTED_EXTS):
            files.append(full)
    return files


def detect_beats(
    path: str,
    limit_seconds: Optional[float] = None,
) -> Tuple[int, float, np.ndarray, np.ndarray, float]:
    """
    Load audio and detect beats.

    Returns:
        sr            : sample rate
        tempo_float   : BPM as float (global tempo estimate)
        beat_times    : np.ndarray of beat times (seconds)
        beat_strength : np.ndarray of per-beat onset strengths (relative energy)
        duration      : float, duration actually analyzed (seconds)
    """
    print(f"  Loading audio for beat detection: {path}")

    # Lower sample rate for speed; duration to analyze only the first N seconds
    if limit_seconds is not None and limit_seconds > 0:
        y, sr = librosa.load(path, sr=22050, mono=True, duration=limit_seconds)
    else:
        y, sr = librosa.load(path, sr=22050, mono=True)

    duration = librosa.get_duration(y=y, sr=sr)
    print(f"  Duration used for analysis: {duration:.2f}s at {sr} Hz")

    # Beat tracking
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Onset envelope -> strength per beat (energy over each beat region)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    if len(onset_env) == 0 or len(beat_frames) == 0:
        strengths = np.zeros_like(beat_times)
    else:
        max_idx = len(onset_env) - 1
        vals: List[float] = []
        for i, bf in enumerate(beat_frames):
            if i == 0:
                left = bf
            else:
                left = 0.5 * (beat_frames[i - 1] + bf)
            if i + 1 < len(beat_frames):
                right = 0.5 * (bf + beat_frames[i + 1])
            else:
                right = bf

            start = int(max(0, np.floor(min(left, right))))
            end = int(min(max_idx, np.ceil(max(left, right))))
            if end <= start:
                start = max(0, int(bf) - 1)
                end = min(max_idx, int(bf) + 1)

            seg = onset_env[start : end + 1]
            vals.append(float(seg.mean()) if seg.size else 0.0)
        strengths = np.asarray(vals, dtype=float)

    # Ensure tempo is a plain float
    try:
        tempo_value = float(tempo[0]) if hasattr(tempo, "__len__") else float(tempo)
    except Exception:
        tempo_value = float(tempo)

    print(
        f"  Estimated tempo: {tempo_value:.2f} BPM, "
        f"beats detected: {len(beat_times)}"
    )
    return sr, tempo_value, beat_times, strengths, float(duration)


def choose_screenshot_time(
    beat_times: np.ndarray,
    song_seconds: float,
    default_time: float = 10.0,
) -> float:
    """
    Choose a good screenshot time for BG/BN:

    - If we have beat information, pick a beat near ~25–30% into the song
      (often close to the first chorus / 'heart' of the track).
    - If beat data is missing, fall back to a fraction of the song length
      or the provided default_time.
    - Clamp so we don't land right at the very start or end.
    """
    if song_seconds is None or song_seconds <= 0:
        return default_time

    # No beats? Just pick ~25% into the song, but not too close to the end.
    if beat_times is None or len(beat_times) == 0:
        t = song_seconds * 0.25
        t = max(1.0, min(t, max(1.0, song_seconds - 3.0)))
        return t

    # Aim for a point around 28% into the track.
    target = song_seconds * 0.28

    # Find the beat closest to that target time.
    idx = int(np.argmin(np.abs(beat_times - target)))
    t = float(beat_times[idx])

    # Safety: if something weird happens, fall back.
    if not np.isfinite(t):
        t = default_time

    # Clamp to a nice range (a little after the start, a bit before the end).
    t = max(3.0, min(t, max(3.0, song_seconds - 3.0)))
    return t


def quantize_beats_to_grid(
    beat_times: np.ndarray,
    beat_strengths: np.ndarray,
    bpm: float,
    subdivision: int = 4,
) -> Tuple[Dict[int, Dict[int, float]], float]:
    """
    Convert beats into a StepMania-style 4-beat-per-measure grid.

    IMPORTANT: instead of using exact beat times (which can be jittery),
    we treat beats as an INDEXED grid:

        beat 0 → measure 0, row 0      (first beat of measure)
        beat 1 → measure 0, row 4      (second beat)
        beat 2 → measure 0, row 8
        beat 3 → measure 0, row 12
        beat 4 → measure 1, row 0
        ...

    This assumes 4/4 timing and constant BPM, which matches how StepMania
    renders charts and tends to feel more “on-beat” when dancing.

    Returns:
        grid: measure_index -> { row_index (0..15) -> strength }
        max_strength: global maximum strength (for normalization)
    """
    n_beats = len(beat_times)
    if n_beats == 0:
        return {}, 0.0

    rows_per_beat = subdivision  # 4 rows per beat
    rows_per_measure = 4 * rows_per_beat

    grid: Dict[int, Dict[int, float]] = {}
    max_strength = 0.0

    for i in range(n_beats):
        s = float(beat_strengths[i]) if i < len(beat_strengths) else 0.0
        if s <= 0.0:
            continue

        measure_idx = i // 4
        beat_in_measure = i % 4
        row_in_beat = 0  # always put base beats on the first row of that beat
        row_index = beat_in_measure * rows_per_beat + row_in_beat
        row_index = max(0, min(rows_per_measure - 1, row_index))

        measure_rows = grid.setdefault(measure_idx, {})
        old = measure_rows.get(row_index, 0.0)
        strength = float(max(old, s))
        measure_rows[row_index] = strength
        if strength > max_strength:
            max_strength = strength

    return grid, max_strength


def _row_class(row_index: int) -> str:
    """Classify row index as 'on' (4th), 'half' (8th), or 'sixteenth' (16th)."""
    if row_index % 4 == 0:
        return "on"           # quarter notes (main beats)
    if row_index % 2 == 0:
        return "half"         # eighth notes
    return "sixteenth"        # sixteenth notes


def pick_step_positions_for_difficulty(
    grid: Dict[int, Dict[int, float]],
    max_strength: float,
    difficulty_name: str,
    total_beats: int,
    bpm: float,
    hard_flag: bool,
    rng: np.random.Generator,
) -> Tuple[Set[Tuple[int, int]], int]:
    """
    Decide where to place steps (positions) for a given difficulty.

    Uses a 16-row grid per measure, synthesizing 8th and 16th positions
    from the nearest base beat strength so higher difficulties can form
    DDR-style streams and patterns.

    BPM-aware behavior:
      - Reference BPM = 140.
      - Slow songs (<140) get slightly more density.
      - Fast songs (>140) get slightly less density, especially 16ths.

    Returns:
        positions: set of (measure_index, row_index)
        target_steps: the targeted number of arrows (approx, 1 per row)
    """
    positions: List[Tuple[int, int, float]] = []
    measure_energy: Dict[int, float] = {}
    rows_per_measure = 16

    # Build candidate positions on the full 16-row grid
    for m in sorted(grid.keys()):
        row_strengths = grid[m]
        if not row_strengths:
            continue

        # Base beat rows (we only put beats on one row per beat)
        base_rows = [r for r, s in row_strengths.items() if s > 0]
        if not base_rows:
            continue

        for r in range(rows_per_measure):
            if r in row_strengths:
                s = float(row_strengths[r])
            else:
                # Synthesize a strength from nearest base beat row
                nearest = min(base_rows, key=lambda rr: abs(rr - r))
                base_s = float(row_strengths.get(nearest, 0.0))
                if base_s <= 0.0:
                    continue

                rc = _row_class(r)
                if rc == "on":
                    s = base_s
                elif rc == "half":        # 8ths slightly weaker
                    s = base_s * 0.7
                else:                     # 16ths weaker still
                    s = base_s * 0.4

            if s > 0.0:
                positions.append((m, r, s))
                prev = measure_energy.get(m, 0.0)
                if s > prev:
                    measure_energy[m] = s

    if not positions:
        return set(), 0

    safe_max_strength = max_strength if max_strength > 0 else 1.0
    max_me = max(measure_energy.values()) if measure_energy else 1.0

    # Base difficulty factors – used only for density, NOT for meter.
    base_cfg = {
        "Beginner":  {"factor": 0.30, "min_steps": 40},
        "Easy":      {"factor": 0.60, "min_steps": 80},
        "Medium":    {"factor": 0.80, "min_steps": 120},
        "Hard":      {"factor": 1.40, "min_steps": 260},
        "Challenge": {"factor": 1.80, "min_steps": 300},
    }

    if difficulty_name not in base_cfg:
        raise ValueError(f"Unknown difficulty_name: {difficulty_name}")

    cfg = base_cfg[difficulty_name].copy()

    # If --hard is enabled, bump high difficulties a bit more
    if hard_flag and difficulty_name in ("Hard", "Challenge"):
        cfg["factor"] *= 1.15

    # BPM-aware scaling around reference BPM
    ref_bpm = 140.0
    clamped_bpm = max(80.0, min(bpm if bpm > 0 else ref_bpm, 200.0))
    bpm_weight = ref_bpm / clamped_bpm  # <1 for fast songs, >1 for slow songs
    bpm_weight = max(0.6, min(1.5, bpm_weight))

    factor = cfg["factor"] * bpm_weight
    min_steps = cfg["min_steps"]

    # Target step count for this chart, approximating 1 row = 1 step
    target_steps = int(max(total_beats * factor, min_steps))

    # Max possible rows = all candidate positions (16 per measure with strength)
    max_possible = len(positions)
    if target_steps > max_possible:
        target_steps = max_possible

    # Base probabilities per difficulty and row class
    slow = bpm > 0 and bpm < 115.0
    fast = bpm > 165.0

    if difficulty_name == "Beginner":
        probs = {
            "on":        (0.40, 0.50),
            "half":      (0.00, 0.00),
            "sixteenth": (0.00, 0.00),
        }
    elif difficulty_name == "Easy":
        probs = {
            "on":        (0.55, 0.45),
            "half":      (0.20, 0.35),
            "sixteenth": (0.00, 0.20),
        }
    elif difficulty_name == "Medium":
        probs = {
            "on":        (0.58, 0.38),
            "half":      (0.32, 0.32),
            "sixteenth": (0.06, 0.18),  # moderated 16ths
        }
    elif difficulty_name == "Hard":
        probs = {
            "on":        (0.65, 0.40),
            "half":      (0.45, 0.35),
            "sixteenth": (0.20, 0.30),
        }
    else:  # Challenge
        probs = {
            "on":        (0.70, 0.35),
            "half":      (0.50, 0.35),
            "sixteenth": (0.30, 0.30),
        }

    # BPM-based tweak: adjust 16ths a bit by tempo
    if fast:
        # At high BPM, dial back 16ths across all diffs
        b, k = probs["sixteenth"]
        probs["sixteenth"] = (b * 0.7, k * 0.7)
    elif slow and difficulty_name in ("Medium", "Hard", "Challenge"):
        # At low BPM, bump 16ths slightly for interest
        b, k = probs["sixteenth"]
        probs["sixteenth"] = (min(b * 1.3, 0.40), min(k * 1.3, 0.40))

    # Gallop probabilities – occasional only
    gallop_prob_map = {
        "Beginner": 0.0,
        "Easy": 0.0,
        "Medium": 0.04,
        "Hard": 0.08,
        "Challenge": 0.10,
    }
    gallop_prob = gallop_prob_map.get(difficulty_name, 0.0)

    chosen: List[Tuple[int, int, float]] = []

    # First pass: probabilistic selection based on strength & rhythm class
    for (m, r, s) in positions:
        norm = float(s) / safe_max_strength
        rc = _row_class(r)
        base_p, k = probs[rc]

        # Energy weighting by measure so choruses hit harder
        me = measure_energy.get(m, 0.0)
        if max_me > 0.0:
            rel_me = me / max_me
        else:
            rel_me = 1.0
        energy_weight = 0.4 + 0.6 * rel_me  # 0.4–1.0

        p = (base_p + k * norm) * energy_weight
        p = max(0.0, min(1.0, p))

        if rng.random() < p:
            chosen.append((m, r, s))

            # Optional gallop: add a neighbor 16th around selected on-beats
            if gallop_prob > 0.0 and rc == "on":
                gallop_chance = gallop_prob * (0.7 + 0.6 * norm)
                gallop_chance = min(1.0, gallop_chance)
                if rng.random() < gallop_chance:
                    neighbor = r + 1 if r + 1 < rows_per_measure else r - 1
                    if 0 <= neighbor < rows_per_measure:
                        chosen.append((m, neighbor, s * 0.5))

    # Adjust density to be near target_steps
    chosen_count = len(chosen)
    lower = int(0.7 * target_steps)
    upper = int(1.3 * target_steps)

    # If too few, add from remaining strongest positions
    if chosen_count < lower:
        chosen_set = {(m, r) for (m, r, _) in chosen}
        remaining = [(m, r, s) for (m, r, s) in positions if (m, r) not in chosen_set]
        remaining.sort(key=lambda x: x[2], reverse=True)  # strongest first

        for (m, r, s) in remaining:
            if len(chosen) >= lower:
                break
            chosen.append((m, r, s))

    # If too many, randomly drop some
    if len(chosen) > upper:
        rng.shuffle(chosen)
        chosen = chosen[:upper]

    final_positions: Set[Tuple[int, int]] = {(m, r) for (m, r, _) in chosen}
    return final_positions, target_steps


def _add_holds_and_mines(
    grid: List[List[List[int]]],
    difficulty_label: str,
    rng: np.random.Generator,
    allow_holds: bool = True,
    allow_mines: bool = True,
):
    """
    Post-process the tap grid to add holds and mines.

    Coding:
        0 = empty
        1 = tap
        2 = hold head
        3 = hold tail
        9 = mine

    Strategy:
        - Holds on ALL difficulties (unless disabled):
            * Beginner/Easy: rare, short (1 beat).
            * Medium: modest, short.
            * Hard: more frequent, 1–2 beats.
            * Challenge: frequent, up to 3 beats.
          Holds generally start on strong (4th) beats.
        - Mines only on Hard and Challenge (unless disabled), and are placed
          as "don't be here" cues near existing arrows rather than random spam.
    """
    n_measures = len(grid)
    if n_measures == 0:
        return

    rows_per_measure = 16
    total_rows = n_measures * rows_per_measure

    # ---- Hold configuration per difficulty ----
    if allow_holds:
        if difficulty_label == "Beginner":
            hold_prob = 0.02
            beat_lengths = [4]         # 1 beat
        elif difficulty_label == "Easy":
            hold_prob = 0.04
            beat_lengths = [4]
        elif difficulty_label == "Medium":
            hold_prob = 0.06
            beat_lengths = [4]
        elif difficulty_label == "Hard":
            hold_prob = 0.10
            beat_lengths = [4, 8]      # 1–2 beats
        else:  # Challenge
            hold_prob = 0.12
            beat_lengths = [4, 8, 12]  # up to 3 beats
    else:
        hold_prob = 0.0
        beat_lengths = []

    # candidate starts = taps on ON-beat rows only (stronger beats)
    candidates = []
    if hold_prob > 0.0:
        for m in range(n_measures):
            for r in range(rows_per_measure):
                rc = _row_class(r)
                if rc != "on":
                    continue
                for lane in range(4):
                    if grid[m][r][lane] == 1:  # tap
                        g_index = m * rows_per_measure + r
                        candidates.append((g_index, lane))

        rng.shuffle(candidates)

        for g_index, lane in candidates:
            if rng.random() >= hold_prob:
                continue

            length_rows = int(rng.choice(beat_lengths))
            start_idx = g_index
            end_idx = g_index + length_rows
            if end_idx >= total_rows:
                continue

            start_m = start_idx // rows_per_measure
            start_r = start_idx % rows_per_measure
            end_m = end_idx // rows_per_measure
            end_r = end_idx % rows_per_measure

            # Check lane is free of other holds/mines; taps in between will be cleared.
            conflict = False
            for idx in range(start_idx, end_idx + 1):
                m = idx // rows_per_measure
                r = idx % rows_per_measure
                val = grid[m][r][lane]
                if val in (2, 3, 9):  # already head/tail/mine
                    conflict = True
                    break
            if conflict:
                continue

            # Place hold: head & tail, clear intermediate taps on this lane
            for idx in range(start_idx + 1, end_idx):
                m = idx // rows_per_measure
                r = idx % rows_per_measure
                grid[m][r][lane] = 0  # clear

            grid[start_m][start_r][lane] = 2  # head
            grid[end_m][end_r][lane] = 3      # tail

    # ---- Mines (Hard and Challenge only) ----
    if not allow_mines:
        return

    if difficulty_label == "Hard":
        mine_prob = 0.02
    elif difficulty_label == "Challenge":
        mine_prob = 0.03
    else:
        mine_prob = 0.0

    if mine_prob <= 0.0:
        return

    for m in range(n_measures):
        for r in range(rows_per_measure):
            # Only place mines on rows that already have at least one arrow
            lanes_with_arrows = [
                lane for lane in range(4) if grid[m][r][lane] in (1, 2, 3)
            ]
            if not lanes_with_arrows:
                continue

            # Determine "don't be here" lanes near current arrows
            empty_lanes = [lane for lane in range(4) if grid[m][r][lane] == 0]
            if not empty_lanes:
                continue

            if len(lanes_with_arrows) == 1:
                base_lane = lanes_with_arrows[0]
                candidate_lanes = [
                    lane for lane in empty_lanes if lane != base_lane
                ]
            elif len(lanes_with_arrows) == 2:
                # For jumps, warn on the non-jump lanes
                candidate_lanes = empty_lanes
            else:
                candidate_lanes = empty_lanes

            if not candidate_lanes:
                continue

            if rng.random() < mine_prob:
                lane = int(rng.choice(candidate_lanes))
                grid[m][r][lane] = 9  # mine


def render_chart_from_positions(
    positions: Set[Tuple[int, int]],
    difficulty_label: str,
    meter: int,
    jump_aggressiveness: float,
    seed: int = 1234,
    rest_measure_cycle: Optional[int] = None,
    allow_holds: bool = True,
    allow_mines: bool = True,
) -> StepChart:
    """
    Turn a set of (measure,row) positions into an actual 4-panel chart.

    This version emphasizes DDRMAX-style body movement and fun:

    - Uses different lane pattern sets per difficulty:
        * Beginner/Easy: simpler, mostly front-facing and side steps.
        * Medium+: adds more crossovers and sweeping patterns.
    - 1/4 and many 1/8 notes follow these patterns to create recognizable
      streams and crossovers.
    - 16ths are shorter "flicks" around the previous lane.
    - Jacks and jumps still apply but are moderated to preserve flow.
    - For long songs, periodic "rest windows" reduce density,
      especially 16ths, so players get micro-breathers without killing flow.
    - A simple "footedness" model helps avoid ugly doublesteps by preferring
      side-to-side alternation when it won't break patterns.
    """
    rows_per_measure = 16

    if not positions:
        empty_measure = "\n".join(["0000"] * rows_per_measure)
        return StepChart(
            description="AutoStepperPy",
            difficulty=difficulty_label,
            meter=meter,
            notes=[empty_measure],
        )

    max_measure = max(m for (m, _) in positions)

    # Initialize numeric grid: measures x rows x lanes
    grid: List[List[List[int]]] = [
        [[0 for _ in range(4)] for _ in range(rows_per_measure)]
        for _ in range(max_measure + 1)
    ]

    rng = np.random.default_rng(seed)

    # Jack probabilities – tuned for "medium/random" feel, Medium softened
    jack_prob_map = {
        "Beginner": 0.0,
        "Easy": 0.01,
        "Medium": 0.03,
        "Hard": 0.12,
        "Challenge": 0.16,
    }
    jack_prob = jack_prob_map.get(difficulty_label, 0.0)

    # Base movement patterns (0=L,1=D,2=U,3=R)
    # We'll choose different subsets based on difficulty.
    simple_patterns = [
        [0, 1, 2, 3],  # L D U R – clockwise sweep
        [3, 2, 1, 0],  # R U D L – counter-clockwise sweep
        [1, 2, 1, 2],  # D U D U – front-facing pivots
        [0, 3, 0, 3],  # L R L R – side shuffles
    ]
    crossover_patterns = [
        [0, 2, 3, 1],  # L U R D – crossover through U
        [3, 1, 0, 2],  # R D L U – crossover through D
        [0, 3, 2, 1],  # L R U D – twisty pattern
        [1, 0, 2, 3],  # D L U R
    ]

    if difficulty_label in ("Beginner", "Easy"):
        allowed_patterns = simple_patterns
    elif difficulty_label == "Medium":
        allowed_patterns = simple_patterns + crossover_patterns[:2]
    else:  # Hard / Challenge
        allowed_patterns = simple_patterns + crossover_patterns

    neighbors_by_lane = {
        0: [1, 2, 3],  # from L you can go D/U/R
        1: [0, 2, 3],  # from D -> any other
        2: [0, 1, 3],  # from U -> any other
        3: [0, 1, 2],  # from R -> any other
    }

    current_pattern = allowed_patterns[0]
    pattern_pos = 0

    last_lane: Optional[int] = None
    last_global_row: Optional[int] = None
    last_row_class: Optional[str] = None
    last_foot: Optional[str] = None
    jack_run_remaining = 0  # how many extra same-lane hits to force

    # Simple "foot side" mapper
    def lane_to_side(lane: int) -> str:
        # 0=L,1=D → left-ish; 2,3 → right-ish
        return "L" if lane in (0, 1) else "R"

    # Phrase motifs: keep a consistent pattern per 4-measure phrase
    phrase_patterns: Dict[int, List[int]] = {}

    def get_phrase_pattern(measure_idx: int) -> List[int]:
        phrase_idx = measure_idx // 4
        if phrase_idx not in phrase_patterns:
            phrase_patterns[phrase_idx] = allowed_patterns[
                int(rng.integers(0, len(allowed_patterns)))
            ]
        return phrase_patterns[phrase_idx]

    # Work positions in strict time order
    ordered_positions = sorted(list(positions))  # (m, r)

    # How likely are we to start a brand new pattern on a strong beat?
    # Higher diffs = more deliberate and continuous patterns.
    if difficulty_label == "Beginner":
        start_pattern_prob = 0.60
    elif difficulty_label == "Easy":
        start_pattern_prob = 0.45
    elif difficulty_label == "Medium":
        start_pattern_prob = 0.35
    else:  # Hard / Challenge
        start_pattern_prob = 0.25

    # Long song rest-window tuning
    # For rest_measure_cycle == 16, treat measures 12–13 as "breathers".
    # For 12, treat 8–9 as breathers. None → no rest windows.
    def is_rest_measure(m: int) -> bool:
        if rest_measure_cycle is None:
            return False
        cycle = rest_measure_cycle
        if cycle <= 0:
            return False
        idx = m % cycle
        # Use last two measures of each phrase as micro-rests
        return idx in (cycle - 4, cycle - 3)

    # Skip probabilities inside rest windows by difficulty & row class.
    rest_skip_config = {
        "Beginner": {
            "on": 0.30,
            "half": 0.80,
            "sixteenth": 1.00,
        },
        "Easy": {
            "on": 0.25,
            "half": 0.75,
            "sixteenth": 0.95,
        },
        "Medium": {
            "on": 0.20,
            "half": 0.65,
            "sixteenth": 0.95,
        },
        "Hard": {
            "on": 0.15,
            "half": 0.60,
            "sixteenth": 0.90,
        },
        "Challenge": {
            "on": 0.10,
            "half": 0.55,
            "sixteenth": 0.90,
        },
    }
    rest_skips = rest_skip_config.get(difficulty_label, rest_skip_config["Medium"])

    for (m, r) in ordered_positions:
        rc = _row_class(r)
        global_row = m * rows_per_measure + r

        # Long-song rest window?
        if is_rest_measure(m):
            skip_prob = rest_skips.get(rc, 0.0)
            if rng.random() < skip_prob:
                # Skip this potential arrow entirely to create breathing room.
                continue

        # --- Lane selection with patterns + jacks ---
        if jack_run_remaining > 0 and last_lane is not None:
            # Continue an existing jack run
            lane_idx = last_lane
            jack_run_remaining -= 1
        else:
            # Are we in a "local stream"? (row spacing small)
            in_stream = (
                last_global_row is not None
                and (global_row - last_global_row) <= 2
            )

            if rc in ("on", "half"):
                if rc == "on":
                    # Often (re)start a pattern on strong beats.
                    # Tie to phrase motif so each 4-measure block has a theme.
                    if pattern_pos == 0 or rng.random() < start_pattern_prob:
                        current_pattern = get_phrase_pattern(m)
                        pattern_pos = 0
                    lane_idx = current_pattern[pattern_pos % len(current_pattern)]
                    pattern_pos += 1
                else:
                    # 1/8 notes: frequently continue the current pattern to
                    # create real DDR-style 1/4–1/8 streams, but with some
                    # chance to weave around the last foot.
                    if rng.random() < 0.6:
                        lane_idx = current_pattern[pattern_pos % len(current_pattern)]
                        pattern_pos += 1
                    elif last_lane is not None:
                        lane_idx = int(rng.choice(neighbors_by_lane[last_lane]))
                    else:
                        lane_idx = int(rng.integers(0, 4))
            else:
                # 16ths: short flicks around the previous foot.
                # If we're in a dense stream, prefer near-neighbors to keep
                # twisting realistic instead of wild teleports.
                if last_lane is not None:
                    # Slightly tighter neighborhood during streams
                    if in_stream and rng.random() < 0.8:
                        lane_idx = int(rng.choice(neighbors_by_lane[last_lane]))
                    elif rng.random() < 0.6:
                        lane_idx = int(rng.choice(neighbors_by_lane[last_lane]))
                    else:
                        lane_idx = int(rng.integers(0, 4))
                else:
                    lane_idx = int(rng.integers(0, 4))

            # Chance to *start* a jack run from this step
            # (avoid starting jacks on dense 16th bursts too often)
            if rc != "sixteenth" and rng.random() < jack_prob:
                jack_run_remaining = int(rng.integers(1, 3))  # 1–2 extra hits

        # Footedness / doublestep mitigation
        foot = lane_to_side(lane_idx)
        if last_lane is not None and last_foot is not None and foot == last_foot:
            # Sometimes reroll to a lane on the opposite side to avoid
            # staying stuck on one foot, as long as it doesn't break flow.
            if rng.random() < 0.5:
                alt_candidates = [
                    n for n in neighbors_by_lane[last_lane]
                    if lane_to_side(n) != last_foot and grid[m][r][n] == 0
                ]
                if alt_candidates:
                    lane_idx = int(rng.choice(alt_candidates))
                    foot = lane_to_side(lane_idx)
        last_foot = foot

        # Place base tap
        grid[m][r][lane_idx] = 1
        last_lane = lane_idx
        last_global_row = global_row
        last_row_class = rc

        # --- Jump logic – occasional doubles on higher diffs ---
        if jump_aggressiveness > 0.0:
            if rc == "on":
                jp = jump_aggressiveness * 1.1
            elif rc == "half":
                jp = jump_aggressiveness * 0.7
            else:
                jp = jump_aggressiveness * 0.4

            if rng.random() < jp:
                zero_idxs = [i for i, v in enumerate(grid[m][r]) if v == 0]
                if zero_idxs:
                    extra_lane = int(rng.choice(zero_idxs))
                    grid[m][r][extra_lane] = 1  # add a tap in other lane

    # Add holds and mines in-place
    _add_holds_and_mines(grid, difficulty_label, rng, allow_holds=allow_holds, allow_mines=allow_mines)

    # Convert numeric grid → .sm lines
    measures: List[str] = []
    for m in range(max_measure + 1):
        lines: List[str] = []
        for r in range(rows_per_measure):
            chars: List[str] = []
            for lane_val in grid[m][r]:
                if lane_val == 0:
                    chars.append("0")
                elif lane_val == 1:
                    chars.append("1")
                elif lane_val == 2:
                    chars.append("2")  # hold head
                elif lane_val == 3:
                    chars.append("3")  # hold tail
                elif lane_val == 9:
                    chars.append("M")  # mine
                else:
                    chars.append("0")
            lines.append("".join(chars))
        measures.append("\n".join(lines))

    return StepChart(
        description="AutoStepperPy",
        difficulty=difficulty_label,
        meter=meter,
        notes=measures,
    )


def write_sm_file(
    meta: ChartMetadata,
    charts: List[StepChart],
    output_path: str,
    bpm_changes: Optional[Dict[float, float]] = None,
):
    """Write a StepMania .sm file for the given song metadata and charts."""
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    bpm = float(meta.bpm)
    offset = float(meta.offset)
    song_len = float(meta.length_seconds) if meta.length_seconds > 0 else 0.0

    # Build multi-BPM string from bpm_changes if provided
    if bpm_changes and len(bpm_changes) > 1:
        parts = []
        bpm_vals = []
        for beat_pos in sorted(bpm_changes.keys()):
            bpm_val = bpm_changes[beat_pos]
            bpm_vals.append(bpm_val)
            parts.append(f"{beat_pos:.3f}={bpm_val:.6f}")
        bpms_str = ",".join(parts)
        min_bpm = min(bpm_vals)
        max_bpm = max(bpm_vals)
    else:
        bpms_str = f"0.000={bpm:.6f}"
        min_bpm = max_bpm = bpm

    # DISPLAYBPM tag: show min:max if there is a real range
    if abs(max_bpm - min_bpm) < 0.5:
        display_bpm_str = f"{min_bpm:.3f}"
    else:
        display_bpm_str = f"{min_bpm:.3f}:{max_bpm:.3f}"

    # Classify song length for informational tags
    if song_len >= 210.0:
        songtype = "Marathon"
    elif song_len >= 150.0:
        songtype = "Long"
    else:
        songtype = "Regular"

    lines: List[str] = []
    # Header modeled after classic StepMania/AutoStepper .sm style
    lines.append(f"#TITLE:{meta.title} ;")
    lines.append("#SUBTITLE:;")
    lines.append("#ARTIST:AutoStepperPy;")
    lines.append("#TITLETRANSLIT:;")
    lines.append("#SUBTITLETRANSLIT:;")
    lines.append("#ARTISTTRANSLIT:;")
    lines.append("#GENRE:;")
    lines.append("#CREDIT:AutoStepperPy;")
    if meta.banner_filename:
        lines.append(f"#BANNER:{meta.banner_filename};")
    else:
        lines.append("#BANNER:;")
    if meta.background_filename:
        lines.append(f"#BACKGROUND:{meta.background_filename};")
    else:
        lines.append("#BACKGROUND:;")
    lines.append("#LYRICSPATH:;")
    lines.append("#CDTITLE:;")
    lines.append(f"#MUSIC:{meta.music_filename};")
    lines.append(f"#OFFSET:{offset:.8f};")
    # Keep sample preview around 30 seconds in
    lines.append("#SAMPLESTART:30.0;")
    lines.append("#SAMPLELENGTH:30.0;")
    lines.append("#SELECTABLE:YES;")
    lines.append(f"#BPMS:{bpms_str};")
    lines.append(f"#DISPLAYBPM:{display_bpm_str};")
    lines.append("#STOPS:;")
    lines.append("#KEYSOUNDS:;")
    lines.append("#ATTACKS:;")
    if song_len > 0:
        lines.append(f"#MUSICLENGTH:{song_len:.3f};")
        lines.append(f"#SONGTYPE:{songtype};")
    lines.append("")

    for chart in charts:
        lines.append("#NOTES:")
        lines.append("     dance-single:")
        # SM order: description, difficulty, meter, radar.
        lines.append(f"     {chart.description}:")
        lines.append(f"     {chart.difficulty}:")
        lines.append(f"     {chart.meter}:")
        # Radar values – left 0; StepMania can recompute if needed
        lines.append("     0.000,0.000,0.000,0.000,0.000:")
        for i, m in enumerate(chart.notes):
            lines.append(m)
            if i != len(chart.notes) - 1:
                lines.append(",")
            else:
                lines.append(";")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  Wrote SM file: {output_path}")


def count_steps_in_chart(chart: StepChart) -> int:
    """
    Rough step count: number of notes (taps+holds+mines).
    Counts '1', '2', '3', 'M' as one each.
    """
    count = 0
    for measure in chart.notes:
        for line in measure.splitlines():
            for ch in line:
                if ch in ("1", "2", "3", "M"):
                    count += 1
    return count


def assign_meters(
    charts_with_counts: List[Tuple[StepChart, int]],
    song_seconds: float,
    bpm: float,
    meter_bias: int = 0,
):
    """
    Assign numeric meters based on steps-per-minute and pattern complexity,
    then enforce monotonic meters across Beginner→Challenge.

    Calibrated roughly against DDRMAX-style charts:

    - Base meter uses a ~30 SPM-per-foot scale:
        meter_base ≈ (SPM - 25) / 30
      (so 200 SPM ≈ 6–7 feet, 280 SPM ≈ 8–9 feet, 400+ SPM ≈ 10 feet).

    - Complexity bonus (0–2) looks at:
        * ratio of 16th-note rows
        * longest stream of consecutive note rows
    """

    if song_seconds <= 0:
        song_seconds = 90.0

    # Difficulty → allowed meter band (based on DDRMAX patterns)
    bands = {
        "Beginner":  (1, 3),   # 1–3 feet
        "Easy":      (2, 5),   # 2–5 feet
        "Medium":    (4, 8),   # 4–8 feet
        "Hard":      (7, 10),  # 7–10 feet
        "Challenge": (8, 10),  # 8–10 feet
    }

    def analyze_pattern_complexity(chart: StepChart) -> int:
        """
        Very lightweight pattern heuristic for complexity bonus:

        - total_note_rows: rows that contain at least one note (1/2/3/M).
        - sixteenth_rows: how many of those rows are on 16th positions.
        - longest_stream: longest run of consecutive note rows.

        Returns a bonus in [0, 2]:
          +1 if there are noticeable 16ths or long streams.
          +2 if there are heavy 16ths AND long streams (boss-song-esque).
        """
        total_note_rows = 0
        sixteenth_rows = 0
        longest_stream = 0
        current_stream = 0

        for measure in chart.notes:
            lines = measure.splitlines()
            for row_index, line in enumerate(lines):
                has_note = any(ch in ("1", "2", "3", "M") for ch in line)
                if has_note:
                    total_note_rows += 1
                    current_stream += 1

                    # In a 16-row measure, odd indexes are 16ths, even are
                    # 8ths/quarters. We treat odd indexes as "16th-ish".
                    if row_index % 2 == 1:
                        sixteenth_rows += 1
                else:
                    if current_stream > longest_stream:
                        longest_stream = current_stream
                    current_stream = 0

        if current_stream > longest_stream:
            longest_stream = current_stream

        if total_note_rows == 0:
            return 0

        ratio_16 = sixteenth_rows / float(total_note_rows)

        bonus = 0

        # Lots of 16ths → more technical
        if ratio_16 > 0.20:
            bonus += 1
        if ratio_16 > 0.40:
            bonus += 1

        # Long streams → stamina
        if longest_stream >= 16:
            bonus += 1

        # Cap bonus so we don't get silly values
        return min(bonus, 2)

    # --- Step 1: initial meters from SPM + complexity ---
    for chart, steps in charts_with_counts:
        # Steps per minute (taps + holds + mines already counted)
        spm = steps * 60.0 / song_seconds

        # Base DDRMAX-like scale: about ~30 SPM per "foot"
        # Shifted a bit so very easy songs still land at 1–2.
        base_float = (spm - 25.0) / 30.0
        base_meter = int(round(base_float))
        base_meter = max(1, min(10, base_meter))

        complexity_bonus = analyze_pattern_complexity(chart)
        raw_meter = base_meter + complexity_bonus

        # Clamp into difficulty-specific band
        lo, hi = bands.get(chart.difficulty, (1, 10))
        meter = max(lo, min(hi, raw_meter))
        chart.meter = meter

    # --- Step 2: enforce monotonicity (Beginner ≤ Easy ≤ Medium ≤ Hard ≤ Challenge) ---
    order = ["Beginner", "Easy", "Medium", "Hard", "Challenge"]
    diff_to_chart = {c.difficulty: c for (c, _) in charts_with_counts}

    last_meter = 1
    for diff in order:
        chart = diff_to_chart.get(diff)
        if not chart:
            continue
        if chart.meter < last_meter:
            chart.meter = last_meter
        last_meter = chart.meter

    # --- Step 3: apply global meter bias and re-enforce monotonicity ---
    if meter_bias != 0:
        for chart, _ in charts_with_counts:
            lo, hi = bands.get(chart.difficulty, (1, 10))
            chart.meter = max(lo, min(hi, chart.meter + meter_bias))

        last_meter = 1
        for diff in order:
            chart = diff_to_chart.get(diff)
            if not chart:
                continue
            if chart.meter < last_meter:
                chart.meter = last_meter
            last_meter = chart.meter


def trim_or_copy_audio(src: str, dest: str, limit_seconds: float):
    """
    Create the song audio file inside the song folder.

    - If pydub is available, actually trims to `limit_seconds`.
    - Otherwise, falls back to copying the full file.
    """
    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if limit_seconds is None or limit_seconds <= 0:
        shutil.copy2(src, dest)
        print(f"  Copied full audio (no duration limit) -> {dest}")
        return

    if HAVE_PYDUB:
        try:
            audio = AudioSegment.from_file(src)
            trimmed = audio[: int(limit_seconds * 1000)]
            ext = os.path.splitext(dest)[1].lstrip(".") or "mp3"
            trimmed.export(dest, format=ext)
            print(f"  Trimmed audio to {limit_seconds:.1f}s -> {dest}")
            return
        except Exception as e:
            print(f"  Warning: trimming failed ({e!r}), copying full audio instead.")

    shutil.copy2(src, dest)
    print(f"  Copied full audio (no trimming library available) -> {dest}")


def get_full_duration(path: str) -> Optional[float]:
    """Best-effort full duration detection for prompting about long songs."""
    try:
        # librosa can read many formats directly (including mp3/ogg, often mp4)
        return float(librosa.get_duration(path=path))
    except Exception:
        if HAVE_PYDUB:
            try:
                audio = AudioSegment.from_file(path)
                return float(audio.duration_seconds)
            except Exception:
                return None
        return None


def _transcode_mp4_to_avi(src: str, dest: str) -> bool:
    """
    Use ffmpeg to transcode an MP4 to AVI for StepMania background.
    Returns True on success, False on failure.
    """
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        # Simple, broadly compatible video-only transcode, drop audio (-an)
        cmd = [
            "ffmpeg",
            "-y",
            "-i", src,
            "-c:v", "mpeg4",
            "-q:v", "5",
            "-an",
            dest,
        ]
        print(f"  Transcoding MP4 to AVI for BACKGROUND:\n    {' '.join(cmd)}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print("  ffmpeg error while creating AVI background:")
            print(result.stderr.decode(errors="ignore"))
            return False
        print(f"  Created AVI background: {dest}")
        return True
    except FileNotFoundError:
        print("  ffmpeg not found on PATH; cannot transcode MP4 to AVI.")
    except Exception as e:
        print(f"  Unexpected error during MP4→AVI transcode: {e!r}")
    return False


def _extract_video_screenshot(
    src: str,
    dest: str,
    time_sec: float = 10.0,
    width: int = 640,
    height: int = 480,
) -> bool:
    """
    Grab a single frame from the video as a PNG and scale it to the given
    width x height, letterboxed, suitable for StepMania BG/BANNER use.

    We call this twice for .mp4 songs:
      - once as 640x480 for the BG image
      - once as 256x80 for the BN image

    Returns True on success, False otherwise.
    """
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        vf_filter = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
        )
        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(time_sec),
            "-i", src,
            "-vframes", "1",
            "-vf",
            vf_filter,
            dest,
        ]
        print(f"  Taking video screenshot for {os.path.basename(dest)}:\n    {' '.join(cmd)}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print("  ffmpeg error while grabbing screenshot:")
            print(result.stderr.decode(errors="ignore"))
            return False
        print(f"  Created screenshot image: {dest}")
        return True
    except FileNotFoundError:
        print("  ffmpeg not found on PATH; cannot grab screenshot.")
    except Exception as e:
        print(f"  Unexpected error during screenshot capture: {e!r}")
    return False


def build_bpm_changes(
    beat_times: np.ndarray,
    default_bpm: float,
    diff_threshold: float = 3.0,
    min_run_beats: int = 4,
) -> Dict[float, float]:
    """
    Build a simple multi-BPM map from instantaneous beat intervals.

    Returns:
        { beat_position (float) -> bpm (float) }

    beat_position is in quarter-note units from song start (0.0, 1.0, 2.0, ...).

    We start with an initial BPM at beat 0, then whenever the smoothed
    instantaneous BPM changes by >= diff_threshold for at least
    `min_run_beats` beats, we add a new BPM segment.
    """
    n = len(beat_times)
    if n < 2:
        return {0.0: float(default_bpm) if default_bpm > 0 else 120.0}

    # Instantaneous BPMs between each consecutive pair of beats
    inst_bpms: List[float] = []
    for i in range(n - 1):
        dt = float(beat_times[i + 1] - beat_times[i])
        if dt <= 0:
            inst_bpms.append(float(default_bpm) if default_bpm > 0 else 120.0)
        else:
            inst_bpms.append(60.0 / dt)

    inst_bpms_arr = np.asarray(inst_bpms, dtype=float)

    # Simple 5-beat median smoothing to reduce jitter
    if len(inst_bpms_arr) >= 5:
        smoothed = inst_bpms_arr.copy()
        for i in range(len(inst_bpms_arr)):
            left = max(0, i - 2)
            right = min(len(inst_bpms_arr), i + 3)
            smoothed[i] = float(np.median(inst_bpms_arr[left:right]))
        inst_bpms_arr = smoothed

    bpms: Dict[float, float] = {}

    current_bpm = float(inst_bpms_arr[0])
    bpms[0.0] = current_bpm
    last_change_beat = 0.0

    beat_pos = 1.0
    for i in range(1, len(inst_bpms_arr)):
        lbpm = float(inst_bpms_arr[i])
        if abs(lbpm - current_bpm) >= diff_threshold and (beat_pos - last_change_beat) >= float(min_run_beats):
            current_bpm = lbpm
            bpms[beat_pos] = current_bpm
            last_change_beat = beat_pos
        beat_pos += 1.0

    # Round for nicer output
    rounded = {round(b, 3): round(v, 3) for b, v in bpms.items()}
    return rounded


def process_file(
    path: str,
    output_dir: str,
    duration: float,
    synctime: float,
    hard: bool,
    tap: bool,
    tapsync: float,  # kept for CLI parity; unused
    bg_format: str = "avi",
    difficulties: Optional[List[str]] = None,
    seed: int = 999,
    allow_mines: bool = True,
    allow_holds: bool = True,
    meter_bias: int = 0,
    bpm_override: float = 0.0,
    offset_override: Optional[float] = None,
):
    """
    Process one audio or video file into a StepMania-ready song folder.

    Folder layout:
        <output_dir>/<SongTitle>/
            <SongTitle>.<ext>   (trimmed or copied audio, or extracted audio)
            <SongTitle>.sm      (generated chart)
            [optional] <SongTitle>.avi or .mp4 as BACKGROUND for videos
            [optional] <SongTitle>_bg.png as #BACKGROUND image
            [optional] <SongTitle>_bn.png as #BANNER image

    bg_format:
        - "avi": transcode MP4 to AVI via ffmpeg for BACKGROUND (default).
        - "mp4": copy MP4 as-is for BACKGROUND.

    difficulties:
        - List of difficulty names to generate, subset of:
          ["Beginner", "Easy", "Medium", "Hard", "Challenge"].
          If None, all five are generated.
    """
    filename = os.path.basename(path)
    title_no_ext, ext = os.path.splitext(filename)
    ext = ext.lower()

    bg_format = (bg_format or "avi").lower()

    # Per-song directory: <output>/<SongTitle>
    song_dir = os.path.join(output_dir, title_no_ext)
    os.makedirs(song_dir, exist_ok=True)

    background_filename = ""
    banner_filename = ""

    # Determine how we'll produce audio + background
    if ext == ".mp4":
        # Video case:
        #   - BACKGROUND VIDEO: AVI (default) or MP4 (depending on bg_format)
        #   - MUSIC: extracted audio as .ogg (if possible)
        if bg_format == "mp4":
            bg_video_filename = filename
            bg_video_dest_path = os.path.join(song_dir, bg_video_filename)
            if os.path.abspath(path) != os.path.abspath(bg_video_dest_path):
                print(f"  Copying video (BACKGROUND): {bg_video_dest_path}")
                shutil.copy2(path, bg_video_dest_path)
        else:
            # Default: transcode to AVI for smoother playback
            bg_video_filename = f"{title_no_ext}.avi"
            bg_video_dest_path = os.path.join(song_dir, bg_video_filename)
            ok = _transcode_mp4_to_avi(path, bg_video_dest_path)
            if not ok:
                # Fallback to original MP4 if transcode fails
                bg_video_filename = filename
                bg_video_dest_path = os.path.join(song_dir, bg_video_filename)
                if os.path.abspath(path) != os.path.abspath(bg_video_dest_path):
                    print(f"  Falling back: copying MP4 (BACKGROUND): {bg_video_dest_path}")
                    shutil.copy2(path, bg_video_dest_path)

        # Static BG/BANNER filenames from video (actual screenshots created later,
        # after beat detection so we can choose a musically meaningful timestamp).
        bg_image_filename = f"{title_no_ext}_bg.png"
        bg_image_path = os.path.join(song_dir, bg_image_filename)
        bn_image_filename = f"{title_no_ext}_bn.png"
        bn_image_path = os.path.join(song_dir, bn_image_filename)

        # We'll decide on the exact screenshot time after beat detection.
        # For now, leave background_filename/banner_filename empty; they will be
        # set once screenshots are generated. If screenshots fail later, we will
        # fall back to using the video file as background.

        # Audio file name to use as #MUSIC
        audio_filename = f"{title_no_ext}.ogg"
        audio_dest_path = os.path.join(song_dir, audio_filename)

        if HAVE_PYDUB:
            try:
                audio = AudioSegment.from_file(path)
                if duration is not None and duration > 0:
                    audio = audio[: int(duration * 1000)]
                audio.export(audio_dest_path, format="ogg")
                print(f"  Extracted audio from mp4 -> {audio_dest_path}")
            except Exception as e:
                print(
                    f"  Warning: mp4 audio extraction failed ({e!r}), "
                    "falling back to using the video file as MUSIC."
                )
                audio_filename = bg_video_filename
                audio_dest_path = bg_video_dest_path
        else:
            print(
                "  Warning: pydub/ffmpeg not available, cannot extract audio from mp4.\n"
                "           Using the video file itself as #MUSIC (if StepMania build supports it)."
            )
            audio_filename = bg_video_filename
            audio_dest_path = bg_video_dest_path

        analysis_path = audio_dest_path
    else:
        # Audio-only case: create trimmed/copied audio into song directory
        audio_filename = filename
        audio_dest_path = os.path.join(song_dir, audio_filename)
        if os.path.abspath(path) != os.path.abspath(audio_dest_path):
            print(f"  Creating song audio: {audio_dest_path}")
        trim_or_copy_audio(path, audio_dest_path, duration)
        analysis_path = audio_dest_path
        background_filename = ""
        banner_filename = ""

    # Metadata: human-friendly title, StepMania #MUSIC uses audio_filename
    meta = ChartMetadata(
        title=title_no_ext,
        artist="AutoStepperPy",
        music_filename=audio_filename,
        offset=float(synctime),  # updated below using first beat time
        bpm=120.0,               # temporary, overwritten after detection
        length_seconds=0.0,
        background_filename=background_filename,
        banner_filename=banner_filename,
    )

    if tap:
        print("Tap mode is not implemented yet; using automatic beat detection.\n")

    # Beat detection (only analyzing first `duration` seconds of analysis_path)
    _, detected_bpm, beat_times, strengths, duration_used = detect_beats(
        analysis_path,
        limit_seconds=duration,
    )

    # Apply BPM override if provided
    if bpm_override and bpm_override > 0.0:
        bpm = float(bpm_override)
        print(f"  BPM override in effect: {detected_bpm:.2f} → {bpm:.2f}")
    else:
        bpm = float(detected_bpm)

    meta.bpm = bpm

    # We'll allow fallback to synthetic beats if detection fails
    effective_beat_times = beat_times
    effective_strengths = strengths

    # Auto-align to first detected beat to improve sync, unless offset override supplied
    total_beats_detected = len(beat_times)
    if offset_override is not None:
        meta.offset = float(offset_override)
    else:
        if total_beats_detected > 0:
            first_beat_time = float(beat_times[0])
            # StepMania OFFSET: time (in seconds) subtracted from chart timing.
            # We want the first beat to be at chart time 0 + synctime.
            meta.offset = float(synctime) - first_beat_time
        else:
            meta.offset = float(synctime)

    # Quantize beats to grid (beat rows only, index-based)
    grid, max_strength = quantize_beats_to_grid(
        effective_beat_times,
        effective_strengths,
        bpm,
        subdivision=4
    )

    # Fallback: if beat detection failed, create a simple 4/4 skeleton chart
    if not grid or len(effective_beat_times) == 0:
        print("  Warning: no beats detected – falling back to simple metronome chart.")
        if bpm <= 0:
            bpm = 120.0
            meta.bpm = bpm
        spb = 60.0 / bpm
        fallback_beats = max(int(duration_used / spb), 32)
        fake_times = np.arange(0.0, fallback_beats * spb, spb, dtype=float)
        fake_strengths = np.ones_like(fake_times)
        effective_beat_times = fake_times
        effective_strengths = fake_strengths
        grid, max_strength = quantize_beats_to_grid(
            effective_beat_times,
            effective_strengths,
            bpm,
            subdivision=4
        )

    total_beats = len(effective_beat_times)

    # Approximate song duration for meter scaling & long-song behavior
    song_seconds = duration_used if duration_used and duration_used > 0 else (
        total_beats * 60.0 / bpm if bpm > 0 else (duration or 90.0)
    )
    meta.length_seconds = song_seconds

    # For video songs, now that we know the approximate song length and beat grid,
    # choose a musically meaningful time for BG/BN screenshots (e.g., near the
    # first chorus) instead of a fixed 10s mark.
    if ext == ".mp4":
        screenshot_time = choose_screenshot_time(
            effective_beat_times,
            song_seconds,
            default_time=10.0,
        )
        print(f"  Picking screenshot around {screenshot_time:.2f}s for BG/BN images.")

        bg_ok = _extract_video_screenshot(
            path,
            bg_image_path,
            time_sec=screenshot_time,
            width=640,
            height=480,
        )
        bn_ok = _extract_video_screenshot(
            path,
            bn_image_path,
            time_sec=screenshot_time,
            width=256,
            height=80,
        )

        if bg_ok:
            background_filename = bg_image_filename
        else:
            # If BG screenshot fails, fall back to the video file as background
            background_filename = bg_video_filename

        if bn_ok:
            banner_filename = bn_image_filename
        else:
            # If BN screenshot fails, reuse BG if that succeeded; otherwise, we
            # can leave the banner empty.
            banner_filename = bg_image_filename if bg_ok else banner_filename

        # Propagate final background/banner filenames into the metadata so the
        # .sm file references the generated images correctly.
        meta.background_filename = background_filename
        meta.banner_filename = banner_filename

    # Determine rest-window cycle based on song length
    if song_seconds >= 240.0:
        rest_measure_cycle = 12
    elif song_seconds >= 150.0:
        rest_measure_cycle = 16
    else:
        rest_measure_cycle = None

    # Build BPM map (for #BPMS) from effective beat times
    bpm_changes = build_bpm_changes(effective_beat_times, bpm)

    # Difficulty & jump aggressiveness presets (in canonical DDR order)
    all_diff_order: List[Tuple[str, float]] = [
        ("Beginner", 0.00),
        ("Easy", 0.05),
        ("Medium", 0.10),
        ("Hard", 0.18),
        ("Challenge", 0.25),
    ]

    if difficulties is None:
        diff_order = all_diff_order
    else:
        wanted = set(difficulties)
        diff_order = [(name, aggr) for (name, aggr) in all_diff_order if name in wanted]
        if not diff_order:
            diff_order = all_diff_order

    charts_with_counts: List[Tuple[StepChart, int]] = []

    rng = np.random.default_rng(seed)

    for idx, (name, jump_aggr) in enumerate(diff_order):
        # Each difficulty gets its own derived seed to keep things reproducible
        diff_seed = int(rng.integers(0, 2**32 - 1))

        positions, target_steps = pick_step_positions_for_difficulty(
            grid=grid,
            max_strength=max_strength,
            difficulty_name=name,
            total_beats=total_beats,
            bpm=bpm,
            hard_flag=hard,
            rng=np.random.default_rng(diff_seed),
        )

        # Temporary meter placeholder (will be overridden by assign_meters)
        temp_meter = 1
        chart = render_chart_from_positions(
            positions=positions,
            difficulty_label=name,
            meter=temp_meter,
            jump_aggressiveness=jump_aggr,
            seed=diff_seed,
            rest_measure_cycle=rest_measure_cycle,
            allow_holds=allow_holds,
            allow_mines=allow_mines,
        )

        actual_steps = count_steps_in_chart(chart)
        print(
            f"  {name}: target ~{target_steps} rows,"
            f" actual notes (taps+holds+mines) ≈ {actual_steps}"
        )
        charts_with_counts.append((chart, actual_steps))

    # Assign meters dynamically based on density & complexity
    assign_meters(charts_with_counts, song_seconds, meta.bpm, meter_bias=meter_bias)

    charts: List[StepChart] = [c for (c, _) in charts_with_counts]

    # .sm file lives alongside audio, named by song title
    out_sm = os.path.join(song_dir, f"{title_no_ext}.sm")
    write_sm_file(meta, charts, out_sm, bpm_changes=bpm_changes)


class AutoStepperCLI:
    def generate(
        self,
        input: str = ".",
        output: str = ".",
        duration: float = 90.0,
        synctime: float = 0.0,
        hard: bool = False,
        tap: bool = False,
        tapsync: float = -0.11,
        bg_format: str = "avi",
        difficulties: str = "Beginner,Easy,Medium,Hard,Challenge",
        seed: int = 999,
        no_mines: bool = False,
        no_holds: bool = False,
        meter_bias: int = 0,
        bpm_override: float = 0.0,
        offset_override: Optional[float] = None,
        workers: int = 1,
    ):
        """
        Generate StepMania .sm charts from audio or video.

        Args:
            input:  Audio/video file or directory containing files.
                    Supported: .mp3, .wav, .ogg, .flac, .mp4
            output: Output directory for generated StepMania song folders.
                    Each song becomes: <output>/<SongTitle>/.
            duration: Default seconds of audio to analyze from each song
                      (and trim to, if trimming is supported).
                      Classic DDR style uses 90 seconds.
            synctime: Extra sync offset (seconds) applied on top of
                      auto-aligned first beat. Use this to nudge timing
                      earlier/later if the chart feels slightly off.
            hard: If True, makes Hard & Challenge charts denser and more jumpy.
            tap: Placeholder for tap-sync mode (not yet implemented).
            tapsync: Tap-sync base offset (unused placeholder, kept for parity).
            bg_format: Background video format for .mp4 input:
                       - "avi" (default): transcode MP4 → AVI via ffmpeg
                       - "mp4": copy MP4 directly as BACKGROUND
            difficulties: Comma-separated list of difficulties to generate,
                          e.g. "Easy,Medium,Hard". Defaults to all 5.
            seed: Base RNG seed for reproducible chart generation.
            no_mines: If True, disables mines on all charts.
            no_holds: If True, disables holds on all charts.
            meter_bias: Integer bias added to all meters (before clamping).
                        Useful if you feel all charts are a bit too easy/hard.
            bpm_override: If >0, force this BPM instead of the detected one.
            offset_override: If set, use this exact StepMania OFFSET instead
                             of auto-aligning to the first beat.
            workers: Number of parallel worker processes to use:
                     * 1  → sequential (default)
                     * N>1 → up to N parallel processes
                     * -1 → use all available CPU cores
        """
        audio_files = find_audio_files(input)
        if not audio_files:
            print("No supported audio or video files found.")
            return

        if not output:
            output = "."

        os.makedirs(output, exist_ok=True)

        print(f"Found {len(audio_files)} file(s) to process.")
        use_full_for_all_long: Optional[bool] = None

        # Parse difficulties string into a list
        diff_list = [
            d.strip() for d in difficulties.split(",") if d.strip()
        ] or ["Beginner", "Easy", "Medium", "Hard", "Challenge"]

        allow_mines = not no_mines
        allow_holds = not no_holds

        # Pre-resolve per-song durations (and any prompts) on the main process.
        tasks: List[Tuple[str, float]] = []

        for p in audio_files:
            print(f"\n=== Preparing {os.path.basename(p)} ===")
            effective_duration = duration

            full_len = get_full_duration(p)
            if (
                duration is not None
                and duration > 0
                and full_len is not None
                and full_len > duration
            ):
                # Song is longer than our default analysis length
                if use_full_for_all_long is True:
                    effective_duration = full_len
                    print(
                        f"  Song is {full_len:.1f}s (> {duration:.0f}s) "
                        "- using full length (ALL long songs)."
                    )
                elif use_full_for_all_long is False:
                    effective_duration = duration
                    print(
                        f"  Song is {full_len:.1f}s (> {duration:.0f}s) "
                        "- keeping default length for this song."
                    )
                else:
                    print(
                        f"  Song '{os.path.basename(p)}' is {full_len:.1f}s "
                        f"(> {duration:.0f}s default)."
                    )
                    resp = builtins.input(
                        "  Use full length? [y]es / [a]ll long songs / [n]o (keep default 90s): "
                    ).strip().lower()
                    if resp in ("y", "yes"):
                        effective_duration = full_len
                        use_full_for_all_long = None  # ask again next time
                    elif resp in ("a", "all"):
                        effective_duration = full_len
                        use_full_for_all_long = True
                    else:
                        effective_duration = duration
                        use_full_for_all_long = False
            else:
                # If duration <= 0, just use full length if we know it
                if (duration is None or duration <= 0) and full_len is not None:
                    effective_duration = full_len

            tasks.append((p, effective_duration))

        # Decide worker count
        if workers is None or workers == 0:
            workers = 1
        if workers < 0:
            try:
                cpu_count = os.cpu_count() or 1
            except Exception:
                cpu_count = 1
            workers = cpu_count

        # Run tasks (sequential or parallel)
        if workers == 1 or len(tasks) == 1:
            for p, effective_duration in tasks:
                print(f"\n=== Processing {os.path.basename(p)} ===")
                try:
                    process_file(
                        path=p,
                        output_dir=output,
                        duration=effective_duration,
                        synctime=synctime,
                        hard=hard,
                        tap=tap,
                        tapsync=tapsync,
                        bg_format=bg_format,
                        difficulties=diff_list,
                        seed=seed,
                        allow_mines=allow_mines,
                        allow_holds=allow_holds,
                        meter_bias=meter_bias,
                        bpm_override=bpm_override,
                        offset_override=offset_override,
                    )
                except Exception as e:
                    print(f"  Error processing {p}: {e!r}")
        else:
            print(f"\nUsing up to {workers} worker processes for parallel generation...")
            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_map = {}
                for p, effective_duration in tasks:
                    print(f"\n=== Queuing {os.path.basename(p)} ===")
                    fut = executor.submit(
                        process_file,
                        path=p,
                        output_dir=output,
                        duration=effective_duration,
                        synctime=synctime,
                        hard=hard,
                        tap=tap,
                        tapsync=tapsync,
                        bg_format=bg_format,
                        difficulties=diff_list,
                        seed=seed,
                        allow_mines=allow_mines,
                        allow_holds=allow_holds,
                        meter_bias=meter_bias,
                        bpm_override=bpm_override,
                        offset_override=offset_override,
                    )
                    future_map[fut] = p

                for fut in as_completed(future_map):
                    p = future_map[fut]
                    try:
                        fut.result()
                    except Exception as e:
                        print(f"  Error processing {p} in worker: {e!r}")

        print("\nDone.")


if __name__ == "__main__":
    cli = AutoStepperCLI()

    # If script is run with no arguments -> behave like original AutoStepper:
    # process current folder with defaults and make per-song dirs here.
    if len(sys.argv) == 1:
        print("No command or arguments provided — running default AutoStepperPy...")
        print("Scanning current folder for audio/video files "
              "(mp3/wav/ogg/flac/mp4)...\n")
        cli.generate()
    else:
        fire.Fire(cli)

