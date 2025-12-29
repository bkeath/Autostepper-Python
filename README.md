# Autostepper-Python
AutoStepperPy – Automatic DDR/ITG StepChart Generator for StepMania
<br/>
[![Autostepper-Py Demo](https://img.youtube.com/vi/lvfqGceitJA/0.jpg)](https://www.youtube.com/watch?v=lvfqGceitJA)

<br/>
AutoStepperPy is a modern Python-based chart generator that creates DDR/ITG-style stepcharts (.sm files) for StepMania from your music and videos.

Point it at a folder of songs, and it will:

Detect BPM and beats using librosa

Build a 16-row rhythm grid per measure (4ths, 8ths, 16ths)

Generate up to five difficulties (Beginner, Easy, Medium, Hard, Challenge)

Add holds, jumps, gallops, jacks, and mines in a DDR/ITG-inspired way

Handle long songs and marathons with built-in pacing and micro-breaks

For MP4 files, automatically:

Extract audio for #MUSIC

Create a background video (AVI or MP4)

Capture BG (640×480) and banner (256×80) images from the video

Charts are saved in standard StepMania format:

<output>/<SongTitle>/
    <SongTitle>.sm        # generated chart
    <SongTitle>.<audio>   # trimmed or full audio / extracted audio
    <SongTitle>.avi/.mp4  # background video (if MP4 input)
    <SongTitle>_bg.png    # static background image
    <SongTitle>_bn.png    # banner image


It’s designed to feel roughly like DDRMAX/ITG-style charts, with:

Energy-aware density (choruses get more notes)

Tempo-aware difficulty (fewer 16ths at high BPM)

Phrase-based patterns and light “footedness” logic to avoid awful doublesteps

Optional parallel processing to speed up large batches

Features

🎵 Automatic timing & BPM

Uses librosa for beat tracking and onset strength.

Builds a smoothed multi-segment #BPMS map.

Auto-aligns #OFFSET to the first detected beat, with an extra manual synctime nudge.

🕺 DDR-style rhythm grid

Full 16-row grid per measure (4ths, 8ths, 16ths).

Synthesized strengths for off-beats so higher difficulties get good streams.

Per-difficulty density based on BPM and overall energy.

🧠 Pattern & movement logic

Pattern banks for simple sweeps, pivots, and crossovers.

4-measure phrase motifs to keep streams feeling intentional.

Gallops, jacks, and jumps tuned by difficulty.

Basic “foot side” logic to reduce ugly doublesteps.

⏱️ Long songs & marathons

Default analysis length (e.g. 90 seconds) for classic DDR-style cuts.

Interactive prompts for longer songs to:

Use full length,

Use full length for all long songs, or

Keep default clip length.

Classifies songs as Regular/Long/Marathon with:

#MUSICLENGTH: and #SONGTYPE: tags.

Built-in “rest windows” for marathon charts to give micro-breaks.

🎬 MP4 / media handling

Extracts audio from MP4 to .ogg (via pydub + ffmpeg).

Creates a background video:

Default: MP4 → AVI (smoother in some StepMania builds).

Or copy MP4 directly using bg_format="mp4".

Grabs video screenshots at ~10s to generate:

<title>_bg.png (640×480) → #BACKGROUND

<title>_bn.png (256×80) → #BANNER

📊 Meters & difficulty

Generates up to 5 difficulties:

Beginner, Easy, Medium, Hard, Challenge.

Meter is based on:

Steps-per-minute, and

Pattern complexity (16ths + longest stream).

Enforces an ordered scale:

Beginner ≤ Easy ≤ Medium ≤ Hard ≤ Challenge.

Optional global meter_bias to shift all ratings up or down.

⚙️ Parallel processing

--workers lets you use multiple CPU cores:

--workers 1 – sequential (default)

--workers 4 – up to 4 parallel processes

--workers -1 – use all available cores

Requirements
Python

Python 3.8+ (recommended 3.9+)

Python packages

Install via pip:

pip install numpy librosa soundfile fire


Optional (but strongly recommended for full functionality):

pip install pydub


pydub is used to:

Trim audio to the selected duration.

Extract audio from MP4 files.

Without pydub, audio trimming and MP4 audio extraction will be limited:

MP4s will fall back to using the video file as #MUSIC if needed.

External tools

ffmpeg (required for MP4 features and screenshots)

Needed to:

Transcode MP4 → AVI for background videos.

Capture BG/BN screenshots.

Install ffmpeg via your package manager:

Windows: via Chocolatey, Scoop, or manual download.

macOS: brew install ffmpeg

Linux: sudo apt install ffmpeg (Debian/Ubuntu), or your distro’s equivalent.

Supported input formats

.mp3

.wav

.ogg

.flac

.mp4 (with optional background video, BG/BN screenshots, and audio extraction)

Installation

Clone the repository:

git clone https://github.com/<your-username>/AutoStepperPy.git
cd AutoStepperPy


(Optional but recommended) create a virtual environment:

python -m venv venv
source venv/bin/activate      # macOS/Linux
# or
venv\Scripts\activate         # Windows


Install dependencies:

pip install numpy librosa soundfile fire pydub


Make sure ffmpeg is installed and available on your system PATH.

Basic Usage
1. Quick start (default: current folder, 90-second cuts)

Place your audio/video files in the current directory and run:

python AutoStepper.py


Scans the current folder for .mp3/.wav/.ogg/.flac/.mp4.

Generates per-song folders in the current directory.

Uses ~90 seconds of the song by default (classic DDR style).

2. Use the CLI directly via Python Fire

You can call the generate command explicitly:

python AutoStepper.py generate --input="songs" --output="StepmaniaSongs"


Key arguments:

--input
File or directory of source media.

--output
Output directory where StepMania-style song folders will be created.

--duration
Default length (in seconds) of audio to analyze and trim to.
Example: --duration=90 (default).

--synctime
Extra sync offset (seconds) applied on top of the auto-aligned first beat.
Use this to nudge charts slightly earlier or later if they feel off.
Example: --synctime=-0.05

--hard
Make Hard & Challenge charts denser and more jumpy.
Example: --hard=True

--bg_format
How to handle MP4 videos as backgrounds:

"avi" – transcode MP4 → AVI (default)

"mp4" – copy MP4 as-is as background

python AutoStepper.py generate --bg_format=mp4


--difficulties
Comma-separated list of difficulties to generate:

# Only Easy/Medium/Hard
python AutoStepper.py generate --difficulties="Easy,Medium,Hard"


--no_mines, --no_holds
Disable mines or holds globally:

python AutoStepper.py generate --no_mines=True --no_holds=True


--meter_bias
Shift all ratings up/down before clamping:

# Make charts feel 1 level harder
python AutoStepper.py generate --meter_bias=1


--bpm_override
Force a specific BPM instead of using detected BPM:

python AutoStepper.py generate --bpm_override=140.0


--offset_override
Force an exact StepMania #OFFSET (seconds).
If set, it replaces the automatic first-beat alignment.

3. Using multiple CPU cores

For big batches, use the --workers flag:

# Use all available CPU cores
python AutoStepper.py generate --workers=-1

# Use 4 workers
python AutoStepper.py generate --workers=4


Notes:

--workers=1 (or omitting it) runs sequentially.

Long-song prompts (for full length vs. 90s) happen before parallel processing, so workers won’t try to read from stdin.

4. Typical workflow

Collect songs you want to chart in a folder:

songs/
  track01.mp3
  track02.mp4
  ...


Run AutoStepperPy:

python AutoStepper.py generate --input="songs" --output="StepmaniaSongs" --workers=-1


Copy StepmaniaSongs/ into your StepMania Songs/ directory.

Launch StepMania, scan for new songs, and play-test your charts.
