"""Nothing derived from the corpus or other third-party audio may be committed.

Checks every file tracked by git: notebooks carry no outputs (outputs embed
audio players and corpus-derived tables), and no text file contains
base64-encoded audio.
"""
from pathlib import Path
import json
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
AUDIO_B64 = re.compile(rb"UklGR[A-Za-z0-9+/]{60}|data:audio/[a-z0-9.+-]+;base64,[A-Za-z0-9+/]{60}")
AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aif", ".aiff"}


def tracked(*patterns):
    out = subprocess.run(["git", "ls-files", *patterns], cwd=ROOT, capture_output=True, text=True, check=True)
    return [ROOT / p for p in out.stdout.splitlines() if (ROOT / p).exists()]


@pytest.mark.parametrize("path", tracked("*.ipynb"), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_notebook_has_no_outputs(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    dirty = [i for i, c in enumerate(nb["cells"]) if c.get("outputs") or c.get("execution_count")]
    assert not dirty, f"cells with outputs: {dirty}. Run: nbstripout {path.name}"


def test_no_audio_files_tracked():
    assert not [p for p in tracked() if p.suffix.lower() in AUDIO_EXT]


def test_no_base64_audio_in_tracked_files():
    hits = [p.relative_to(ROOT).as_posix() for p in tracked()
            if p.suffix.lower() not in {".png", ".jpg", ".gif"} and AUDIO_B64.search(p.read_bytes())]
    assert not hits, f"embedded audio in: {hits}"
