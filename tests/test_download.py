"""The download step never contacts OSF when the annotations are already there."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from djembe import dataset as d  # noqa: E402


def no_network(*args, **kwargs):
    raise AssertionError("OSF contacted")


def test_complete_local_copy_skips_osf(tmp_path, monkeypatch):
    for take in d.TAKES:
        for suffix in d.REQUIRED_ANNOTATIONS:
            path = d._annotation(tmp_path, take, suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x")
    monkeypatch.setattr(d, "_fetch", no_network)
    assert d.missing_annotations(tmp_path) == []
    assert d.download_corpus(tmp_path, verbose=False) == 0


def test_missing_files_are_listed(tmp_path):
    assert len(d.missing_annotations(tmp_path)) == len(d.TAKES) * len(d.REQUIRED_ANNOTATIONS)
