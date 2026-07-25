"""Extrait une reference Djembe Loops exploitable par la boussole.

Le script lit les fichiers publics du site et genere un JSON compact avec :
- les matrices rythmiques encodees dans js/rythms.js ;
- les metadonnees de patterns quand elles sont disponibles ;
- les URLs des samples audio references dans js/global.js.

Il copie aussi les MP3 dans data/djembeloops_samples/ pour que le player
de reference externe puisse fonctionner localement.
"""
from __future__ import annotations

import ast
import json
import os
import re
import urllib.request


BASE_URL = "https://www.djembeloops.com"
RYTHMS_URL = f"{BASE_URL}/js/rythms.js"
GLOBAL_URL = f"{BASE_URL}/js/global.js"
OUT_JSON = "data/djembeloops_reference.json"
SAMPLE_DIR = "data/djembeloops_samples"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=40) as response:
        return response.read().decode("utf-8", errors="replace")


def extract_js_array(text: str, name: str):
    match = re.search(r"\bvar\s+" + re.escape(name) + r"\s*=\s*\[", text)
    if not match:
        return None

    start = match.end() - 1
    depth = 0
    in_string = False
    quote = ""
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
        else:
            if char in ("'", '"'):
                in_string = True
                quote = char
            elif char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    raw = text[start : index + 1]
                    raw = raw.replace("false", "False").replace("true", "True")
                    raw = raw.replace("null", "None")
                    return ast.literal_eval(raw)
    return None


def extract_matrix_names(text: str) -> list[str]:
    return sorted(set(re.findall(r"\bvar\s+([A-Za-z0-9_]+)\s*=\s*\[", text)))


def normalize_row(row):
    instrument = str(row[0]).rstrip("_")
    notes = [
        None if value is False or value is None else str(value)
        for value in row[1:]
    ]
    return {"instrument": instrument, "notes": notes}


def extract_matrices(text: str):
    matrices = {}
    for name in extract_matrix_names(text):
        rows = extract_js_array(text, name)
        if not isinstance(rows, list) or not rows:
            continue
        if not all(isinstance(row, list) and row for row in rows):
            continue
        step_count = max(len(row) - 1 for row in rows)
        matrices[name] = {
            "steps": int(step_count),
            "rows": [normalize_row(row) for row in rows],
        }
    return matrices


def extract_pattern_metadata(text: str):
    metadata = {}
    for match in re.finditer(
        r"\b([A-Za-z0-9_]+)\s*:\s*\{\s*name\s*:\s*(['\"])(.*?)\2",
        text,
        flags=re.S,
    ):
        key = match.group(1)
        block_start = match.start()
        next_match = re.search(r"\n\s*[A-Za-z0-9_]+\s*:\s*\{", text[match.end() :])
        block_end = (
            match.end() + next_match.start()
            if next_match
            else min(len(text), match.end() + 2500)
        )
        block = text[block_start:block_end]
        steps = re.search(r"\bsteps\s*:\s*([0-9]+)", block)
        matrix = re.search(r"\bmatrix\s*:\s*(['\"])(.*?)\1", block)
        tempo = re.search(r"\btempo\s*:\s*([0-9]+)", block)
        if not steps or not matrix:
            continue
        metadata[key] = {
            "name": match.group(3),
            "steps": int(steps.group(1)),
            "rawSteps": steps.group(1),
            "matrix": matrix.group(2),
            "tempo": None if not tempo else int(tempo.group(1)),
        }
    return metadata


def sample_entry(url: str):
    filename = url.rsplit("/", 1)[-1]
    name = filename.removesuffix(".mp3")
    parts = name.split("_")
    if name.startswith("Shekere"):
        family = name
        articulation = "shake"
        variant = ""
    elif len(parts) >= 2 and parts[1] == "Bell":
        family = parts[0]
        articulation = "_".join(parts[1:])
        variant = ""
    else:
        family = parts[0]
        articulation = parts[1] if len(parts) > 1 else ""
        variant = "L" if articulation.endswith("L") else ""
        if variant:
            articulation = articulation[:-1]
    return {
        "name": name,
        "family": family,
        "articulation": articulation,
        "variant": variant,
        "url": url,
        "localPath": f"{SAMPLE_DIR}/{filename}",
    }


def extract_samples(text: str):
    urls = sorted(set(re.findall(r"https://www\.djembeloops\.com/snd/[^\"']+", text)))
    return [sample_entry(url) for url in urls]


def download_samples(samples):
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    downloaded = 0
    for sample in samples:
        path = sample["localPath"]
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        request = urllib.request.Request(sample["url"], headers=HEADERS)
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read()
        with open(path, "wb") as handle:
            handle.write(content)
        downloaded += 1
    return downloaded


def build_reference():
    rythms_js = fetch_text(RYTHMS_URL)
    global_js = fetch_text(GLOBAL_URL)
    matrices = extract_matrices(rythms_js)
    metadata = extract_pattern_metadata(rythms_js)
    patterns = []
    for key, meta in sorted(metadata.items()):
        matrix_key = meta["matrix"]
        if matrix_key not in matrices:
            continue
        if meta["steps"] != matrices[matrix_key]["steps"]:
            meta = {
                **meta,
                "declaredSteps": meta["steps"],
                "steps": matrices[matrix_key]["steps"],
            }
        patterns.append({
            "key": key,
            **meta,
            "matrixSteps": matrices[matrix_key]["steps"],
        })
    samples = extract_samples(global_js)
    downloaded = download_samples(samples)
    return {
        "source": BASE_URL,
        "rythmsUrl": RYTHMS_URL,
        "globalUrl": GLOBAL_URL,
        "patterns": patterns,
        "matrices": matrices,
        "samples": samples,
        "localSampleDir": SAMPLE_DIR,
        "downloadedSamples": downloaded,
    }


def main():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    data = build_reference()
    with open(OUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, separators=(",", ":"))
    print(
        f"OK -> {OUT_JSON} "
        f"({len(data['patterns'])} patterns, {len(data['matrices'])} matrices, "
        f"{len(data['samples'])} samples)"
    )


if __name__ == "__main__":
    main()
