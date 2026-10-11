"""Record docs/figures/boussole.gif from a synthesis-only build of the Boussole.

The page is built without any corpus recording or DjembeLoops reference
(``_corpus_recordings`` and ``_load_djembeloops_reference`` are stubbed), so
the capture can only show the resynthesis. It is driven in headless Chrome
through the DevTools protocol: synthesis mode, play, then each piece in turn.

    python tools/capture_boussole_gif.py [--chrome PATH] [--seconds-per-piece 4]
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image
import websocket

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT_HTML = ROOT / "output" / "boussole_synth_only.html"
OUT_GIF = ROOT / "docs" / "figures" / "boussole.gif"
PORT_HTTP, PORT_CDP = 8766, 9333
VIEW_W, VIEW_H = 1440, 1100
DEFAULT_CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def build_synth_only() -> None:
    import boussole_rythmes_V2 as b
    b._corpus_recordings = lambda rhythms: {}
    b._load_djembeloops_reference = lambda path=None: {"source": "", "patterns": [], "matrices": {}, "samples": []}
    b.build(str(OUT_HTML.relative_to(ROOT)))
    html = OUT_HTML.read_text(encoding="utf-8")
    assert ".wav" not in html and ".mp3" not in html and "data:audio" not in html


class CDP:
    def __init__(self, ws_url: str):
        self.ws = websocket.create_connection(ws_url, timeout=30)
        self.n = 0

    def call(self, method: str, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})

    def js(self, expr: str):
        return self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)["result"].get("value")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chrome", default=DEFAULT_CHROME)
    parser.add_argument("--seconds-per-piece", type=float, default=4.0)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--width", type=int, default=720, help="GIF width in pixels")
    args = parser.parse_args()

    build_synth_only()
    handler = partial(SimpleHTTPRequestHandler, directory=str(ROOT))
    server = ThreadingHTTPServer(("127.0.0.1", PORT_HTTP), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    profile = tempfile.mkdtemp(prefix="boussole_chrome_")
    chrome = subprocess.Popen([
        args.chrome, "--headless=new", f"--remote-debugging-port={PORT_CDP}", f"--user-data-dir={profile}",
        f"--window-size={VIEW_W},{VIEW_H}", "--autoplay-policy=no-user-gesture-required", "--mute-audio",
        "--hide-scrollbars", "about:blank",
    ])
    try:
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT_CDP}/json"))
                page = next(t for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.2)
        cdp = CDP(page["webSocketDebuggerUrl"])
        cdp.call("Emulation.setDeviceMetricsOverride", width=VIEW_W, height=VIEW_H, deviceScaleFactor=1, mobile=False)
        cdp.call("Page.enable")
        cdp.call("Page.navigate", url=f"http://127.0.0.1:{PORT_HTTP}/output/{OUT_HTML.name}")
        time.sleep(3)
        cdp.js("document.getElementById('circleSynthesisBtn').click()")
        cdp.js("document.getElementById('playBtn').click()")
        # Clip: page header down to the bottom of the circle panel.
        rect = cdp.js("""(() => { const c = document.getElementById('circle').getBoundingClientRect();
                         return {x: 0, y: 0, width: innerWidth, height: Math.ceil(c.bottom + 24)}; })()""")
        tabs_js = "[...document.querySelectorAll('button.tab')].slice(0,3)"
        frames, step = [], 1.0 / args.fps
        for i in range(3):
            cdp.js(f"{tabs_js}[{i}].click()")
            t_end = time.time() + args.seconds_per_piece
            while time.time() < t_end:
                t0 = time.time()
                shot = cdp.call("Page.captureScreenshot", format="png", clip={**rect, "scale": 1})
                frames.append(Image.open(io.BytesIO(base64.b64decode(shot["data"]))).convert("RGB"))
                time.sleep(max(0.0, step - (time.time() - t0)))
        cdp.js("document.getElementById('playBtn').click()")
    finally:
        chrome.terminate()
        server.shutdown()

    h = round(frames[0].height * args.width / frames[0].width)
    small = [f.resize((args.width, h), Image.LANCZOS) for f in frames]
    palette = small[len(small) // 2].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    gif = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in small]
    OUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    gif[0].save(OUT_GIF, save_all=True, append_images=gif[1:], duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"{len(gif)} frames, {OUT_GIF.stat().st_size / 1e6:.1f} MB -> {OUT_GIF.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
