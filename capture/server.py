"""LAN capture server for the iPhone swipe study.

Serves index.html and accepts POST /save with one JSON payload per
sentence (either a block-A gesture bundle or a block-B native-keyboard
transcript). Each payload lands as its own timestamped file under data/.

Run:  python server.py   (binds 0.0.0.0:8765)
"""

import json
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        # Safari aggressively caches LAN pages; the study page must never be stale.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        if self.path != "/save":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        if not 0 < n < 5_000_000:
            self.send_error(400)
            return
        raw = self.rfile.read(n)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self.send_error(400)
            return
        name = "{}_{}_{}_{}.json".format(
            payload.get("kind", "x"),
            payload.get("session", "anon"),
            payload.get("ts", int(time.time() * 1000)),
            abs(hash(payload.get("sentence", ""))) % 10_000,
        )
        (DATA / name).write_text(json.dumps(payload, indent=1))
        print(f"saved {name}  ({payload.get('sentence', '')!r})", flush=True)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def log_message(self, fmt, *args):  # quieter GET logging
        if "save" in (args[0] if args else ""):
            super().log_message(fmt, *args)


if __name__ == "__main__":
    print("capture server on http://0.0.0.0:8765  (data -> ./data/)", flush=True)
    HTTPServer(("0.0.0.0", 8765), Handler).serve_forever()
