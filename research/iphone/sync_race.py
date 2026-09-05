#!/usr/bin/env python3
"""Pull uploaded records from the Cloudflare Worker's R2 bucket into data/.

    .venv/bin/python iphone/sync_race.py [--url https://swipe-upload.<account>.workers.dev]

Lists every object through the Worker's admin endpoints (token in
.secrets/admin_token), downloads the ones not yet on disk, and names them the
way server.py does — `<kind>_<session>_<ts>_<hash>.json` — so race_to_capture.py,
join_kbcapture.py and benchmark_keyboards.py see no difference between a record
that came over the LAN and one that came from the internet. A `.sync_index`
file in data/ remembers which keys are already local.
"""
from __future__ import annotations

import argparse, json, sys, urllib.parse, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DEFAULT_URL_FILE = HERE / ".secrets" / "worker_url"


def get(url: str, token: str) -> bytes:
    # Cloudflare answers 403 to urllib's default User-Agent; name ourselves.
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "User-Agent": "swipe-sync/1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL_FILE.read_text().strip() if DEFAULT_URL_FILE.exists() else None)
    ap.add_argument("--data", default=str(DATA))
    a = ap.parse_args()
    if not a.url:
        sys.exit("pass --url or write the Worker URL to iphone/.secrets/worker_url")
    token = (HERE / ".secrets" / "admin_token").read_text().strip()
    data = Path(a.data); data.mkdir(exist_ok=True)
    index_file = data / ".sync_index"
    have = set(index_file.read_text().split()) if index_file.exists() else set()

    keys, cursor = [], None
    while True:
        q = f"?cursor={urllib.parse.quote(cursor)}" if cursor else ""
        page = json.loads(get(f"{a.url.rstrip('/')}/list{q}", token))
        keys += page["keys"]
        cursor = page.get("cursor")
        if not cursor: break
    new = [k for k in keys if k["key"] not in have]
    print(f"{len(keys)} objects in the bucket, {len(new)} new")
    for k in new:
        body = get(f"{a.url.rstrip('/')}/obj/{urllib.parse.quote(k['key'], safe='/')}", token)
        try:
            p = json.loads(body)
        except json.JSONDecodeError:
            print("  skipping unparsable", k["key"]); continue
        name = "{}_{}_{}_{}.json".format(p.get("kind", "x"), p.get("session", "anon"), p.get("ts", 0), abs(hash(p.get("sentence", ""))) % 10_000)
        (data / name).write_text(json.dumps(p, indent=1))
        have.add(k["key"])
        print(f"  {k['key']} -> {name}  ({p.get('sentence', '')!r})")
    index_file.write_text("\n".join(sorted(have)))


if __name__ == "__main__":
    main()
