#!/usr/bin/env python3
"""Run the gesture-replay benchmark for one keyboard and store scorer-ready results.

    ../research/.venv/bin/python tools/replay_bench.py --keyboard quickpath --source capture
    ../research/.venv/bin/python tools/replay_bench.py --keyboard swipe --source capture --device sim

Drives `GestureReplayBench.testReplay` (UITests/) with xcodebuild on the
connected iPhone (default) or the iPhone 17 simulator, parses the "BENCH {json}"
lines the test prints, and writes one `bench_<keyboard>_<n>.json` per sentence
into research/iphone/data/ — the same shape Block C uploads have, so
`research/iphone/benchmark_keyboards.py` scores replay and human sessions
alike (replay sessions are named `replay-<source>-<recording session>`).

Every keyboard sees byte-identical touch paths (recorded coordinates mapped
into that keyboard's measured letter grid, recorded timing), so the
comparison across keyboards is exactly paired.
"""
from __future__ import annotations

import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEYBOARD = HERE.parent
DATA = KEYBOARD.parent / "research" / "iphone" / "data"
SIM = "9A0DAD5E-DE02-4DB1-A063-83E4DBA378B8"


def device_udid() -> str:
    out = subprocess.run(["xcrun", "devicectl", "list", "devices", "--json-output", "/tmp/swipe_devices.json"],
                         capture_output=True, text=True)
    d = json.load(open("/tmp/swipe_devices.json"))["result"]["devices"]
    d = [x for x in d if x.get("hardwareProperties", {}).get("platform") == "iOS"
         and x.get("connectionProperties", {}).get("tunnelState") != "unavailable"]
    if not d:
        sys.exit("no iPhone connected")
    return d[0]["identifier"]


def team_id() -> str:
    out = subprocess.run(["defaults", "read", "com.apple.dt.Xcode", "IDEProvisioningTeamByIdentifier"],
                         capture_output=True, text=True).stdout
    m = re.search(r'teamID = "?([A-Z0-9]{10})', out)
    return m.group(1) if m else ""


def measure(base, env, a) -> None:
    """Screenshot the keyboard through the UI test, export the attachment, measure the grid."""
    import shutil, tempfile
    shots = DATA / "layout"; shots.mkdir(parents=True, exist_ok=True)
    bundle = Path(tempfile.mkdtemp()) / "shot.xcresult"
    cmd = base + ["test-without-building", "-resultBundlePath", str(bundle),
                  "-only-testing:GlyphUITests/GestureReplayBench/testScreenshotKeyboard"]
    print(f"screenshotting {a.keyboard} ({a.device})…", flush=True)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if "error:" in line or "BENCH " in line: print(line.rstrip())
    out = Path(tempfile.mkdtemp())
    subprocess.run(["xcrun", "xcresulttool", "export", "attachments", "--path", str(bundle), "--output-path", str(out)],
                   capture_output=True, text=True)
    pngs = sorted(out.glob("*.png"), key=lambda p: p.stat().st_size)
    if not pngs:
        sys.exit("no screenshot attachment exported (was the keyboard enabled in Settings?)")
    dst = shots / f"{a.keyboard}_{a.device}.png"; shutil.copy(pngs[-1], dst)
    m = subprocess.run([sys.executable, str(KEYBOARD / "tools/measure_layout.py"), str(dst), "--grid"], capture_output=True, text=True)
    print(f"screenshot -> {dst}")
    if m.returncode:
        print(m.stderr.strip()[-2000:]); sys.exit("grid not measurable from the screenshot; inspect it")
    print(f"grid: {m.stdout.strip()}   (pass as --grid to the replay)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyboard", required=True, choices=["quickpath", "gboard", "swiftkey", "swipe", "swipe-nolm"],
                    help="swipe-nolm = Swipe with the sentence LM off (first pass only)")
    ap.add_argument("--source", default="capture", choices=["capture", "futo", "all"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default=None, help="i/n: replay every n-th sentence from i; run one shard per simulator and score together")
    ap.add_argument("--speed", type=float, default=None,
                    help="time pre-compensation; default 1.2 on the simulator, 1.1 on the phone (both from testTimingCalibration)")
    ap.add_argument("--device", default="phone", help="phone | sim")
    ap.add_argument("--no-build", action="store_true", help="reuse the last build-for-testing")
    ap.add_argument("--tag", default=None, help="suffix for the keyboard name in the saved records, e.g. 'phone' -> quickpath-phone (keeps device runs apart in the scorer)")
    ap.add_argument("--grid", default=None, help="left,width,top,rowPitch in points (third-party keyboards; from --measure)")
    ap.add_argument("--measure", action="store_true",
                    help="bring the keyboard up, screenshot it, and print its letter grid instead of replaying")
    ap.add_argument("--sim", default=SIM, help="simulator UDID (several can run in parallel)")
    a = ap.parse_args()

    os.chdir(KEYBOARD)
    if a.device == "sim":
        dest = f"platform=iOS Simulator,id={a.sim}"
        sign = ["CODE_SIGNING_ALLOWED=NO"]
    else:
        dest = f"id={device_udid()}"
        sign = ["-allowProvisioningUpdates", f"DEVELOPMENT_TEAM={team_id()}", "CODE_SIGN_STYLE=Automatic"]
    base = ["xcodebuild", "-project", "Glyph.xcodeproj", "-scheme", "Glyph",
            "-destination", dest, "-derivedDataPath", "build"]
    if not a.no_build:
        # xcodegen substitutes ${DEVELOPMENT_TEAM} at generation; the UI test
        # runner's signing needs it baked in (command-line overrides do not reach it).
        env_gen = dict(os.environ, DEVELOPMENT_TEAM=team_id() if a.device != "sim" else "")
        subprocess.run(["xcodegen", "generate"], env=env_gen, capture_output=True)
        print("building for testing…", flush=True)
        r = subprocess.run(base + sign + ["build-for-testing"], capture_output=True, text=True)
        if "** TEST BUILD SUCCEEDED **" not in r.stdout:
            print("\n".join(l for l in r.stdout.splitlines() if "error:" in l)[:4000]); sys.exit("build failed")
    speed = a.speed if a.speed is not None else (1.2 if a.device == "sim" else 1.1)
    env = dict(os.environ, TEST_RUNNER_BENCH_KEYBOARD=a.keyboard, TEST_RUNNER_BENCH_SOURCE=a.source,
               TEST_RUNNER_BENCH_SPEED=str(speed))
    if a.limit:
        env["TEST_RUNNER_BENCH_LIMIT"] = str(a.limit)
    if a.shard:
        env["TEST_RUNNER_BENCH_SHARD"] = a.shard
    if a.grid:
        env["TEST_RUNNER_BENCH_GRID"] = a.grid
    if a.measure:
        measure(base, env, a); return
    cmd = base + ["test-without-building", "-only-testing:GlyphUITests/GestureReplayBench/testReplay"]
    print(f"replaying {a.source} on {a.keyboard} ({a.device})…", flush=True)
    DATA.mkdir(exist_ok=True)
    n = 0
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    stamp = int(time.time())
    if a.shard:   # shards launched together share a second; keep their files apart
        stamp = f"{stamp}s{a.shard.split('/')[0]}"
    for line in proc.stdout:
        m = re.search(r"BENCH (\{.*\})\s*$", line)
        if not m:
            if "error:" in line or "** TEST" in line:
                print(line.rstrip())
            continue
        rec = json.loads(m.group(1))
        if rec.get("event"):
            print(f"  {rec}")
            continue
        n += 1
        kb = f"{a.keyboard}-{a.tag}" if a.tag else a.keyboard
        rec["keyboard"] = kb
        (DATA / f"bench_{kb}_{a.source}_{stamp}_{n:04d}.json").write_text(json.dumps(rec, indent=1))
        ok = sum(x == y for x, y in zip(rec["sentence"].split(), rec["typed"].lower().split()))
        print(f"  [{n}] {rec['sentence']!r} -> {rec['typed'].strip()!r}  ({ok}/{len(rec['sentence'].split())})", flush=True)
    proc.wait()
    print(f"{n} sentences in {time.time() - t0:.0f}s -> {DATA}/bench_{a.keyboard}{'-' + a.tag if a.tag else ''}_{a.source}_{stamp}_*.json")
    print("score with: .venv/bin/python iphone/benchmark_keyboards.py   (in research/)")


if __name__ == "__main__":
    main()
