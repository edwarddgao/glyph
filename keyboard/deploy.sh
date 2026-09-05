#!/bin/zsh
# Build, sign, install and launch the Glyph keyboard on a connected iPhone.
#
#   ./deploy.sh                 # auto-detect team and device
#   DEVELOPMENT_TEAM=ABCDE12345 ./deploy.sh
#   DEVICE=<udid> ./deploy.sh
#
# One-time setup on the Mac: Xcode › Settings › Accounts › add your Apple ID
# (a free personal team is enough; the build is then valid for 7 days).
# One-time setup on the phone: Settings › Privacy & Security › Developer Mode,
# then trust this Mac when it asks. After the first install, go to
# Settings › General › VPN & Device Management and trust the developer profile.
set -euo pipefail
cd "$(dirname "$0")"

if [[ -z "${DEVELOPMENT_TEAM:-}" ]]; then
  # Xcode stores the teams of signed-in accounts here.
  DEVELOPMENT_TEAM=$(defaults read com.apple.dt.Xcode IDEProvisioningTeamByIdentifier 2>/dev/null \
    | grep -oE 'teamID = "?[A-Z0-9]{10}' | head -1 | grep -oE '[A-Z0-9]{10}$' || true)
fi
if [[ -z "${DEVELOPMENT_TEAM:-}" ]]; then
  echo "No development team found. Sign in to Xcode (Settings › Accounts) or pass DEVELOPMENT_TEAM=<id>." >&2
  exit 1
fi

if [[ -z "${DEVICE:-}" ]]; then
  DEVICE=$(xcrun devicectl list devices --json-output /tmp/swipe_devices.json >/dev/null 2>&1 \
    && python3 -c 'import json;d=json.load(open("/tmp/swipe_devices.json"))["result"]["devices"];d=[x for x in d if x.get("hardwareProperties",{}).get("platform")=="iOS" and x.get("connectionProperties",{}).get("tunnelState")!="unavailable"];print(d[0]["identifier"] if d else "")' || true)
fi
if [[ -z "${DEVICE:-}" ]]; then
  echo "No iPhone found. Plug it in (or pair over Wi-Fi), unlock it, trust this Mac, then retry. 'xcrun devicectl list devices' shows what is visible." >&2
  exit 1
fi

# The game's upload token (research/iphone/.secrets/upload_token) goes into
# Info.plist at generation; without it the build runs but keeps records on the phone.
if [[ -z "${GLYPH_UPLOAD_TOKEN:-}" && -f ../research/iphone/.secrets/upload_token ]]; then
  GLYPH_UPLOAD_TOKEN=$(cat ../research/iphone/.secrets/upload_token)
fi
echo "team $DEVELOPMENT_TEAM, device $DEVICE, upload token $([[ -n "${GLYPH_UPLOAD_TOKEN:-}" ]] && echo set || echo absent)"
GLYPH_BUILD=$(date +%Y%m%d%H%M)   # visible in `xcrun devicectl device info apps` as Bundle Version
echo "build $GLYPH_BUILD"
DEVELOPMENT_TEAM="$DEVELOPMENT_TEAM" GLYPH_UPLOAD_TOKEN="${GLYPH_UPLOAD_TOKEN:-}" GLYPH_BUILD="$GLYPH_BUILD" xcodegen generate >/dev/null

BUILD_LOG=$(mktemp)
if ! xcodebuild -project Glyph.xcodeproj -scheme Glyph -configuration Debug \
  -destination "id=$DEVICE" -derivedDataPath build -allowProvisioningUpdates \
  -allowProvisioningDeviceRegistration \
  DEVELOPMENT_TEAM="$DEVELOPMENT_TEAM" CODE_SIGN_STYLE=Automatic build > "$BUILD_LOG" 2>&1; then
  grep -E 'error:|BUILD' "$BUILD_LOG" | head -20 >&2
  echo "build failed; nothing installed" >&2
  exit 1
fi
grep -E '\*\* BUILD' "$BUILD_LOG"

APP=build/Build/Products/Debug-iphoneos/Glyph.app
[[ -d "$APP" ]] || { echo "build failed: no Glyph.app produced" >&2; exit 1; }

# Installing over a running copy has left the old one in place; stop it first, then verify the swap.
xcrun devicectl device process launch --terminate-existing --device "$DEVICE" com.edwardgao.glyph >/dev/null 2>&1 || true
sleep 1
xcrun devicectl device install app --device "$DEVICE" "$APP" | grep -E 'installed|error' || true
INSTALLED=$(xcrun devicectl device info apps --device "$DEVICE" 2>/dev/null | awk '/com\.edwardgao\.glyph /{print $NF}')
if [[ "$INSTALLED" != "$GLYPH_BUILD" ]]; then
  # iOS has kept a stale copy in place after a reported-successful install (seen 2026-09-05);
  # uninstalling first is the only thing that reliably swaps it. Loses the app's UserDefaults
  # (onboarding flag, best score, anonymous id) — keyboard settings are unaffected.
  echo "install did not take (device reports build '${INSTALLED:-none}', expected $GLYPH_BUILD); uninstalling the stale copy and reinstalling" >&2
  xcrun devicectl device uninstall app --device "$DEVICE" com.edwardgao.glyph >/dev/null 2>&1 || true
  for attempt in 1 2 3; do   # the app list lags the install by a moment; check after a pause, retry if needed
    xcrun devicectl device install app --device "$DEVICE" "$APP" | grep -E 'installed|error' || true
    sleep 3
    INSTALLED=$(xcrun devicectl device info apps --device "$DEVICE" 2>/dev/null | awk '/com\.edwardgao\.glyph /{print $NF}')
    [[ "$INSTALLED" == "$GLYPH_BUILD" ]] && break
  done
fi
echo "device now runs build ${INSTALLED:-unknown}"
if ! xcrun devicectl device process launch --terminate-existing --device "$DEVICE" com.edwardgao.glyph >/dev/null 2>&1; then
  echo
  echo "Installed, but iOS refused to launch it: the developer profile is not trusted yet (normal on first install)."
  echo "On the phone: Settings › General › VPN & Device Management › Developer App › trust it, then open Glyph."
fi
echo
echo "Installed. On the phone: Settings › General › Keyboard › Keyboards › Add New Keyboard… › Glyph."
echo "Then hold the globe key in any app and pick Glyph."
