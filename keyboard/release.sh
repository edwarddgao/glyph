#!/usr/bin/env bash
# Archive Glyph and upload it to App Store Connect (TestFlight).
#
#   ./release.sh              # archive, export for App Store Connect, upload
#   ./release.sh --archive    # archive only (works on a personal team; validates the build)
#
# Needs Xcode signed in to a paid Apple Developer team (Settings › Accounts) —
# a personal team cannot sign for distribution. Optional: an App Store
# Connect API key in ../research/iphone/.secrets/ (asc_key.p8, asc_key_id,
# asc_issuer_id) lets the export run without the Xcode account prompt.
# The upload token is required: a build without it would collect nothing.
set -euo pipefail
cd "$(dirname "$0")"
SECRETS=../research/iphone/.secrets
MODE=${1:-upload}

GLYPH_BUILD=$(date +%Y%m%d%H%M)
if [[ -f "$SECRETS/upload_token" ]]; then export GLYPH_UPLOAD_TOKEN="$(cat "$SECRETS/upload_token")"; fi
if [[ -z "${GLYPH_UPLOAD_TOKEN:-}" && "$MODE" != "--archive" ]]; then
  echo "No upload token at $SECRETS/upload_token — refusing to release a build that cannot upload." >&2; exit 1
fi

mkdir -p build
# Team: env, else the first non-personal team Xcode knows about.
if [[ -z "${DEVELOPMENT_TEAM:-}" ]]; then
  DEVELOPMENT_TEAM=$(defaults export com.apple.dt.Xcode - 2>/dev/null | python3 -c '
import plistlib, sys
d = plistlib.loads(sys.stdin.buffer.read()).get("IDEProvisioningTeamByIdentifier", {})
teams = [t for ts in d.values() for t in ts]
paid = [t["teamID"] for t in teams if t.get("teamType") != "Personal Team"]
any_ = [t["teamID"] for t in teams]
print((paid or any_ or [""])[0]); print("paid" if paid else "personal", file=sys.stderr)' 2>build/.team_kind || true)
fi
KIND=$(cat build/.team_kind 2>/dev/null || echo unknown)
if [[ -z "${DEVELOPMENT_TEAM:-}" ]]; then echo "No team. Sign in to Xcode or pass DEVELOPMENT_TEAM=<id>." >&2; exit 1; fi
if [[ "$KIND" == "personal" && "$MODE" != "--archive" ]]; then
  echo "Xcode only knows the personal team $DEVELOPMENT_TEAM; distribution needs the paid team. Open Xcode › Settings › Accounts and let it refresh, or pass DEVELOPMENT_TEAM=<paid team id>." >&2; exit 1
fi
echo "team $DEVELOPMENT_TEAM ($KIND), build $GLYPH_BUILD, upload token $([[ -n "${GLYPH_UPLOAD_TOKEN:-}" ]] && echo set || echo absent), mode $MODE"

DEVELOPMENT_TEAM="$DEVELOPMENT_TEAM" GLYPH_UPLOAD_TOKEN="${GLYPH_UPLOAD_TOKEN:-}" GLYPH_BUILD="$GLYPH_BUILD" xcodegen generate >/dev/null

ARCHIVE=build/Glyph.xcarchive
rm -rf "$ARCHIVE"
echo "archiving…"
if ! xcodebuild -project Glyph.xcodeproj -scheme Glyph -configuration Release \
    -destination 'generic/platform=iOS' -archivePath "$ARCHIVE" -allowProvisioningUpdates \
    DEVELOPMENT_TEAM="$DEVELOPMENT_TEAM" CODE_SIGN_STYLE=Automatic archive > build/archive.log 2>&1; then
  grep -E 'error:|error ' build/archive.log | head -20 >&2; echo "archive failed (build/archive.log)" >&2; exit 1
fi
echo "archived $ARCHIVE ($(plutil -extract ApplicationProperties.CFBundleVersion raw "$ARCHIVE/Info.plist" 2>/dev/null || echo '?'))"
[[ "$MODE" == "--archive" ]] && exit 0

cat > build/ExportOptions.plist <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>method</key><string>app-store-connect</string>
  <key>destination</key><string>upload</string>
  <key>signingStyle</key><string>automatic</string>
  <key>teamID</key><string>$DEVELOPMENT_TEAM</string>
  <key>uploadSymbols</key><true/>
  <key>manageAppVersionAndBuildNumber</key><false/>
</dict></plist>
EOF

AUTH=()
if [[ -f "$SECRETS/asc_key.p8" && -f "$SECRETS/asc_key_id" && -f "$SECRETS/asc_issuer_id" ]]; then
  AUTH=(-authenticationKeyPath "$(cd "$SECRETS" && pwd)/asc_key.p8" -authenticationKeyID "$(cat "$SECRETS/asc_key_id")" -authenticationKeyIssuerID "$(cat "$SECRETS/asc_issuer_id")")
  echo "using App Store Connect API key $(cat "$SECRETS/asc_key_id")"
fi
echo "exporting and uploading…"
if ! xcodebuild -exportArchive -archivePath "$ARCHIVE" -exportOptionsPlist build/ExportOptions.plist \
    -exportPath build/export -allowProvisioningUpdates "${AUTH[@]}" > build/export.log 2>&1; then
  grep -E 'error:|error |Error' build/export.log | head -20 >&2; echo "export/upload failed (build/export.log)" >&2; exit 1
fi
grep -E 'Upload succeeded|EXPORT SUCCEEDED|Uploaded' build/export.log | head -3 || true
echo "uploaded build $GLYPH_BUILD — processing takes a few minutes in App Store Connect › TestFlight"
