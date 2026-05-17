#!/usr/bin/env bash
# Upload tiles and team photos to Cloudflare R2.
# Requires rclone configured with an R2 remote.
# See: https://rclone.org/s3/#cloudflare-r2
#
# One-time rclone setup:
#   rclone config
#   → New remote → name: r2-e4e-mangrove → type: s3 → provider: Cloudflare
#   → access_key_id / secret_access_key: R2 API token (from Cloudflare dashboard)
#   → endpoint: https://<account-id>.r2.cloudflarestorage.com
#
# Usage:
#   bash scripts/upload_tiles.sh [region]   # upload map tiles for a region
#   bash scripts/upload_tiles.sh --team     # upload team photos only
#   bash scripts/upload_tiles.sh --all      # upload tiles + team photos

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TILES_DIR="$REPO_ROOT/landing/public/tiles"
TEAM_DIR="$REPO_ROOT/landing/public/team"
BUCKET="${R2_BUCKET:-e4e-mangrove}"
REMOTE="${RCLONE_REMOTE:-r2-e4e-mangrove}"

upload_tiles() {
  local REGION="$1"
  if [[ ! -d "$TILES_DIR/$REGION" ]]; then
    echo "Error: $TILES_DIR/$REGION not found. Run generate_raster_tiles.py first." >&2
    exit 1
  fi
  echo "Uploading $REGION PNG tiles..."
  rclone sync "$TILES_DIR/$REGION" "$REMOTE:$BUCKET/tiles/$REGION" \
    --progress \
    --transfers=16 \
    --s3-no-check-bucket
  if [[ -f "$TILES_DIR/$REGION.pmtiles" ]]; then
    echo "Uploading $REGION.pmtiles..."
    rclone copyto "$TILES_DIR/$REGION.pmtiles" "$REMOTE:$BUCKET/tiles/$REGION.pmtiles" \
      --progress \
      --s3-no-check-bucket
  fi
}

upload_team() {
  if [[ ! -d "$TEAM_DIR" ]]; then
    echo "Error: $TEAM_DIR not found." >&2
    exit 1
  fi
  echo "Uploading team photos..."
  rclone sync "$TEAM_DIR" "$REMOTE:$BUCKET/team" \
    --progress \
    --transfers=8 \
    --s3-no-check-bucket
}

case "${1:---team}" in
  --team)
    upload_team
    ;;
  --all)
    upload_tiles "florida"
    upload_team
    ;;
  *)
    upload_tiles "$1"
    ;;
esac

echo "Done. Ensure VITE_TILES_BASE_URL is set to your R2 public bucket URL."
