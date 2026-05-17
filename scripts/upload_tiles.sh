#!/usr/bin/env bash
# Upload generated tiles to Cloudflare R2.
# Requires rclone configured with an R2 remote named "r2".
# See: https://rclone.org/s3/#cloudflare-r2
#
# One-time rclone setup:
#   rclone config
#   → New remote → name: r2 → type: s3 → provider: Cloudflare
#   → access_key_id / secret_access_key: R2 API token (from Cloudflare dashboard)
#   → endpoint: https://<account-id>.r2.cloudflarestorage.com
#
# Usage: bash scripts/upload_tiles.sh [region]
# Example: bash scripts/upload_tiles.sh florida

set -euo pipefail

REGION="${1:-florida}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TILES_DIR="$REPO_ROOT/landing/public/tiles"
BUCKET="${R2_BUCKET:-e4e-mangrove}"

if [[ ! -d "$TILES_DIR/$REGION" ]]; then
  echo "Error: $TILES_DIR/$REGION not found. Run generate_raster_tiles.py first." >&2
  exit 1
fi

REMOTE="${RCLONE_REMOTE:-r2-e4e-mangrove}"

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

echo "Done. Set VITE_TILES_BASE_URL to your R2 public bucket URL in GitHub secrets."
