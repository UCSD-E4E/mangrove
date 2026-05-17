#!/usr/bin/env bash
# Upload public assets to Cloudflare R2 via rclone.
#
# Usage:
#   ./scripts/upload.sh            # upload Florida tiles only
#   ./scripts/upload.sh --team     # upload team photos only
#   ./scripts/upload.sh --blog     # upload blog images only
#   ./scripts/upload.sh --all      # upload everything
#
# Requires rclone configured with a remote named "r2":
#   rclone config  →  add Cloudflare R2 remote named "r2"

set -euo pipefail

REMOTE="r2"
BUCKET="mangrove-landing"          # change to your actual R2 bucket name

TILES_DIR="public/tiles/florida"
TEAM_DIR="public/team"
BLOG_DIR="public/blog"

upload_tiles() {
  echo "→ Uploading Florida tiles..."
  rclone sync "$TILES_DIR" "$REMOTE:$BUCKET/tiles/florida" \
    --transfers 32 --checkers 16 --progress
}

upload_team() {
  echo "→ Uploading team photos..."
  rclone sync "$TEAM_DIR" "$REMOTE:$BUCKET/team" \
    --transfers 8 --progress
}

upload_blog() {
  echo "→ Uploading blog images..."
  rclone sync "$BLOG_DIR" "$REMOTE:$BUCKET/blog" \
    --transfers 8 --progress
}

case "${1:-}" in
  --team)  upload_team ;;
  --blog)  upload_blog ;;
  --all)
    upload_tiles
    upload_team
    upload_blog
    ;;
  *)
    upload_tiles
    ;;
esac

echo "Done."
