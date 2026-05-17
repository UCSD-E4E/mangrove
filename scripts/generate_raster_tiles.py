"""
Generate a PNG raster tile pyramid from ESA WorldCover GeoTIFFs.

Input:  ../../ml-mangrove/Scale Adaption/GEE/data/pmtiles_pipeline/{region}/tifs/*.tif
Output: ../landing/public/tiles/{region}/{z}/{x}/{y}.png

Each TIF band 1 contains ESA WorldCover class codes:
  10=Trees, 20=Shrubland, 30=Grassland, 40=Cropland,
  50=Built-up, 60=Bare/sparse, 70=Snow, 80=Water,
  90=Wetland, 95=Mangroves, 100=Moss

ESA values are remapped to class_idx 0-5, then painted with CLASS_COLORS.
Background (class 0) is rendered fully transparent so the satellite shows through.
"""

import argparse
import math
import os
from pathlib import Path

import mercantile
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject
from PIL import Image

# ── class mapping ────────────────────────────────────────────────────────────
ESA_TO_CLASS = {10: 1, 20: 0, 30: 0, 40: 0, 50: 2, 60: 0, 70: 0, 80: 3, 90: 4, 95: 5, 100: 0}

# RGBA — background is fully transparent (alpha=0) so satellite shows through
CLASS_COLORS: dict[int, tuple[int, int, int, int]] = {
    0: (242, 239, 232,   0),   # Background  — transparent
    1: ( 61, 107,  74, 200),   # Tree Cover
    2: (138, 106,  74, 200),   # Built-up
    3: ( 74, 144, 184, 200),   # Water
    4: (120, 160, 120, 200),   # Wetland
    5: ( 30,  80,  50, 230),   # Mangrove
}

WGS84 = CRS.from_epsg(4326)
WEB_MERCATOR = CRS.from_epsg(3857)
TILE_SIZE = 256

# Build look-up table (LUT): ESA raw value (0-255) → RGBA
_LUT = np.zeros((256, 4), dtype=np.uint8)
for esa_val, class_idx in ESA_TO_CLASS.items():
    _LUT[esa_val] = CLASS_COLORS[class_idx]
# ESA 0 (nodata) → transparent
_LUT[0] = (0, 0, 0, 0)


def esa_to_rgba(esa_band: np.ndarray) -> np.ndarray:
    """Apply LUT to a 2-D ESA value array → (H, W, 4) RGBA uint8."""
    return _LUT[esa_band]


def tile_bounds_wgs84(tile: mercantile.Tile) -> tuple[float, float, float, float]:
    """Return (west, south, east, north) in WGS-84 for a tile."""
    b = mercantile.bounds(tile)
    return b.west, b.south, b.east, b.north


def warp_to_tile(src_path: str, tile: mercantile.Tile) -> np.ndarray | None:
    """
    Re-project the ESA TIF to EPSG:3857 cropped to the tile bounds.
    Returns a (TILE_SIZE, TILE_SIZE) uint8 array of ESA raw values,
    or None if the tile has no overlap with the file.
    """
    west, south, east, north = tile_bounds_wgs84(tile)

    with rasterio.open(src_path) as src:
        # Quick bounds check (both in WGS-84)
        fb = src.bounds
        if east < fb.left or west > fb.right or north < fb.bottom or south > fb.top:
            return None

        # Destination transform (EPSG:3857 tile)
        tile_bounds_3857 = mercantile.xy_bounds(tile)
        dst_transform = from_bounds(
            tile_bounds_3857.left, tile_bounds_3857.bottom,
            tile_bounds_3857.right, tile_bounds_3857.top,
            TILE_SIZE, TILE_SIZE,
        )
        dst_array = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)

        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_crs=WGS84,
            dst_crs=WEB_MERCATOR,
            dst_transform=dst_transform,
            resampling=Resampling.nearest,
        )

    return dst_array


def generate_tiles(
    tif_dir: Path,
    out_dir: Path,
    zoom_levels: list[int],
    region: str,
) -> None:
    tif_files = sorted(tif_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {tif_dir}")

    print(f"Found {len(tif_files)} TIF files for region '{region}'")

    # Compute union bounding box (WGS-84) across all TIFs
    west_all, south_all, east_all, north_all = 180.0, 90.0, -180.0, -90.0
    for tf in tif_files:
        with rasterio.open(tf) as src:
            b = src.bounds
            west_all  = min(west_all,  b.left)
            south_all = min(south_all, b.bottom)
            east_all  = max(east_all,  b.right)
            north_all = max(north_all, b.top)

    print(f"Coverage: W={west_all:.3f} S={south_all:.3f} E={east_all:.3f} N={north_all:.3f}")

    for zoom in zoom_levels:
        tiles = list(mercantile.tiles(west_all, south_all, east_all, north_all, zooms=zoom))
        print(f"  z={zoom}: {len(tiles)} candidate tiles")

        written = 0
        for tile in tiles:
            out_path = out_dir / str(tile.z) / str(tile.x) / f"{tile.y}.png"

            # Composite contributions from all overlapping TIFs
            composite = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)  # ESA values
            has_data = False

            for tf in tif_files:
                arr = warp_to_tile(str(tf), tile)
                if arr is None:
                    continue
                mask = arr > 0
                if mask.any():
                    composite[mask] = arr[mask]
                    has_data = True

            if not has_data:
                continue

            rgba = esa_to_rgba(composite)

            # Skip tiles that are purely transparent (no classified pixels)
            if rgba[:, :, 3].max() == 0:
                continue

            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgba, mode="RGBA").save(out_path, optimize=False)
            written += 1

        print(f"    wrote {written} tiles")

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PNG tile pyramid from ESA WorldCover TIFs")
    parser.add_argument("--region", default="florida", help="Region name (default: florida)")
    parser.add_argument(
        "--zooms", default="6,7,8,9,10,11,12",
        help="Comma-separated zoom levels to generate (default: 6-12)",
    )
    parser.add_argument(
        "--tif-dir", default=None,
        help="Override TIF input directory",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Override output directory",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    repo_root = script_dir.parent          # mangrove/
    ml_root = repo_root.parent / "ml-mangrove"  # sibling ml-mangrove repo

    tif_dir = Path(args.tif_dir) if args.tif_dir else (
        ml_root / "Scale Adaption" / "GEE" / "data" / "pmtiles_pipeline" / args.region / "tifs"
    )
    out_dir = Path(args.out_dir) if args.out_dir else (
        repo_root / "landing" / "public" / "tiles" / args.region
    )

    zoom_levels = [int(z.strip()) for z in args.zooms.split(",")]

    print(f"TIF dir:  {tif_dir}")
    print(f"Out dir:  {out_dir}")
    print(f"Zooms:    {zoom_levels}")

    generate_tiles(tif_dir, out_dir, zoom_levels, args.region)


if __name__ == "__main__":
    main()
