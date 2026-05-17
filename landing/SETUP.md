# Setup Guide — Mangrove Monitor

Everything you need to go from a fresh clone to a fully running site with real model outputs.

---

## Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| Node.js | 20+ | `node -v` |
| npm | 10+ | `npm -v` |
| Git | any | `git --version` |

**For tile generation only** (not required to run the frontend):

| Tool | Purpose |
|------|---------|
| Python 3.10+ (conda `mangrove` env) | GEE download + vectorization |
| `earthengine-api geemap rasterio shapely tqdm pyyaml` | pip deps in the mangrove env |
| `tippecanoe` | GeoJSON → PMTiles conversion |

tippecanoe install:

```bash
brew install tippecanoe          # macOS
sudo apt install tippecanoe      # Ubuntu / WSL
# Windows without WSL: see the Docker fallback in generate_pmtiles.ipynb
```

---

## 1 — Clone and Install

```bash
git clone git@github.com:UCSD-E4E/ml-mangrove.git
cd ml-mangrove/landing
npm install
```

---

## 2 — Environment Variables

```bash
cp .env.example .env.local
```

Open `.env.local` and fill in:

```env
# Required — get a free token at mapbox.com (satellite basemap tiles)
VITE_MAPBOX_TOKEN=your_mapbox_token_here

# Optional — real image URLs for the SplitCompare before/after viewer
# Leave blank to show placeholder gradients instead
VITE_SPLIT_LEFT_URL=
VITE_SPLIT_RIGHT_URL=
```

No tile server is required. Classification overlays are served as static PMTiles files —
the browser fetches only the tiles it needs via HTTP range requests.

---

## 3 — Run the Frontend

```bash
npm run dev
```

Site is live at **http://localhost:5173**.

Without any PMTiles files configured, the TerrainViewer shows the Mapbox satellite
basemap only. Classification overlays appear once PMTiles files are wired in (see
**Adding Tiles** below).

---

## 4 — Adding Classification Tiles

Classification overlays are rendered from **PMTiles** — one static file per region,
no tile server needed.

### Option A — GEE WorldCover tiles (ESA land cover data)

Use the notebook at `Scale Adaption/GEE/generate_pmtiles.ipynb`:

```bash
# Activate the mangrove conda environment
conda activate mangrove

# Open the notebook, set REGION_KEY to any key in config/regions.yaml, run all cells
jupyter notebook "../Scale Adaption/GEE/generate_pmtiles.ipynb"
```

The notebook:
1. Connects to Google Earth Engine and downloads ESA WorldCover 10 m tiles for all
   mangrove-containing cells in the region
2. Vectorizes the raster to GeoJSON (class indices 1–5; background excluded)
3. Runs tippecanoe to produce a PMTiles archive (zoom 0–14)
4. Copies the result to `landing/public/tiles/` and prints the `regions.ts` snippet

> **GEE auth:** if the token is expired, uncomment `ee.Authenticate()` at the top of the
> notebook and re-run it.

### Option B — Model prediction tiles

Once model inference has run and outputs exist as GeoTIFFs (band 1 = class indices 0–5):

```bash
conda activate mangrove
python pipeline/vectorize.py \
  --input_dir /path/to/prediction_geotiffs/ \
  --output public/tiles/everglades.pmtiles
```

### Wire up the URL

In `src/data/regions.ts`, set `pmtilesUrl` for the region:

```ts
// Local dev (file in landing/public/tiles/)
{ id: 'everglades', pmtilesUrl: '/tiles/everglades.pmtiles', ... }

// Production (GCS public URL)
{ id: 'everglades', pmtilesUrl: 'https://storage.googleapis.com/e4e-mangrove-data/pmtiles/everglades.pmtiles', ... }
```

### Deploy tiles to GCS

```bash
gsutil cp public/tiles/everglades.pmtiles gs://e4e-mangrove-data/pmtiles/everglades.pmtiles
gsutil acl ch -u AllUsers:R gs://e4e-mangrove-data/pmtiles/everglades.pmtiles
```

Tiles can be served from any static host or CDN — the browser uses HTTP range requests,
so no server-side rendering is required.

---

## 5 — SplitCompare Images (optional)

The SplitCompare section shows a before/after drag comparison. To use real imagery:

1. Export two image crops of the same area — one at 10 m resolution (Sentinel-2 RGB),
   one at 0.35 m (SR-UNet output rendered as an image).
2. Host them anywhere with a public URL (GCS, S3, static server).
3. Set in `.env.local`:

```env
VITE_SPLIT_LEFT_URL=https://your-storage/split/everglades-10m.png
VITE_SPLIT_RIGHT_URL=https://your-storage/split/everglades-035m.png
```

Restart the dev server and the images appear in the viewer.

---

## 6 — Running Tests

```bash
# Unit tests (174 tests)
npm test

# Unit tests in watch mode
npm run test:watch

# E2E tests — Playwright auto-starts the dev server
npm run test:e2e

# Coverage report
npm run test:coverage
```

All unit and E2E tests must pass (exit 0) before any change is merged.

---

## 7 — Production Build

```bash
npm run build      # outputs to landing/dist/
npm run preview    # serves the dist locally at http://localhost:4173
```

Bundle sizes after `build`:

| Asset | Minified | Gzipped |
|-------|---------|---------|
| `index.html` | ~1 KB | ~0.5 KB |
| `index.css` | ~1.3 KB | ~0.6 KB |
| `index.js` | ~940 KB | ~290 KB |

The JS bundle is dominated by deck.gl. If bundle size becomes a concern, wrap
`TerrainViewer` with `React.lazy()` + `Suspense` to code-split it out of the main chunk.

---

## Troubleshooting

**Map shows black / nothing**
→ Check `VITE_MAPBOX_TOKEN` in `.env.local`. Open the browser console — a 401 from
Mapbox means the token is missing or invalid.

**Classification overlay doesn't appear**
→ `pmtilesUrl` for the active region is empty in `src/data/regions.ts`. Either generate
a PMTiles file (Section 4) or leave blank — the satellite basemap shows without it.

**tippecanoe not found on Windows**
→ Use WSL (`sudo apt install tippecanoe`) or uncomment the Docker fallback cell in
`generate_pmtiles.ipynb`.

**GEE authentication error**
→ Uncomment and run `ee.Authenticate()` at the top of the notebook, or verify that
`~/.config/earthengine/credentials` is valid.

**SplitCompare shows gradients instead of images**
→ `VITE_SPLIT_LEFT_URL` / `VITE_SPLIT_RIGHT_URL` are not set. That is correct default
behaviour — set those env vars and restart `npm run dev`.
