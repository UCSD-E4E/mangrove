# Mangrove Monitor
### UCSD Engineers for Exploration

An interactive research showcase for the E4E Lab's global mangrove segmentation work. The site communicates the output of a Sentinel-2-based land cover model — trained across six geographically diverse regions — to conservation ecologists, potential research collaborators, and the scientific public.

---

## What It Is

A scrolling editorial site, not a data dashboard. The centerpiece is an interactive satellite map that renders the lab's mangrove classification outputs as vector tiles. Supporting sections explain the model pipeline, show a 28× resolution comparison between coarse Sentinel-2 input and the SR-UNet output, and introduce the team.

---

## Tech Stack

| Layer | Choice |
|-------|--------|
| Framework | React 19 + Vite 6 |
| Language | TypeScript 5 |
| Globe (Hero) | deck.gl 9 (ScatterplotLayer, MapView) |
| Map (Terrain) | MapLibre GL 5 + deck.gl interleaved |
| Classification tiles | PMTiles (vector) — no tile server required |
| Tile client | `pmtiles` + `@loaders.gl/mvt` + deck.gl TileLayer |
| Scroll animations | GSAP + ScrollTrigger |
| Fonts | Instrument Serif, DM Sans, DM Mono (Google Fonts) |
| Unit tests | Vitest 2 + React Testing Library + happy-dom |
| E2E tests | Playwright (Chromium) |

---

## Project Structure

```
landing/
├── src/
│   ├── App.tsx                      — page assembly + ErrorBoundary wiring
│   ├── components/
│   │   ├── Nav.tsx                  — sticky nav with scroll-hide behaviour
│   │   ├── Hero.tsx                 — hero section with embedded GlobeViewer
│   │   ├── GlobeViewer.tsx          — deck.gl rotating globe (forwardRef + imperative handle)
│   │   ├── About.tsx                — animated stat counters (IntersectionObserver)
│   │   ├── TerrainViewer.tsx        — pinned GSAP scroll section with region controls
│   │   ├── MapViewer.tsx            — MapLibre satellite + deck.gl PMTiles overlay
│   │   ├── RegionHUD.tsx            — coordinate overlay for active region
│   │   ├── TerrainLegend.tsx        — colour legend sourced from classConfig
│   │   ├── RegionStrip.tsx          — region pill selector
│   │   ├── TerrainControls.tsx      — zoom-in / zoom-out / reset buttons
│   │   ├── SplitCompare.tsx         — draggable before/after image viewer
│   │   ├── Pipeline.tsx             — 4-step pipeline with GSAP stagger on scroll
│   │   ├── Team.tsx                 — researcher cards
│   │   ├── JoinCTA.tsx              — collaboration call-to-action
│   │   ├── Footer.tsx               — copyright + nav links
│   │   └── ErrorBoundary.tsx        — class component; wraps TerrainViewer for graceful WebGL fallback
│   ├── hooks/
│   │   ├── useNavHide.ts            — scroll-direction detection for nav visibility
│   │   ├── useCountUp.ts            — eased number animation driven by RAF
│   │   └── useScrollAnimations.ts   — GSAP ScrollTrigger choreography for SplitCompare / Team / JoinCTA
│   ├── data/
│   │   └── regions.ts               — coordinates, zoom, pitch, pmtilesUrl for all 6 regions
│   ├── lib/
│   │   └── classConfig.ts           — class names, deck.gl RGBA colours, legend items
│   └── styles/
│       └── globals.css              — design tokens, fonts, typography utilities
├── pipeline/
│   └── vectorize.py                 — model predictions → GeoJSON → PMTiles (tippecanoe)
├── e2e/                             — Playwright spec files (one per phase)
├── src/test/                        — Vitest test files (unit + hook tests)
├── .env.example                     — required environment variables
└── index.html                       — meta description, og:* tags, font preconnect
```

---

## Getting Started

### Prerequisites

- Node.js 20+
- npm 10+

### 1 — Install and run

```bash
cd landing
npm install
cp .env.example .env.local   # add VITE_MAPBOX_TOKEN
npm run dev
```

The site is available at `http://localhost:5173`.

Without any PMTiles files configured, the TerrainViewer shows the satellite basemap only. Classification overlays appear once PMTiles files are wired in (see **Adding Tiles** below).

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `VITE_MAPBOX_TOKEN` | Yes | Mapbox public token — satellite basemap tiles |
| `VITE_SPLIT_LEFT_URL` | No | Image URL for the "10m input" side of SplitCompare |
| `VITE_SPLIT_RIGHT_URL` | No | Image URL for the "SR output" side of SplitCompare |

---

## Adding Classification Tiles

Classification overlays are served as **PMTiles** — a single static file, no tile server required.

### 1 — Generate PMTiles from model predictions

```bash
# Activate the mangrove conda environment
conda activate mangrove

python pipeline/vectorize.py \
  --input_dir /path/to/prediction_geotiffs/ \
  --output public/tiles/everglades.pmtiles
```

The script:
- Reads band-1 class indices (0–5) from each prediction GeoTIFF
- Vectorises per-class polygons with rasterio
- Runs tippecanoe to produce a PMTiles archive (zoom 0–14)

> **Windows:** tippecanoe requires WSL or Docker.  
> Docker: `docker run --rm -v $(pwd):/data felt/tippecanoe tippecanoe ...`

### 2 — Wire up the URL

In `src/data/regions.ts`, set `pmtilesUrl` for the region:

```ts
// Local dev (file in landing/public/tiles/)
{ id: 'everglades', pmtilesUrl: '/tiles/everglades.pmtiles', ... }

// Production (GCS public URL)
{ id: 'everglades', pmtilesUrl: 'https://storage.googleapis.com/bucket/tiles/everglades.pmtiles', ... }
```

### 3 — Deploy

Upload the `.pmtiles` file to GCS or Cloudflare R2. The browser fetches only the tiles it needs via HTTP range requests — no server-side rendering.

---

## Testing

```bash
# Unit tests — 174 tests
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

## Regions

Six mangrove regions are defined in [`src/data/regions.ts`](src/data/regions.ts). Each entry holds:

| Field | Description |
|-------|-------------|
| `id` | Unique slug (`everglades`, `sundarbans`, …) |
| `name` | Short display name |
| `displayName` | Full human-readable name |
| `coords` | `{ lng, lat }` for the camera fly-to target |
| `zoom` | MapLibre zoom level on arrival |
| `pitch` | Camera pitch in degrees |
| `pmtilesUrl` | HTTPS / GCS / local path to the `.pmtiles` file; `''` = no tiles yet |
| `status` | `'trained'` or `'testing'` |

To add a region: append an entry to the `regions` array and generate a PMTiles file via `pipeline/vectorize.py`.

---

## Class Configuration

All class metadata lives in [`src/lib/classConfig.ts`](src/lib/classConfig.ts) — the single source of truth for:

| Export | Used by |
|--------|---------|
| `CLASS_NAMES` | tooltips, analytics |
| `CLASS_COLORS` | `MapViewer` `getFillColor` |
| `LEGEND_ITEMS` | `TerrainLegend` |

Classes follow the GEE pipeline's `ESA_TO_CLASS` mapping:

| Index | Class | ESA code |
|-------|-------|----------|
| 0 | Background | 20, 30, 40, 60, 70, 100 |
| 1 | Tree Cover | 10 |
| 2 | Built-up | 50 |
| 3 | Water | 80 |
| 4 | Wetland | 90 |
| 5 | Mangrove | 95 |

---

## Design System

All tokens live in `src/styles/globals.css`. Never use hardcoded colour or font values in component styles — reference the tokens.

**The single most important rule:** this is a research site, not a SaaS product. No glowing effects, no dashboard cards, no blue/purple gradients. The terrain and globe are the design — everything else gets out of the way.

---

## Contributing

- All frontend code is TypeScript with strict mode enabled.
- New sections go in `src/components/` with a matching test file in `src/test/` and an E2E spec in `e2e/`.
- Hooks go in `src/hooks/` and must have their own unit test file.
- Class metadata goes in `src/lib/classConfig.ts` — do not hardcode colours elsewhere.
- Region data lives in `src/data/regions.ts` — single source of truth.
- Design tokens and typography utilities live in `src/styles/globals.css`.

---

## Related

- [E4E Global Mangrove Observatory](../Observatory/) — the full-featured data exploration tool for the Florida dataset
- [GEE Scale Adaptation Pipeline](../Scale%20Adaption/GEE/) — the multi-region training pipeline that produces the predictions shown here
