# Mangrove Monitoring — Landing Site

Interactive web map for visualizing mangrove classification results from the [UCSD Engineers for Exploration](https://e4e.ucsd.edu/) Mangrove Monitoring project.

## Repository layout

```
mangrove/
├── landing/          # React + deck.gl web app (Vite)
├── scripts/          # Data pipeline scripts
│   ├── generate_raster_tiles.py   # Build PNG tile pyramid from ESA WorldCover TIFs
│   └── generate_pmtiles.ipynb     # Build vector PMTiles via tippecanoe (WSL)
└── archive/          # Original UCSD-E4E/mangrove repo content (historical reference)
```

## Landing site

The landing site displays mangrove classification results on an interactive satellite basemap. Classification tiles are served as:

- **Raster PNG** (z0–12): pre-rendered 256×256 tiles generated from ESA WorldCover GeoTIFFs
- **Vector PMTiles** (z13+): single-file vector tiles for high-zoom detail

### Dev setup

```bash
cd landing
npm install
npm run dev
```

Tests:

```bash
npm test
```

### Building

```bash
npm run build
```

## Generating tiles

The tile generation scripts expect the `ml-mangrove` repo to be cloned as a sibling directory:

```
e4e-mangrove/
├── mangrove/       ← this repo
└── ml-mangrove/    ← ML pipeline repo (clone separately)
```

### Raster PNG tiles (z6–12)

Activate the `mangrove` conda environment, then:

```bash
python scripts/generate_raster_tiles.py --region florida --zooms 6,7,8,9,10,11,12
```

Output goes to `landing/public/tiles/{region}/`.

### Vector PMTiles

Requires [tippecanoe](https://github.com/felt/tippecanoe) via WSL on Windows.
Run `scripts/generate_pmtiles.ipynb` and place the resulting `.pmtiles` file in `landing/public/tiles/`.

## Class legend

| Class | Color | ESA value |
|-------|-------|-----------|
| Background | transparent | — |
| Tree Cover | dark green | 10 |
| Built-up | brown | 50 |
| Water | blue | 80 |
| Wetland | muted green | 90 |
| Mangrove | deep green | 95 |
