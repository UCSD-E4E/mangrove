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

## Deploying

The site is automatically deployed to [GitHub Pages](https://ucsd-e4e.github.io/mangrove/) on every push to `main` via `.github/workflows/deploy.yml`.

Tiles (PNG + PMTiles) are served from **Cloudflare R2** and are not committed to the repo.

### First-time R2 setup

1. **Create a Cloudflare account** at [cloudflare.com](https://cloudflare.com) if you don't have one.

2. **Create an R2 bucket**
   - Cloudflare dashboard → R2 → Create bucket
   - Name it (e.g. `e4e-mangrove`)

3. **Enable public access**
   - Open the bucket → Settings → **Public Development URL** → Enable
   - Copy the URL shown (format: `https://pub-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx.r2.dev`)

4. **Create an R2 API token**
   - R2 → Manage R2 API Tokens → Create API token → **User API Token**
   - Permissions: **Object Read & Write**
   - Scope: your bucket
   - Copy the **Access Key ID** and **Secret Access Key** (shown once only)

5. **Configure rclone** (install from [rclone.org](https://rclone.org/downloads/))
   ```
   rclone config
   ```
   - New remote → name: `r2-e4e-mangrove`
   - Storage type: `s3` → Provider: `Cloudflare`
   - Enter credentials from step 4 when prompted
   - Region: leave blank
   - Endpoint: `https://<account-id>.r2.cloudflarestorage.com`
     (account ID is in the Cloudflare dashboard URL)
   - Leave all other options as default

6. **Upload tiles** (from repo root, on Windows use PowerShell directly)
   ```bash
   bash scripts/upload_tiles.sh florida
   ```
   Or on Windows PowerShell:
   ```powershell
   .\rclone sync "landing\public\tiles\florida" r2-e4e-mangrove:e4e-mangrove/tiles/florida --progress --transfers=16 --s3-no-check-bucket
   .\rclone copyto "landing\public\tiles\florida.pmtiles" r2-e4e-mangrove:e4e-mangrove/tiles/florida.pmtiles --progress --s3-no-check-bucket
   ```

7. **Add GitHub secrets**
   - Repo → Settings → Secrets and variables → Actions → New repository secret
   - `VITE_MAPBOX_TOKEN` — your Mapbox public token (required for satellite basemap)
   - `VITE_TILES_BASE_URL` — the R2 public URL from step 3 (e.g. `https://pub-xxx.r2.dev`)

8. **Add to `landing/.env.local`** for local dev pointing at R2:
   ```
   VITE_TILES_BASE_URL=https://pub-xxx.r2.dev
   ```
   Or leave it unset to use local tiles from `landing/public/tiles/`.

After that, push to `main` and the site will deploy pointing at R2 for tiles.

### Local development

```bash
cd landing && npm run dev
```

Tiles are served from R2 if `VITE_TILES_BASE_URL` is set in `landing/.env.local`, otherwise from `landing/public/tiles/`.

## Class legend

| Class | Color | ESA value |
|-------|-------|-----------|
| Background | transparent | — |
| Tree Cover | dark green | 10 |
| Built-up | brown | 50 |
| Water | blue | 80 |
| Wetland | muted green | 90 |
| Mangrove | deep green | 95 |
