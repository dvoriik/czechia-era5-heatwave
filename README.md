# Czechia's record heatwave (June 2026) — ERA5 on 3D terrain

Hourly **ERA5 2 m air temperature** for the record-breaking Czech heatwave of
**25 June – 2 July 2026**, sharpened to 250 m using the terrain and animated over a
3D relief of Czechia with [forge3d](https://github.com/milos-agathon/forge3d).

The national record of **41.9 °C at Doksany on 28 June 2026** (ČHMÚ station measurement)
fell during this event.

![Animated preview](docs/heatwave.gif)

Two versions are included:

| Draped temperature on real terrain | Temperature as 3D columns |
|---|---|
| ![Drape version](docs/preview_drape.jpg) | ![Columns version](docs/preview_columns.jpg) |
| `czechia_era5_heatwave.py` | `czechia_era5_heatwave_3d.py` |
| Colour = temperature, height = real relief (DEM) | Colour **and** height = temperature (6 km cells) |

---

## Data sources

All sources are open and need no API key or account.

| Data | Source | Used for |
|---|---|---|
| 2 m temperature, hourly, 0.25° | ECMWF **ERA5** (Copernicus C3S), read from Google's public [ARCO-ERA5](https://github.com/google-research/arco-era5) Zarr mirror (`gs://gcp-public-data-arco-era5`) | the temperature field |
| Surface geopotential | same ERA5 store (`geopotential_at_surface`) | ERA5's own model terrain height, needed for the height correction |
| Elevation | [AWS Terrain Tiles](https://registry.opendata.aws/terrain-tiles/) (Terrarium PNG, zoom 9) | 250 m DEM of Czechia |
| Border | [OpenStreetMap](https://www.openstreetmap.org/relation/51684) relation 51684, via Nominatim | country mask |

Only a small window over Czechia is downloaded (about 0.4 MB of ERA5 and 84 terrain tiles).

---

## Method

1. **Terrain.** Terrarium tiles are mosaicked, decoded to metres and reprojected to
   EPSG:3035 (LAEA Europe) at 250 m. The OSM border gives the country mask.
2. **ERA5.** Hourly `2m_temperature` for the event window plus `geopotential_at_surface`
   (÷ g = model terrain height) are read from ARCO-ERA5.
3. **Downscaling 25 km → 250 m.** ERA5 is interpolated onto the 250 m grid (cubic spline)
   and corrected for the difference between real and model elevation with a standard
   lapse rate of 6.5 °C per km:

   ```
   T_250m = T_ERA5 + 0.0065 · (z_ERA5 − z_DEM)
   ```

   This brings out warmer valleys and cooler ridges. It is a height-based estimate, not
   new information: ERA5 still underestimates local station extremes.
4. **Time.** Two frames per ERA5 hour; temperature is linearly interpolated in between.
5. **Rendering (forge3d).** The DEM is loaded once into the forge3d viewer
   (`open_viewer_async` → `load_terrain`). For every frame a coloured RGBA temperature
   raster is draped with `load_overlay` and captured with `snapshot`. Camera, sun and
   shadows stay fixed.
6. **Colour calibration.** The lit overlay does not reproduce input colours 1:1, so each
   colour stop of the scale is rendered over the whole country, the on-screen colour is
   measured and the input colour is corrected (4 iterations). The legend shows the
   colours as they actually appear on the map.
7. **City labels.** Screen positions come from probe renders: a white dot on a black
   country, diffed against the same render without the dot.
8. **Layout and video.** Title, local clock (CEST), hottest cell, legend, hourly-maximum
   sparkline and credits are drawn with Pillow; ffmpeg encodes H.264 at 24 fps.

The **3D columns** version reuses steps 1–4, then averages temperature into 6 km cells and
writes a new heightfield every frame: column height ∝ (T − 8 °C), colour = T. The camera
target is locked so the view does not jump between frames, and the camera projection is
recovered once (DLT from probe renders) so labels can ride on the column tops.

---

## Code structure

`czechia_era5_heatwave.py` (one file, ~530 lines):

| Function | What it does |
|---|---|
| `fetch_boundary` | downloads the Czech border from OSM |
| `build_dem` | builds the 250 m DEM from Terrarium tiles, reprojects to EPSG:3035, makes the mask |
| `fetch_era5` | reads ERA5 temperature and model terrain height from ARCO-ERA5 |
| `Downscaler` | ERA5 → 250 m grid with lapse-rate correction |
| `colorize`, `overlay_rgba` | temperature → colours |
| `open_viewer`, `snap_overlay` | drive the forge3d viewer |
| `calibrate_ramp` | colour calibration against the rendered output |
| `locate_cities` | on-screen city positions and country mask |
| `compose` | final frame layout |
| `main` | frame loop and ffmpeg encode |

`czechia_era5_heatwave_3d.py` imports the data and downscaling parts from the first
script (keep both files in the same folder) and adds `Columns` (6 km column heightfield), `calibrate_ramp`, `fit_camera` and its
own `compose`.

Settings worth changing are constants at the top of each script: `START` / `END`
(event window, UTC), `T_STOPS` (colour scale), `TERRAIN` (camera, sun, exaggeration),
`CITIES`, `LAPSE_RATE`, `CELL` and `M_PER_DEG` (3D version).

---

## How to run

```bash
pip install -r requirements.txt
# ffmpeg must be on PATH

python czechia_era5_heatwave.py --preview 2026-06-28T14   # one still -> out/preview.png
python czechia_era5_heatwave.py                            # full video -> out/*.mp4

python czechia_era5_heatwave_3d.py --preview 2026-06-28T14
python czechia_era5_heatwave_3d.py                         # -> out3d/*.mp4
```

- Downloads and intermediate results are cached in `data/`, `out/work/` and `out3d/work/`.
  A second run is fast, and an interrupted render resumes from the last finished frame.
- The first run also does the colour calibration and label probes (~15 min on CPU).
- **Headless Linux / no GPU:** forge3d's viewer needs a window and a Vulkan adapter. Use a
  virtual display and Mesa's software rasteriser:

  ```bash
  sudo apt-get install xvfb mesa-vulkan-drivers
  Xvfb :99 -screen 0 1920x1080x24 &
  export DISPLAY=:99 XDG_RUNTIME_DIR=/tmp/xdg && mkdir -p -m 700 $XDG_RUNTIME_DIR
  ```

  On CPU-only rendering (lavapipe) a frame takes 15–35 s, so the full 361-frame video
  takes 1.5–3 h. With a GPU it is much faster.

---

## Caveats

- ERA5 is a ~25 km reanalysis. The 250 m detail comes only from the elevation correction,
  so peak values are smoother than station records (hottest cell here: 41.8 °C on
  28 June at 16:00 CEST, vs. the 41.9 °C station record).
- A fixed lapse rate ignores inversions (e.g. cold valley nights) and very steep afternoon
  lapse rates.
- The forge3d viewer accumulates GPU textures, so the scripts restart it every 40 frames.

---

## Credits and licences

- Rendering: [forge3d](https://github.com/milos-agathon/forge3d) by **Milos Popovic, PhD**
  (Apache-2.0 / MIT). This project only uses the open viewer API.
- Contains modified Copernicus Climate Change Service information (2026). Neither the
  European Commission nor ECMWF is responsible for any use of this information.
  ERA5 via Google ARCO-ERA5.
- Elevation: AWS Terrain Tiles, see the
  [source attribution](https://github.com/tilezen/joerd/blob/master/docs/attribution.md)
  (includes Copernicus EU-DEM, SRTM and others).
- Border: © OpenStreetMap contributors, ODbL.
- Record value: Czech Hydrometeorological Institute (ČHMÚ).

Code: MIT, see [LICENSE](LICENSE).

---

## Česky (stručně)

Animace rekordní vlny veder v Česku (25. 6. – 2. 7. 2026). Hodinová teplota ve 2 m
z ERA5 (Copernicus) je zjemněná na 250 m podle nadmořské výšky (−6,5 °C na 1000 m)
a vykreslená na 3D reliéfu ČR knihovnou forge3d (Miloš Popović, PhD). Druhá verze
zobrazuje teplotu jako 3D sloupce 6 × 6 km. Všechna data jsou volně dostupná bez
registrace, postup spuštění je výše.
