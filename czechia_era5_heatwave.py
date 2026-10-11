#!/usr/bin/env python3
"""Czechia, June 2026 record heatwave — ERA5 2 m temperature animated over 3D terrain.

forge3d renders every frame: the Czech DEM is loaded as terrain and a
temperature raster is draped on it as an overlay that is swapped each frame.

Data (all keyless / open):
  * ERA5 hourly 2 m temperature + surface geopotential, via Google's public
    ARCO-ERA5 Zarr mirror (gs://gcp-public-data-arco-era5).
  * Terrain: AWS Terrain Tiles (Terrarium PNG), zoom 9 (~250 m after reprojection).
  * Border: OpenStreetMap relation 51684 via Nominatim.

ERA5 is 0.25° (~20 km), so it is downscaled to the 250 m DEM with a standard
environmental lapse rate (6.5 K/km) against ERA5's own orography:
    T_250m = T_era5(interp) + 0.0065 * (z_era5(interp) - z_dem)
This adds valley/ridge structure; it does not add information ERA5 lacks
(station peaks such as Doksany's 41.9 °C are underestimated by a gridded
reanalysis).

Usage:
    python czechia_era5_heatwave.py                    # full MP4
    python czechia_era5_heatwave.py --preview "2026-06-29T15"   # single still
Requirements: forge3d, rasterio, xarray, zarr, gcsfs, scipy, pyproj, shapely,
pillow, ffmpeg on PATH. Headless Linux: Xvfb + Mesa lavapipe.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# --------------------------------------------------------------------------- config
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "out"

START = np.datetime64("2026-06-25T00")  # UTC
END = np.datetime64("2026-07-02T12")    # UTC, inclusive
SUBFRAMES = 2                           # frames per ERA5 hour (linear time interp)
FPS = 24
WIDTH, HEIGHT = 1920, 1080

DEM_ZOOM = 9
DEM_RES_M = 250.0
DEM_UNIT_SCALE = 10.0   # viewer world units = metres / 10 (keeps camera < 1e6 units)
LAPSE_RATE = 0.0065     # K per metre

TITLE = "Czechia's record heatwave"
SUBTITLE = "ERA5 hourly 2 m air temperature, downscaled to 250 m terrain"
CREDIT = ("Data: ECMWF ERA5 (Copernicus C3S) via Google ARCO-ERA5 · Terrain: AWS Terrain Tiles · "
          "Border: © OpenStreetMap · Rendered with forge3d")

# temperature ramp (°C, hex) — stays luminous at the hot end so terrain shading reads
T_STOPS = [
    (8, "#23407a"), (12, "#2f6fae"), (16, "#58a7d3"), (19, "#9fd5e3"),
    (22, "#e9f2c8"), (25, "#fde28a"), (28, "#fdb85b"), (31, "#f88c42"),
    (34, "#f0603a"), (37, "#e3385a"), (40, "#cc2f86"), (43, "#9a3aa8"),
]
T_MIN, T_MAX = T_STOPS[0][0], T_STOPS[-1][0]

CITIES = [  # name, lon, lat
    ("Praha", 14.4213, 50.0875),
    ("Brno", 16.6068, 49.1951),
    ("Ostrava", 18.2820, 49.8209),
    ("Plzeň", 13.3776, 49.7384),
    ("Doksany", 14.1606, 50.4555),
    ("Sněžka", 15.7400, 50.7360),
    ("České Budějovice", 14.4747, 48.9745),
]

TERRAIN = {
    "cmd": "set_terrain",
    "phi": 90.0, "theta": 44.0, "radius": 330000.0, "fov": 33.0,
    "zscale": 0.45,
    "sun_azimuth": 315.0, "sun_elevation": 32.0, "sun_intensity": 1.0,
    "ambient": 0.55, "shadow": 0.45,
    "background": [1.0, 1.0, 1.0],
}
PBR = {
    "cmd": "set_terrain_pbr", "enabled": True,
    "shadow_technique": "pcss", "shadow_map_res": 4096, "exposure": 2.0, "msaa": 4,
    "ibl_intensity": 0.30, "normal_strength": 1.6,
    "height_ao": {"enabled": True, "directions": 12, "steps": 32, "max_distance": 200.0,
                  "strength": 0.6, "resolution_scale": 0.75},
    "tonemap": {"operator": "aces", "white_point": 5.0, "white_balance_enabled": True,
                "temperature": 6500.0, "tint": 0.0},
}

UA = "forge3d-czechia-era5/1.0"


# --------------------------------------------------------------------------- data
def fetch_boundary() -> dict:
    path = DATA / "czechia.geojson"
    if not path.exists():
        url = ("https://nominatim.openstreetmap.org/lookup?osm_ids=R51684&format=geojson"
               "&polygon_geojson=1&polygon_threshold=0.001")
        path.write_bytes(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA})).read())
    return json.loads(path.read_text())["features"][0]["geometry"]


def build_dem(geom_wgs84: dict) -> dict:
    """Terrarium mosaic -> EPSG:3035 250 m grid. Returns grid info + arrays."""
    import rasterio
    from rasterio.features import geometry_mask
    from rasterio.transform import from_bounds
    from rasterio.warp import Resampling, reproject, transform_geom
    from shapely.geometry import shape

    scaled = DATA / "czechia_dem_scaled.tif"
    meta = DATA / "dem_grid.npz"
    if scaled.exists() and meta.exists():
        z = np.load(meta)
        return {"path": scaled, "dem": z["dem"], "inside": z["inside"],
                "transform": rasterio.Affine(*z["transform"][:6])}

    W, S, E, N = 11.9, 48.45, 19.0, 51.15
    Z = DEM_ZOOM
    tx = lambda lon: int((lon + 180) / 360 * 2**Z)
    ty = lambda lat: int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * 2**Z)
    xs, ys = range(tx(W), tx(E) + 1), range(ty(N), ty(S) + 1)
    tiles = DATA / "tiles"
    tiles.mkdir(parents=True, exist_ok=True)
    mosaic = np.zeros((256 * len(ys), 256 * len(xs)), np.float32)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            p = tiles / f"{Z}_{x}_{y}.png"
            if not p.exists():
                p.write_bytes(urllib.request.urlopen(
                    f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{Z}/{x}/{y}.png").read())
            a = np.asarray(Image.open(p).convert("RGB"), np.float32)
            mosaic[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768
    R = 6378137.0
    ts = 2 * math.pi * R / 2**Z
    src_tr = rasterio.Affine(ts / 256, 0, -math.pi * R + xs[0] * ts, 0, -ts / 256, math.pi * R - ys[0] * ts)

    g3035 = transform_geom("EPSG:4326", "EPSG:3035", geom_wgs84)
    b = shape(g3035).bounds
    pad = 6000.0
    L, B, Rr, T = b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad
    w, h = int((Rr - L) / DEM_RES_M), int((T - B) / DEM_RES_M)
    dst_tr = from_bounds(L, B, L + w * DEM_RES_M, B + h * DEM_RES_M, w, h)
    dem = np.zeros((h, w), np.float32)
    reproject(mosaic, dem, src_transform=src_tr, src_crs="EPSG:3857", dst_transform=dst_tr,
              dst_crs="EPSG:3035", resampling=Resampling.bilinear)
    inside = ~geometry_mask([g3035], (h, w), dst_tr)

    # forge3d viewer: keep world extent < 1e6 units -> write a unit-scaled, CRS-less copy
    dm = np.where(inside, dem, np.nan).astype(np.float32)
    s = DEM_RES_M / DEM_UNIT_SCALE
    prof = dict(driver="GTiff", width=w, height=h, count=1, dtype="float32", nodata=np.nan,
                transform=rasterio.Affine(s, 0, 0, 0, -s, 0), compress="deflate")
    with rasterio.open(scaled, "w", **prof) as d:
        d.write(dm, 1)
    np.savez(meta, dem=dem, inside=inside, transform=np.array(tuple(dst_tr)))
    return {"path": scaled, "dem": dem, "inside": inside, "transform": dst_tr}


def fetch_era5() -> tuple:
    import xarray as xr
    path = DATA / "era5_t2m.nc"
    if not path.exists():
        ds = xr.open_zarr("gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
                          chunks=None, storage_options={"token": "anon"})
        box = dict(latitude=slice(51.75, 48.0), longitude=slice(11.25, 19.75))
        t2m = ds["2m_temperature"].sel(time=slice(str(START), str(END)), **box).load()
        zs = ds["geopotential_at_surface"].sel(time="2020-01-01T00", **box).load() / 9.80665
        xr.Dataset({"t2m": t2m, "zs": zs.drop_vars("time", errors="ignore")}).to_netcdf(path)
    ds = xr.open_dataset(path)
    lat = ds.latitude.values[::-1]
    lon = ds.longitude.values
    return ds.time.values, lat, lon, ds.t2m.values[:, ::-1, :] - 273.15, ds.zs.values[::-1, :]


class Downscaler:
    """Cubic-spline ERA5 -> DEM grid with lapse-rate correction (fractional-index sampling)."""

    def __init__(self, grid: dict, lat, lon, zs):
        import rasterio
        from pyproj import Transformer
        from scipy.ndimage import map_coordinates
        self.map_coordinates = map_coordinates
        dem = grid["dem"]
        h, w = dem.shape
        rows, cols = np.mgrid[0:h, 0:w]
        X, Y = rasterio.transform.xy(grid["transform"], rows.ravel(), cols.ravel())
        lo, la = Transformer.from_crs(3035, 4326, always_xy=True).transform(np.asarray(X), np.asarray(Y))
        # lat ascending, lon ascending, regular 0.25° grid -> fractional indices
        self.coords = np.vstack([(la - lat[0]) / (lat[1] - lat[0]), (lo - lon[0]) / (lon[1] - lon[0])])
        self.shape = (h, w)
        zi = map_coordinates(zs, self.coords, order=1, mode="nearest").reshape(h, w)
        self.correction = (LAPSE_RATE * (zi - dem)).astype(np.float32)
        self.inside = grid["inside"]

    def __call__(self, t_field: np.ndarray) -> np.ndarray:
        t = self.map_coordinates(np.asarray(t_field, np.float64), self.coords, order=3, mode="nearest")
        return t.reshape(self.shape) + self.correction


# --------------------------------------------------------------------------- colour
def _hex(c):
    return np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)], np.float32)


_LUT_T = np.linspace(T_MIN, T_MAX, 1024)
_LUT = np.stack([np.interp(_LUT_T, [s[0] for s in T_STOPS], [_hex(s[1])[k] for s in T_STOPS]) for k in range(3)], -1)


def colorize(t: np.ndarray) -> np.ndarray:
    idx = np.clip(((t - T_MIN) / (T_MAX - T_MIN) * 1023).round().astype(int), 0, 1023)
    return _LUT[idx].astype(np.uint8)


_LUT_IN = _LUT.copy()  # replaced by calibrate_ramp()


def overlay_rgba(t: np.ndarray, inside: np.ndarray) -> np.ndarray:
    rgba = np.zeros(t.shape + (4,), np.uint8)
    idx = np.clip(((t - T_MIN) / (T_MAX - T_MIN) * 1023).round().astype(int), 0, 1023)
    rgba[..., :3] = _LUT_IN[idx].astype(np.uint8)
    rgba[..., 3] = np.where(inside, 255, 0)
    return rgba


# --------------------------------------------------------------------------- render
def open_viewer(dem_path: Path):
    import forge3d as f3d
    v = f3d.open_viewer_async(width=WIDTH, height=HEIGHT, timeout=180)
    v.load_terrain(str(dem_path.resolve()))
    v.send_ipc(TERRAIN)
    v.send_ipc(PBR)
    return v


def snap_overlay(v, rgba: np.ndarray, out_png: Path, tmp: Path) -> None:
    Image.fromarray(rgba, "RGBA").save(tmp)
    v.load_overlay("t2m", str(tmp.resolve()), extent=(0.0, 0.0, 1.0, 1.0), opacity=1.0, preserve_colors=True)
    v.send_ipc({"cmd": "set_overlays_enabled", "enabled": True})
    v.send_ipc({"cmd": "set_overlay_solid", "solid": True})
    v.snapshot(str(out_png.resolve()), width=WIDTH, height=HEIGHT)


def _srgb_to_lin(c):
    c = np.asarray(c, np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def calibrate_ramp(v, grid: dict, mask: np.ndarray, work: Path, iters: int = 4) -> None:
    """The viewer's lit overlay does not reproduce input colours 1:1 (lighting + tonemap
    darken and shift saturated colours). For each ramp stop, render the whole country in
    that colour, measure the median on-screen colour, and nudge the input colour until the
    rendered median matches the legend colour. The corrected stops drive the overlay; the
    legend keeps the target colours, so what you see on a typical slope matches the legend."""
    global _LUT_IN
    cal = work / "ramp_calibration.json"
    if cal.exists():
        stops_in = np.array(json.loads(cal.read_text())["stops_in"], np.float32)
    else:
        h, w = grid["dem"].shape
        m_small = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).resize((WIDTH // 3, HEIGHT // 3))) > 200
        targets = np.array([_hex(c) for _, c in T_STOPS], np.float32)
        stops_in = targets.copy()
        rendered = targets.copy()
        for it in range(iters):
            for k in range(len(T_STOPS)):
                ov = np.zeros((h, w, 4), np.uint8)
                ov[grid["inside"]] = (*np.clip(stops_in[k], 0, 255).astype(np.uint8), 255)
                Image.fromarray(ov, "RGBA").save(work / "cal_ov.png")
                v.load_overlay("t2m", str((work / "cal_ov.png").resolve()), extent=(0.0, 0.0, 1.0, 1.0),
                               opacity=1.0, preserve_colors=True)
                v.send_ipc({"cmd": "set_overlays_enabled", "enabled": True})
                v.send_ipc({"cmd": "set_overlay_solid", "solid": True})
                v.snapshot(str((work / "cal.png").resolve()), width=WIDTH // 3, height=HEIGHT // 3)
                got = np.median(np.asarray(Image.open(work / "cal.png").convert("RGB"), np.float32)[m_small], 0)
                rendered[k] = got
                stops_in[k] = np.clip(stops_in[k] + 1.1 * (targets[k] - got), 0, 255)
                if it == iters - 1:
                    print(f"  stop {T_STOPS[k][0]:>2}°C target {targets[k].astype(int)} rendered {got.astype(int)}", flush=True)
        cal.write_text(json.dumps({"stops_in": stops_in.tolist(), "rendered": rendered.tolist()}))
    # legend shows the colours actually measured on screen (after the final correction)
    rendered = np.array(json.loads(cal.read_text())["rendered"], np.float32)
    _LUT[:] = np.stack([np.interp(_LUT_T, [t for t, _ in T_STOPS], rendered[:, c]) for c in range(3)], -1)
    _LUT_IN = np.stack([np.interp(_LUT_T, [t for t, _ in T_STOPS], stops_in[:, c]) for c in range(3)], -1)


def locate_cities(v, grid: dict, work: Path) -> tuple[dict, np.ndarray]:
    """Probe passes. A black country overlay gives the on-screen country mask; then one
    bright dot per city on that overlay, diffed against the baseline, gives each city's
    exact screen position (so labels sit on the rendered 3D surface)."""
    cache = work / "city_screen.json"
    mask_path = work / "country_mask.png"
    import rasterio
    from pyproj import Transformer
    from scipy.ndimage import binary_opening
    h, w = grid["dem"].shape
    base_rgba = np.zeros((h, w, 4), np.uint8)
    base_rgba[..., 3] = np.where(grid["inside"], 255, 0)
    if not mask_path.exists():
        snap_overlay(v, base_rgba, work / "probe_base.png", work / "probe_overlay.png")
        base = np.asarray(Image.open(work / "probe_base.png").convert("RGB"), np.float32)
        m = binary_opening(base.mean(-1) < 90, iterations=1)
        Image.fromarray((m * 255).astype(np.uint8)).save(mask_path)
    mask = np.asarray(Image.open(mask_path)) > 127
    calibrate_ramp(v, grid, mask, work)
    if cache.exists():
        return json.loads(cache.read_text()), mask
    base = np.asarray(Image.open(work / "probe_base.png").convert("RGB"), np.float32)
    tr = Transformer.from_crs(4326, 3035, always_xy=True)
    yy, xx = np.ogrid[:h, :w]
    out = {}
    for name, lon, lat in CITIES:
        x, y = tr.transform(lon, lat)
        r, c = rasterio.transform.rowcol(grid["transform"], x, y)
        rgba = base_rgba.copy()
        rgba[(yy - r) ** 2 + (xx - c) ** 2 <= 100] = (255, 255, 255, 255)
        p = work / f"probe_{name}.png"
        snap_overlay(v, rgba, p, work / "probe_overlay.png")
        diff = np.abs(np.asarray(Image.open(p).convert("RGB"), np.float32) - base).max(-1)
        ys_, xs_ = np.nonzero(diff > 60)
        if len(xs_):
            wgt = diff[ys_, xs_]
            out[name] = [float((xs_ * wgt).sum() / wgt.sum()), float((ys_ * wgt).sum() / wgt.sum())]
        print(f"  probe {name}: {out.get(name)}", flush=True)
    cache.write_text(json.dumps(out))
    return out, mask


# --------------------------------------------------------------------------- compose
def font(size: int, bold: bool = False):
    for n in (["DejaVuSans-Bold.ttf", "Arial Bold.ttf"] if bold else ["DejaVuSans.ttf", "Arial.ttf"]):
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
    return ImageFont.load_default()


INK = (34, 36, 42)
MUTED = (104, 108, 118)


def mask_box(mask: np.ndarray, pad: int = 10) -> tuple:
    ys, xs = np.nonzero(mask)
    return (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(mask.shape[1], int(xs.max()) + pad + 1), min(mask.shape[0], int(ys.max()) + pad + 1))


BG = (250, 249, 246)


def compose(raw: Path, out: Path, *, when_utc: np.datetime64, t_grid: np.ndarray, ds: Downscaler,
            mask: np.ndarray, cities: dict, series: dict, frame_pos: float) -> None:
    W, H = WIDTH, HEIGHT
    box = mask_box(mask)
    canvas = Image.new("RGB", (W, H), BG)
    src = Image.open(raw).convert("RGB").crop(box)
    alpha = Image.fromarray((mask * 255).astype(np.uint8)).crop(box).filter(ImageFilter.GaussianBlur(0.8))
    scale = min((W * 0.84) / src.width, (H * 0.68) / src.height)
    cw, ch = int(src.width * scale), int(src.height * scale)
    src = src.resize((cw, ch), Image.Resampling.LANCZOS)
    alpha = alpha.resize((cw, ch), Image.Resampling.LANCZOS)
    ox, oy = (W - cw) // 2, 200
    # soft contact shadow under the slab
    sh = Image.new("L", (W, H), 0)
    sh.paste(alpha, (ox + 6, oy + 14))
    sh = sh.filter(ImageFilter.GaussianBlur(14)).point(lambda v: int(v * 0.28))
    canvas = Image.composite(Image.new("RGB", (W, H), (60, 55, 50)), canvas, sh)
    canvas.paste(src, (ox, oy), alpha)
    d = ImageDraw.Draw(canvas)

    # city labels
    fl = font(19, True)
    for name, (sx, sy) in cities.items():
        px = ox + (sx - box[0]) * scale
        py = oy + (sy - box[1]) * scale
        d.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(255, 255, 255), outline=INK, width=2)
        d.text((px + 9, py - 12), name, font=fl, fill=INK, stroke_width=3, stroke_fill=(255, 255, 255))
        if name == "Doksany":
            note = "41.9 °C station record, 28 Jun"
            bb = d.textbbox((px + 9, py + 10), note, font=font(14), stroke_width=2)
            d.rounded_rectangle((bb[0] - 4, bb[1] - 2, bb[2] + 4, bb[3] + 2), radius=4, fill=(255, 255, 255))
            d.text((px + 9, py + 10), note, font=font(14), fill=INK)

    # title block
    d.text((70, 52), TITLE, font=font(46, True), fill=INK)
    d.text((72, 112), SUBTITLE, font=font(22), fill=MUTED)

    # clock (local time CEST = UTC+2)
    local = (when_utc + np.timedelta64(2, "h")).astype("datetime64[m]").astype(dt.datetime)
    d.text((W - 70, 50), local.strftime("%a %d %b %Y"), font=font(34, True), fill=INK, anchor="ra")
    d.text((W - 70, 94), local.strftime("%H:%M") + " CEST", font=font(30), fill=MUTED, anchor="ra")

    # national max readout
    inside = ds.inside
    tmax = float(np.nanmax(np.where(inside, t_grid, np.nan)))
    d.text((W - 70, 146), f"Hottest 250 m cell  {tmax:4.1f} °C", font=font(22, True), fill=tuple(int(c) for c in colorize(np.array([tmax]))[0]) if tmax > 30 else INK, anchor="ra")

    # legend bar (bottom left)
    lx, ly, lw, lh = 70, H - 118, 520, 16
    bar = colorize(np.linspace(T_MIN, T_MAX, lw)[None, :].repeat(lh, 0))
    canvas.paste(Image.fromarray(bar), (lx, ly))
    d.rectangle((lx, ly, lx + lw, ly + lh), outline=(200, 200, 200))
    d.text((lx, ly - 30), "2 m temperature (°C)", font=font(18, True), fill=INK)
    for t in range(10, 44, 5):
        x = lx + (t - T_MIN) / (T_MAX - T_MIN) * lw
        d.line((x, ly + lh, x, ly + lh + 5), fill=MUTED)
        d.text((x, ly + lh + 8), str(t), font=font(15), fill=MUTED, anchor="ma")

    # sparkline: daily cycle of national max, with cursor (bottom right)
    sx0, sy0, sw, sh = W - 70 - 640, H - 150, 640, 70
    vals = series["max"]
    vmin, vmax = 14.0, 42.0
    pts = [(sx0 + i / (len(vals) - 1) * sw, sy0 + sh - (v - vmin) / (vmax - vmin) * sh) for i, v in enumerate(vals)]
    for yv in (20, 30, 40):
        y = sy0 + sh - (yv - vmin) / (vmax - vmin) * sh
        d.line((sx0, y, sx0 + sw, y), fill=(226, 224, 218))
        d.text((sx0 - 8, y), f"{yv}°", font=font(13), fill=MUTED, anchor="rm")
    d.line(pts, fill=(170, 170, 176), width=2)
    k = int(frame_pos * (len(vals) - 1))
    if k > 0:
        d.line(pts[: k + 1], fill=(208, 44, 69), width=3)
    cx, cy = pts[k]
    d.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=(208, 44, 69))
    d.text((sx0, sy0 - 26), "Hottest cell in Czechia, hourly", font=font(16, True), fill=INK)
    for day, lbl in series["ticks"]:
        x = sx0 + day * sw
        d.text((x, sy0 + sh + 6), lbl, font=font(13), fill=MUTED, anchor="ma")

    d.text((W // 2, H - 30), CREDIT, font=font(14), fill=MUTED, anchor="ma")
    tmp = out.with_suffix('.tmp.png')
    canvas.save(tmp)
    os.replace(tmp, out)


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="UTC time like 2026-06-29T15 -> render single still")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    work = OUT / "work"
    frames_dir = OUT / "frames"
    work.mkdir(exist_ok=True)
    frames_dir.mkdir(exist_ok=True)

    geom = fetch_boundary()
    grid = build_dem(geom)
    times, lat, lon, t2m, zs = fetch_era5()
    ds = Downscaler(grid, lat, lon, zs)

    # hourly national max series (on downscaled grid, cheap: subsample)
    series_path = work / "series.json"
    if series_path.exists():
        series = json.loads(series_path.read_text())
    else:
        mx = [float(np.nanmax(np.where(ds.inside, ds(t2m[i]), np.nan))) for i in range(len(times))]
        t0 = times[0]
        ticks = []
        for i, t in enumerate(times):
            loc = (t + np.timedelta64(2, "h")).astype("datetime64[h]").astype(dt.datetime)
            if loc.hour == 0:
                ticks.append((i / (len(times) - 1), loc.strftime("%d %b")))
        series = {"max": mx, "ticks": ticks}
        series_path.write_text(json.dumps(series))
        print("series peak", max(mx), "at", str(times[int(np.argmax(mx))]))

    v = open_viewer(grid["path"])
    try:
        time.sleep(1.0)
        cities, mask = locate_cities(v, grid, work)

        n_hours = len(times)
        if args.preview:
            i = int(np.argmin(np.abs(times - np.datetime64(args.preview))))
            plan = [(i, 0.0)]
        else:
            plan = [(i, s / SUBFRAMES) for i in range(n_hours - 1) for s in range(SUBFRAMES)] + [(n_hours - 1, 0.0)]
        if args.max_frames:
            plan = plan[: args.max_frames]
        total = len(plan)
        t_start = time.time()
        rendered_count = 0
        for fi, (i, frac) in enumerate(plan):
            out_png = frames_dir / f"f_{fi:05d}.png"
            if args.preview:
                out_png = OUT / "preview.png"
            elif out_png.exists():
                continue
            rendered_count += 1
            if rendered_count % 40 == 0:  # the viewer accumulates overlay textures; recycle it
                v.close()
                v = open_viewer(grid["path"])
                time.sleep(1.0)
            field = t2m[i] if frac == 0 else (1 - frac) * t2m[i] + frac * t2m[i + 1]
            tg = ds(field)
            raw = work / "raw.png"
            snap_overlay(v, overlay_rgba(tg, grid["inside"]), raw, work / "ov.png")
            when = times[i] + np.timedelta64(int(frac * 60), "m")
            pos = (i + frac) / (n_hours - 1)
            compose(raw, out_png, when_utc=when, t_grid=tg, ds=ds, mask=mask, cities=cities, series=series, frame_pos=pos)
            if fi % 10 == 0:
                el = time.time() - t_start
                print(f"frame {fi + 1}/{total}  {el / (fi + 1):.1f}s/frame", flush=True)
    finally:
        v.close()

    if args.preview:
        print("preview ->", OUT / "preview.png")
        return 0
    mp4 = OUT / "czechia_heatwave_2026_era5.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames_dir / "f_%05d.png"),
                    "-vf", "tpad=stop_mode=clone:stop_duration=2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", "-preset", "slow", "-movflags", "+faststart", str(mp4)], check=True)
    print("video ->", mp4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
