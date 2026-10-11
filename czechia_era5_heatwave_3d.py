#!/usr/bin/env python3
"""Czechia, June 2026 record heatwave — ERA5 2 m temperature as 3D columns (forge3d).

Companion to czechia_era5_heatwave.py (same data + downscaling). Instead of
draping colour over real relief, this version uses the "value-as-height"
approach from forge3d's population spike examples: the temperature field itself
becomes the terrain. Czechia is tiled into 6 km cells; each cell is extruded
into a flat-topped column whose height AND colour encode the cell's mean 2 m
temperature. Every frame loads a new heightfield into the forge3d viewer.

Usage:
    python czechia_era5_heatwave_3d.py                        # full MP4
    python czechia_era5_heatwave_3d.py --preview 2026-06-28T14
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import czechia_era5_heatwave as base
from czechia_era5_heatwave import INK, MUTED, T_MAX, T_MIN, T_STOPS, font

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out3d"
WIDTH, HEIGHT = base.WIDTH, base.HEIGHT
FPS, SUBFRAMES = base.FPS, base.SUBFRAMES

# geometry
AGG = 2            # 250 m DEM grid -> 500 m heightfield pixels
CELL = 12          # heightfield pixels per column (12 x 500 m = 6 km)
GAP = 2            # pixels of gap between columns
COVER_MIN = 0.35   # min share of a cell inside Czechia to get a column
T_BASE = 8.0       # °C at which a column has zero height
M_PER_DEG = 300.0  # metres of column per °C (then exaggerated by zscale)
PIX_UNITS = 500.0 / base.DEM_UNIT_SCALE  # heightfield pixel size in viewer world units

TITLE = "Czechia's record heatwave, in 3D"
SUBTITLE = "Column height and colour = ERA5 2 m air temperature (6 km cells)"

TERRAIN = {
    "cmd": "set_terrain",
    "phi": 90.0, "theta": 58.0, "radius": 330000.0, "fov": 33.0, "zscale": 0.15,
    "sun_azimuth": 300.0, "sun_elevation": 40.0, "sun_intensity": 1.0,
    "ambient": 0.6, "shadow": 0.5, "background": [1.0, 1.0, 1.0],
}
PBR = {
    "cmd": "set_terrain_pbr", "enabled": True, "exposure": 2.0,
    "shadow_technique": "pcss", "shadow_map_res": 4096, "msaa": 4, "normal_strength": 1.0,
    "height_ao": {"enabled": True, "directions": 6, "steps": 14, "max_distance": 120.0,
                  "strength": 0.45, "resolution_scale": 0.5},
}
BG = (250, 249, 246)


# --------------------------------------------------------------------------- geometry
class Columns:
    def __init__(self, grid: dict):
        ins = grid["inside"]
        self.h, self.w = ins.shape
        self.f = AGG * CELL
        self.hc, self.wc = self.h // self.f, self.w // self.f
        self.h2, self.w2 = self.h // AGG, self.w // AGG
        cov = ins[: self.hc * self.f, : self.wc * self.f].reshape(self.hc, self.f, self.wc, self.f).mean((1, 3))
        self.cell_ok = cov >= COVER_MIN
        self.inside = ins
        gm = np.zeros((CELL, CELL), bool)
        gm[:GAP, :] = True
        gm[:, :GAP] = True
        self.gap = np.tile(gm, (self.hc, self.wc))
        self.transform = grid["transform"]

    def cell_means(self, t250: np.ndarray) -> np.ndarray:
        t = np.where(self.inside, t250, np.nan)[: self.hc * self.f, : self.wc * self.f]
        with np.errstate(invalid="ignore"):
            tc = np.nanmean(t.reshape(self.hc, self.f, self.wc, self.f), axis=(1, 3))
        return np.where(self.cell_ok, tc, np.nan)

    def _up(self, a: np.ndarray, fill) -> np.ndarray:
        out = np.full((self.h2, self.w2), fill, dtype=np.float32)
        u = np.repeat(np.repeat(a, CELL, 0), CELL, 1)
        out[: u.shape[0], : u.shape[1]] = u
        return out

    def heightfield(self, tc: np.ndarray) -> np.ndarray:
        top = np.maximum(tc - T_BASE, 0.3) * M_PER_DEG
        H = self._up(top, np.nan)
        g = np.zeros((self.h2, self.w2), bool)
        g[: self.gap.shape[0], : self.gap.shape[1]] = self.gap
        H[g & np.isfinite(H)] = 0.0
        H[self._ring(np.isfinite(H))] = 0.0  # floor ring so border columns get full walls
        return H

    @staticmethod
    def _ring(ok: np.ndarray) -> np.ndarray:
        from scipy.ndimage import binary_dilation
        return binary_dilation(ok, iterations=GAP + 1) & ~ok

    def overlay(self, tc: np.ndarray, lut_in: np.ndarray) -> np.ndarray:
        from scipy.ndimage import distance_transform_edt
        T = self._up(tc, np.nan)
        ring = self._ring(np.isfinite(T))
        idx_r, idx_c = distance_transform_edt(~np.isfinite(T), return_distances=False, return_indices=True)
        T = np.where(ring, T[idx_r, idx_c], T)  # ring takes the neighbouring column's colour
        ok = np.isfinite(T)
        idx = np.clip(((np.nan_to_num(T) - T_MIN) / (T_MAX - T_MIN) * 1023).round().astype(int), 0, 1023)
        rgba = np.zeros((self.h2, self.w2, 4), np.uint8)
        rgba[..., :3] = lut_in[idx].astype(np.uint8)
        rgba[..., 3] = np.where(ok, 255, 0)
        return rgba

    def cell_of(self, lon: float, lat: float) -> tuple[int, int]:
        import rasterio
        from pyproj import Transformer
        x, y = Transformer.from_crs(4326, 3035, always_xy=True).transform(lon, lat)
        r, c = rasterio.transform.rowcol(self.transform, x, y)
        return int(r) // self.f, int(c) // self.f

    def world_xz(self, cr: int, cc: int) -> tuple[float, float]:
        """Cell-top centre in heightfield pixel coords (col, row)."""
        return (cc * CELL + (CELL + GAP) / 2.0, cr * CELL + (CELL + GAP) / 2.0)


def write_dem(H: np.ndarray, path: Path) -> Path:
    import rasterio
    with rasterio.open(path, "w", driver="GTiff", width=H.shape[1], height=H.shape[0], count=1,
                       dtype="float32", nodata=np.nan,
                       transform=rasterio.Affine(PIX_UNITS, 0, 0, 0, -PIX_UNITS, 0)) as d:
        d.write(H, 1)
    return path


# --------------------------------------------------------------------------- viewer
def open_viewer():
    import forge3d as f3d
    return f3d.open_viewer_async(width=WIDTH, height=HEIGHT, timeout=180)


def render(v, H: np.ndarray, rgba: np.ndarray, out_png: Path, work: Path, size=(WIDTH, HEIGHT)) -> None:
    dem = write_dem(H, work / "frame_dem.tif")
    v.load_terrain(str(dem.resolve()))
    time.sleep(0.5)
    v.send_ipc(TERRAIN)
    v.send_ipc(PBR)
    Image.fromarray(rgba, "RGBA").save(work / "frame_ov.png")
    v.load_overlay("t2m", str((work / "frame_ov.png").resolve()), extent=(0.0, 0.0, 1.0, 1.0),
                   opacity=1.0, preserve_colors=True)
    v.send_ipc({"cmd": "set_overlays_enabled", "enabled": True})
    v.send_ipc({"cmd": "set_overlay_solid", "solid": True})
    v.snapshot(str(out_png.resolve()), width=size[0], height=size[1])


def subject_mask(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("RGB"), np.int16)
    return np.abs(a - 250).max(-1) > 16


def calibrate_ramp(v, cols: Columns, work: Path, iters: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Same idea as the drape version, measured on flat column tops: render every column at
    one fixed height in one ramp colour, read the median top colour, correct the input."""
    cal = work / "ramp3d.json"
    if cal.exists():
        c = json.loads(cal.read_text())
        stops_in, rendered = np.array(c["stops_in"], np.float32), np.array(c["rendered"], np.float32)
    else:
        tc = np.where(cols.cell_ok, 30.0, np.nan)
        H = cols.heightfield(tc)
        targets = np.array([base._hex(c) for _, c in T_STOPS], np.float32)
        stops_in, rendered = targets.copy(), targets.copy()
        small = (WIDTH // 3, HEIGHT // 3)
        top_mask = None
        for it in range(iters):
            for k in range(len(T_STOPS)):
                lut = np.tile(np.clip(stops_in[k], 0, 255), (1024, 1))
                render(v, H, cols.overlay(tc, lut), work / "cal.png", work, size=small)
                img = np.asarray(Image.open(work / "cal.png").convert("RGB"), np.float32)
                if top_mask is None:
                    # column tops: brightest third of the subject (walls are shaded darker)
                    m = subject_mask(Image.open(work / "cal.png"))
                    lum = img.mean(-1)
                    top_mask = m & (lum >= np.percentile(lum[m], 60))
                got = np.median(img[top_mask], 0)
                rendered[k] = got
                stops_in[k] = np.clip(stops_in[k] + 1.1 * (targets[k] - got), 0, 255)
                if it == iters - 1:
                    print(f"  stop {T_STOPS[k][0]:>2}°C target {targets[k].astype(int)} rendered {got.astype(int)}", flush=True)
        cal.write_text(json.dumps({"stops_in": stops_in.tolist(), "rendered": rendered.tolist()}))
    lut_in = np.stack([np.interp(base._LUT_T, [t for t, _ in T_STOPS], stops_in[:, c]) for c in range(3)], -1)
    lut_view = np.stack([np.interp(base._LUT_T, [t for t, _ in T_STOPS], rendered[:, c]) for c in range(3)], -1)
    return lut_in, lut_view


def fit_camera(v, cols: Columns, work: Path) -> np.ndarray:
    """Recover the viewer's 3x4 projection (DLT) from probe renders: each city's column top
    is lit up white at several heights and located on screen, and the
    (col, row, height) -> (x, y) correspondences give P. Labels can then follow the
    column tops as they rise and fall."""
    cache = work / "camera_P.json"
    if cache.exists():
        return np.array(json.loads(cache.read_text()))
    pts3, pts2 = [], []
    for h_deg in (12.0, 24.0, 36.0):
        # all columns at the probe height, all black; then one city's top-face interior white.
        # The difference against the all-black baseline isolates that top face on screen.
        tc = np.where(cols.cell_ok, h_deg + T_BASE, np.nan)
        H = cols.heightfield(tc)
        black = cols.overlay(tc, np.zeros((1024, 3), np.float32))
        render(v, H, black, work / "probe_base.png", work)
        base_img = np.asarray(Image.open(work / "probe_base.png").convert("RGB"), np.float32)
        for name, lon, lat in base.CITIES:
            cr, cc = cols.cell_of(lon, lat)
            rgba = black.copy()
            rgba[cr * CELL + GAP + 1:(cr + 1) * CELL - 1, cc * CELL + GAP + 1:(cc + 1) * CELL - 1, :3] = 255
            render(v, H, rgba, work / "probe.png", work)
            diff = np.abs(np.asarray(Image.open(work / "probe.png").convert("RGB"), np.float32) - base_img).max(-1)
            ys, xs = np.nonzero(diff > 60)
            if len(xs) < 5:
                print(f"  camera probe {name} {h_deg}: not visible", flush=True)
                continue
            x_px, z_px = cols.world_xz(cr, cc)
            pts3.append((x_px, z_px, h_deg * M_PER_DEG))
            pts2.append((float(xs.mean()), float(ys.mean())))
        print(f"  camera probes at {h_deg:.0f} °C done", flush=True)
    # normalised DLT
    X3, X2 = np.asarray(pts3, np.float64), np.asarray(pts2, np.float64)
    m3, s3 = X3.mean(0), X3.std(0) + 1e-9
    m2, s2 = X2.mean(0), X2.std(0) + 1e-9
    N3, N2 = (X3 - m3) / s3, (X2 - m2) / s2
    A = []
    for (X, Y, Z), (u, w) in zip(N3, N2):
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -w * X, -w * Y, -w * Z, -w])
    _, _, Vt = np.linalg.svd(np.asarray(A, np.float64))
    Pn = Vt[-1].reshape(3, 4)
    T3 = np.eye(4); T3[:3, :3] = np.diag(1 / s3); T3[:3, 3] = -m3 / s3
    T2i = np.array([[s2[0], 0, m2[0]], [0, s2[1], m2[1]], [0, 0, 1]])
    P = T2i @ Pn @ T3
    err = [np.hypot(*(project(P, *p3) - np.array(p2))) for p3, p2 in zip(pts3, pts2)]
    print(f"  camera fit: {len(pts3)} points, median reprojection error {np.median(err):.1f}px", flush=True)
    cache.write_text(json.dumps(P.tolist()))
    return P


def project(P: np.ndarray, x: float, z: float, h: float) -> np.ndarray:
    p = P @ np.array([x, z, h, 1.0])
    return p[:2] / p[2]


# --------------------------------------------------------------------------- compose
def compose(raw: Path, out: Path, *, box: tuple, when_utc, tc: np.ndarray, cols: Columns, P: np.ndarray,
            lut_view: np.ndarray, series: dict, frame_pos: float) -> None:
    W, H = WIDTH, HEIGHT
    src_full = Image.open(raw).convert("RGB")
    mask = subject_mask(src_full)
    canvas = Image.new("RGB", (W, H), BG)
    src = src_full.crop(box)
    alpha = Image.fromarray((mask * 255).astype(np.uint8)).crop(box).filter(ImageFilter.GaussianBlur(0.7))
    scale = min((W * 0.86) / src.width, (H * 0.70) / src.height)
    cw, ch = int(src.width * scale), int(src.height * scale)
    src, alpha = src.resize((cw, ch), Image.Resampling.LANCZOS), alpha.resize((cw, ch), Image.Resampling.LANCZOS)
    ox, oy = (W - cw) // 2, 190
    sh = Image.new("L", (W, H), 0)
    sh.paste(alpha, (ox + 8, oy + 16))
    sh = sh.filter(ImageFilter.GaussianBlur(16)).point(lambda v: int(v * 0.25))
    canvas = Image.composite(Image.new("RGB", (W, H), (60, 55, 50)), canvas, sh)
    canvas.paste(src, (ox, oy), alpha)
    d = ImageDraw.Draw(canvas)

    def colour(t):
        i = int(np.clip((t - T_MIN) / (T_MAX - T_MIN) * 1023, 0, 1023))
        return tuple(int(c) for c in lut_view[i])

    # city labels on column tops
    fl = font(19, True)
    for name, lon, lat in base.CITIES:
        cr, cc = cols.cell_of(lon, lat)
        t = tc[cr, cc]
        if not np.isfinite(t):
            continue
        x_px, z_px = cols.world_xz(cr, cc)
        sx, sy = project(P, x_px, z_px, max(t - T_BASE, 0.3) * M_PER_DEG)
        px, py = ox + (sx - box[0]) * scale, oy + (sy - box[1]) * scale
        d.line((px, py, px, py - 26), fill=INK, width=2)
        d.ellipse((px - 3, py - 3, px + 3, py + 3), fill=INK)
        d.text((px + 6, py - 44), f"{name}  {t:.0f}°", font=fl, fill=INK, stroke_width=3, stroke_fill=(255, 255, 255))

    d.text((70, 52), TITLE, font=font(46, True), fill=INK)
    d.text((72, 112), SUBTITLE, font=font(22), fill=MUTED)
    local = (when_utc + np.timedelta64(2, "h")).astype("datetime64[m]").astype(dt.datetime)
    d.text((W - 70, 50), local.strftime("%a %d %b %Y"), font=font(34, True), fill=INK, anchor="ra")
    d.text((W - 70, 94), local.strftime("%H:%M") + " CEST", font=font(30), fill=MUTED, anchor="ra")
    tmax = float(np.nanmax(tc))
    d.text((W - 70, 146), f"Hottest 6 km cell  {tmax:4.1f} °C", font=font(22, True),
           fill=colour(tmax) if tmax > 30 else INK, anchor="ra")

    # legend (colours as measured on the rendered column tops)
    lx, ly, lw, lh = 70, H - 118, 520, 16
    bar = np.stack([lut_view[np.clip(((np.linspace(T_MIN, T_MAX, lw) - T_MIN) / (T_MAX - T_MIN) * 1023).astype(int), 0, 1023)]] * lh).astype(np.uint8)
    canvas.paste(Image.fromarray(bar), (lx, ly))
    d.rectangle((lx, ly, lx + lw, ly + lh), outline=(200, 200, 200))
    d.text((lx, ly - 30), "2 m temperature (°C)  ·  height ∝ °C above 8", font=font(18, True), fill=INK)
    for t in range(10, 44, 5):
        x = lx + (t - T_MIN) / (T_MAX - T_MIN) * lw
        d.line((x, ly + lh, x, ly + lh + 5), fill=MUTED)
        d.text((x, ly + lh + 8), str(t), font=font(15), fill=MUTED, anchor="ma")

    sx0, sy0, sw, shh = W - 70 - 640, H - 150, 640, 70
    vals, vmin, vmax = series["max"], 14.0, 42.0
    pts = [(sx0 + i / (len(vals) - 1) * sw, sy0 + shh - (v - vmin) / (vmax - vmin) * shh) for i, v in enumerate(vals)]
    for yv in (20, 30, 40):
        y = sy0 + shh - (yv - vmin) / (vmax - vmin) * shh
        d.line((sx0, y, sx0 + sw, y), fill=(226, 224, 218))
        d.text((sx0 - 8, y), f"{yv}°", font=font(13), fill=MUTED, anchor="rm")
    d.line(pts, fill=(170, 170, 176), width=2)
    k = int(frame_pos * (len(vals) - 1))
    if k > 0:
        d.line(pts[: k + 1], fill=(208, 44, 69), width=3)
    cx, cy = pts[k]
    d.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=(208, 44, 69))
    d.text((sx0, sy0 - 26), "Hottest 6 km cell in Czechia, hourly", font=font(16, True), fill=INK)
    for day, lbl in series["ticks"]:
        d.text((sx0 + day * sw, sy0 + shh + 6), lbl, font=font(13), fill=MUTED, anchor="ma")
    d.text((W // 2, H - 30), base.CREDIT, font=font(14), fill=MUTED, anchor="ma")
    tmp = out.with_suffix(".tmp.png")
    canvas.save(tmp)
    os.replace(tmp, out)


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="UTC time like 2026-06-28T14 -> single still")
    args = ap.parse_args()
    work, frames = OUT / "work", OUT / "frames"
    work.mkdir(parents=True, exist_ok=True)
    frames.mkdir(parents=True, exist_ok=True)

    grid = base.build_dem(base.fetch_boundary())
    times, lat, lon, t2m, zs = base.fetch_era5()
    ds = base.Downscaler(grid, lat, lon, zs)
    cols = Columns(grid)
    # lock the orbit target (otherwise the viewer re-centres on each new heightfield's mean height)
    TERRAIN["target"] = [cols.w2 * PIX_UNITS / 2.0, 900.0, cols.h2 * PIX_UNITS / 2.0]

    series_path = work / "series.json"
    if series_path.exists():
        series = json.loads(series_path.read_text())
    else:
        mx = [float(np.nanmax(cols.cell_means(ds(t2m[i])))) for i in range(len(times))]
        ticks = [(i / (len(times) - 1), (t + np.timedelta64(2, "h")).astype("datetime64[h]").astype(dt.datetime).strftime("%d %b"))
                 for i, t in enumerate(times) if (t + np.timedelta64(2, "h")).astype("datetime64[h]").astype(dt.datetime).hour == 0]
        series = {"max": mx, "ticks": ticks}
        series_path.write_text(json.dumps(series))

    v = open_viewer()
    try:
        time.sleep(1.0)
        lut_in, lut_view = calibrate_ramp(v, cols, work)
        P = fit_camera(v, cols, work)
        # fixed crop box from the tallest frame (+ margin for everything shorter)
        box_path = work / "box.json"
        if not box_path.exists():
            ipk = int(np.argmax(series["max"]))
            tc = cols.cell_means(ds(t2m[ipk]))
            render(v, cols.heightfield(tc), cols.overlay(tc, lut_in), work / "ref.png", work)
            ys, xs = np.nonzero(subject_mask(Image.open(work / "ref.png")))
            pad = 24
            box_path.write_text(json.dumps([max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad - 30),
                                            min(WIDTH, int(xs.max()) + pad), min(HEIGHT, int(ys.max()) + pad)]))
        box = tuple(json.loads(box_path.read_text()))

        n = len(times)
        if args.preview:
            plan = [(int(np.argmin(np.abs(times - np.datetime64(args.preview)))), 0.0)]
        else:
            plan = [(i, s / SUBFRAMES) for i in range(n - 1) for s in range(SUBFRAMES)] + [(n - 1, 0.0)]
        t_start, done = time.time(), 0
        for fi, (i, frac) in enumerate(plan):
            out_png = OUT / "preview.png" if args.preview else frames / f"f_{fi:05d}.png"
            if not args.preview and out_png.exists():
                continue
            done += 1
            if done % 40 == 0:  # recycle viewer (overlay/terrain textures accumulate)
                v.close()
                v = open_viewer()
                time.sleep(1.0)
            field = t2m[i] if frac == 0 else (1 - frac) * t2m[i] + frac * t2m[i + 1]
            tc = cols.cell_means(ds(field))
            render(v, cols.heightfield(tc), cols.overlay(tc, lut_in), work / "raw.png", work)
            compose(work / "raw.png", out_png, box=box, when_utc=times[i] + np.timedelta64(int(frac * 60), "m"),
                    tc=tc, cols=cols, P=P, lut_view=lut_view, series=series, frame_pos=(i + frac) / (n - 1))
            if done % 10 == 1:
                print(f"frame {fi + 1}/{len(plan)}  {(time.time() - t_start) / done:.1f}s/frame", flush=True)
    finally:
        v.close()

    if args.preview:
        print("preview ->", OUT / "preview.png")
        return 0
    mp4 = OUT / "czechia_heatwave_2026_era5_3d_columns.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames / "f_%05d.png"),
                    "-vf", "tpad=stop_mode=clone:stop_duration=2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", "-preset", "slow", "-movflags", "+faststart", str(mp4)], check=True)
    print("video ->", mp4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
