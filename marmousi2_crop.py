"""Prepare cropped Marmousi2 prestack data for probabilistic inversion.

This script reads VP, VS, and density SEG-Y models, crops a target area,
converts the depth-domain elastic models to the time domain, generates
three-angle synthetic prestack seismic data with the Aki--Richards
approximation, and saves MATLAB-compatible .mat files for later inversion
experiments.
"""

import gzip
import shutil
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import fftconvolve


# -----------------------------------------------------------------------------
# Global configuration
# -----------------------------------------------------------------------------
# This script converts the original Marmousi2 VP/VS/density SEG-Y models into
# compact 4-D arrays used by the inversion code. The saved arrays follow the
# convention:
#   Stack: [L, T, A, N]
#   Mod:   [L, T, 3, N]
# where L is the cropped horizontal sample number, T is the 2-D section index
# dimension kept for compatibility, A is the number of incident angles, and N is
# the number of time samples.

# Local working directory containing the original SEG-Y or SEG-Y.gz files.
# Change this path according to your own project directory.
WORKDIR = Path(r"D:\pycharm\project\marmousi")

# Output directory for cropped models, synthetic stacks, auxiliary data, and
# quick-look figures.
OUTDIR = WORKDIR / "marmousi2_cropped"
WORKDIR.mkdir(parents=True, exist_ok=True)
OUTDIR.mkdir(parents=True, exist_ok=True)

# Original AGL Marmousi2 model files. The script accepts either compressed
# .segy.gz files or already uncompressed .segy files in WORKDIR.
FILES = {
    "vp": "vp_marmousi-ii.segy.gz",
    "vs": "vs_marmousi-ii.segy.gz",
    "rho": "density_marmousi-ii.segy.gz",
}

# Crop area in the full Marmousi2 model, in meters.
# The selected window is used as the synthetic test region.
X_MIN_M = 11_300.0
X_MAX_M = 16_900.0
Z_MIN_M = 500.0
Z_MAX_M = 2_050.0

# Downsampling factors. Increasing these values reduces data size but also
# decreases the spatial resolution of the cropped model.
X_DECIM = 8
Z_DECIM = 2

# Original spatial sampling intervals of the Marmousi2 model, in meters.
DX0 = 1.25
DZ0 = 1.25

# Synthetic seismic-data parameters. DT is the time sampling interval in seconds,
# FDOM is the dominant frequency of the Ricker wavelet, and ANGLES_DEG defines
# the prestack incident angles.
DT = 0.001
FDOM = 35.0
ANGLES_DEG = [5.0, 15.0, 25.0]
WAVELET_LEN_S = 0.128
NORMALIZE_STACK = True
STACK_PCT = 99.5


def get_local_segy_path(name: str, filename: str) -> Path:
    """Return a local SEGY file path, unzipping the .gz file if needed."""
    gz_path = WORKDIR / filename
    segy_path = WORKDIR / filename.replace(".gz", "")

    if segy_path.exists():
        print(f"[LOCAL] {name}: use existing {segy_path}")
        return segy_path

    if not gz_path.exists():
        raise FileNotFoundError(
            f"Cannot find {name} file. Expected either:\n"
            f"  {gz_path}\n"
            f"or already unzipped:\n"
            f"  {segy_path}"
        )

    print(f"[UNZIP] {gz_path}")
    with gzip.open(gz_path, "rb") as f_in, open(segy_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    return segy_path


def ibm2float(u: np.ndarray) -> np.ndarray:
    """Convert IBM 32-bit floating-point words to IEEE float32."""
    u = np.asarray(u, dtype=np.uint32)
    sign = np.where((u >> 31) & 1, -1.0, 1.0)
    exp = ((u >> 24) & 0x7F).astype(np.int32)
    frac = (u & 0x00FFFFFF).astype(np.float64)
    out = sign * (frac / float(0x01000000)) * np.power(16.0, exp - 64)
    out[u == 0] = 0.0
    return out.astype(np.float32)


def _read_u2_be(buf: bytes, offset: int) -> int:
    return int.from_bytes(buf[offset:offset + 2], byteorder="big", signed=False)


def read_marmousi2_segy(path: Path, chunk_traces: int = 512) -> np.ndarray:
    """
    Read an AGL Marmousi2 model SEGY file.

    The expected model shape is [x, z] = [13601, 2801]. Both IBM float
    (format code 1) and IEEE float (format code 5) are supported.
    """
    path = Path(path)
    print(f"[READ] {path}")

    with open(path, "rb") as f:
        f.seek(3200)
        bin_hdr = f.read(400)

    ns = _read_u2_be(bin_hdr, 20)
    fmt = _read_u2_be(bin_hdr, 24)

    if ns <= 0:
        raise RuntimeError(f"Invalid ns={ns} in SEGY binary header: {path}")
    if fmt not in (1, 5):
        raise RuntimeError(
            f"Unsupported SEGY sample format code={fmt}. "
            f"This script supports IBM float(1) and IEEE float(5)."
        )

    bytes_per_sample = 4
    trace_bytes = 240 + ns * bytes_per_sample
    file_bytes = path.stat().st_size
    data_bytes = file_bytes - 3600

    if data_bytes <= 0 or data_bytes % trace_bytes != 0:
        raise RuntimeError(
            f"Unexpected SEGY size. file_bytes={file_bytes}, ns={ns}, "
            f"trace_bytes={trace_bytes}, data_bytes={data_bytes}."
        )

    ntraces = data_bytes // trace_bytes
    print(f"[SEGY] ns={ns}, ntraces={ntraces}, sample_format={fmt}")

    if fmt == 1:
        dtype = np.dtype([("hdr", "u1", 240), ("data", ">u4", ns)])
        mm = np.memmap(path, dtype=dtype, mode="r", offset=3600, shape=(ntraces,))
        arr = np.empty((ntraces, ns), dtype=np.float32)
        for s in range(0, ntraces, chunk_traces):
            e = min(s + chunk_traces, ntraces)
            arr[s:e] = ibm2float(mm[s:e]["data"])
    else:
        dtype = np.dtype([("hdr", "u1", 240), ("data", ">f4", ns)])
        mm = np.memmap(path, dtype=dtype, mode="r", offset=3600, shape=(ntraces,))
        arr = np.asarray(mm["data"], dtype=np.float32)

    if arr.shape[0] < arr.shape[1]:
        print(f"[WARN] read shape={arr.shape}, auto transpose to [x,z].")
        arr = arr.T

    print(f"[READ DONE] shape={arr.shape}, min={float(arr.min()):.6g}, max={float(arr.max()):.6g}")
    return arr.astype(np.float32, copy=False)


def crop_and_downsample(arr: np.ndarray) -> np.ndarray:
    """Crop the target physical window and apply spatial downsampling."""
    # Convert physical coordinates to array indices using the original sampling.
    ix0 = int(round(X_MIN_M / DX0))
    ix1 = int(round(X_MAX_M / DX0))
    iz0 = int(round(Z_MIN_M / DZ0))
    iz1 = int(round(Z_MAX_M / DZ0))

    # Clip the indices to avoid out-of-range slicing if the input model has a
    # slightly different extent.
    ix0 = max(0, min(ix0, arr.shape[0] - 1))
    ix1 = max(ix0 + 1, min(ix1, arr.shape[0]))
    iz0 = max(0, min(iz0, arr.shape[1] - 1))
    iz1 = max(iz0 + 1, min(iz1, arr.shape[1]))

    # Slice the cropped area and decimate it to a lighter computational scale.
    out = arr[ix0:ix1:X_DECIM, iz0:iz1:Z_DECIM].astype(np.float32)
    print(
        f"[CROP] x=[{X_MIN_M:g},{X_MAX_M:g}]m -> ix=[{ix0},{ix1}), "
        f"z=[{Z_MIN_M:g},{Z_MAX_M:g}]m -> iz=[{iz0},{iz1}), shape={out.shape}"
    )
    return out


def ricker(fdom: float, dt: float, length_s: float = 0.128) -> np.ndarray:
    """Generate an energy-normalized Ricker wavelet."""
    n = int(round(length_s / dt))
    if n % 2 == 0:
        n += 1
    t = (np.arange(n, dtype=np.float32) - n // 2) * dt
    x = np.pi * float(fdom) * t
    w = (1.0 - 2.0 * x**2) * np.exp(-x**2)
    w = w / (np.sqrt(np.sum(w * w)) + 1e-12)
    return w.astype(np.float32)


def depth_to_time(vp_z, vs_z, rho_z, dz: float, dt: float):
    """Convert depth-domain models [nx, nz] to time-domain models [nx, nt].

    Two-way traveltime is computed from the P-wave velocity model and then used
    to resample VP, VS, and density onto a uniform time axis.
    """
    nx, _ = vp_z.shape
    vp_safe = np.clip(vp_z.astype(np.float32), 1000.0, None)

    # Compute cumulative two-way time. The minimum final time across traces is
    # used so that every trace has valid samples over the full output time axis.
    twt = 2.0 * np.cumsum(np.ones_like(vp_safe, dtype=np.float32) * dz / vp_safe, axis=1)
    tmax = float(np.min(twt[:, -1]))
    t_axis = np.arange(0.0, tmax, dt, dtype=np.float32)
    nt = len(t_axis)

    if nt < 20:
        raise RuntimeError(f"Too few time samples after depth_to_time: nt={nt}. Check crop/depth units.")

    vp_t = np.empty((nx, nt), dtype=np.float32)
    vs_t = np.empty((nx, nt), dtype=np.float32)
    rho_t = np.empty((nx, nt), dtype=np.float32)

    # Interpolate each lateral trace from depth-indexed samples to time samples.
    for ix in range(nx):
        vp_t[ix] = np.interp(t_axis, twt[ix], vp_z[ix]).astype(np.float32)
        vs_t[ix] = np.interp(t_axis, twt[ix], vs_z[ix]).astype(np.float32)
        rho_t[ix] = np.interp(t_axis, twt[ix], rho_z[ix]).astype(np.float32)

    return vp_t, vs_t, rho_t, t_axis


def aki_richards_reflectivity(vp, vs, rho, angle_deg: float) -> np.ndarray:
    """
    Compute PP reflectivity using the Aki-Richards approximation.

    Input and output shapes are [nx, nt].
    """
    theta = np.deg2rad(float(angle_deg))

    # Adjacent time samples define the upper and lower layer properties.
    vp1, vp2 = vp[:, :-1], vp[:, 1:]
    vs1, vs2 = vs[:, :-1], vs[:, 1:]
    r1, r2 = rho[:, :-1], rho[:, 1:]

    vpm = 0.5 * (vp1 + vp2)
    vsm = 0.5 * (vs1 + vs2)
    rm = 0.5 * (r1 + r2)

    d_vp = (vp2 - vp1) / np.clip(vpm, 1e-6, None)
    d_vs = (vs2 - vs1) / np.clip(vsm, 1e-6, None)
    d_rho = (r2 - r1) / np.clip(rm, 1e-6, None)

    sin2 = np.sin(theta) ** 2
    tan2 = np.tan(theta) ** 2

    # Aki--Richards three-term approximation for PP reflectivity.
    a = 0.5 * (d_vp + d_rho)
    b = 0.5 * d_vp - 2.0 * (vsm / np.clip(vpm, 1e-6, None)) ** 2 * (2.0 * d_vs + d_rho)
    c = 0.5 * d_vp

    rc = a + b * sin2 + c * (tan2 - sin2)
    rc = np.pad(rc, ((0, 0), (1, 0)), mode="constant")
    return np.nan_to_num(rc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def safe_units(vp, vs, rho):
    """Convert VP/VS from km/s to m/s if needed and keep density in g/cc."""
    vp = vp.astype(np.float32, copy=False)
    vs = vs.astype(np.float32, copy=False)
    rho = rho.astype(np.float32, copy=False)

    if np.nanmax(vp) < 20.0:
        print("[UNIT] VP appears to be km/s -> convert to m/s")
        vp = vp * 1000.0
    if np.nanmax(vs) < 20.0:
        print("[UNIT] VS appears to be km/s -> convert to m/s")
        vs = vs * 1000.0

    return vp.astype(np.float32), vs.astype(np.float32), rho.astype(np.float32)


def make_preview(vp_c, vs_c, rho_c, stack, out_png: Path):
    """Save a preview of the cropped depth models and synthetic angle stacks."""
    import matplotlib.pyplot as plt

    dx_eff = DX0 * X_DECIM
    dz_eff = DZ0 * Z_DECIM

    # Build physical plotting extents for depth-domain and time-domain panels.
    nx, nz = vp_c.shape
    nt = stack.shape[2]

    x0 = X_MIN_M
    x1 = X_MIN_M + (nx - 1) * dx_eff
    z0 = Z_MIN_M
    z1 = Z_MIN_M + (nz - 1) * dz_eff
    t0_ms = 0.0
    t1_ms = (nt - 1) * DT * 1000.0

    extent_depth = [x0, x1, z1, z0]
    extent_time = [x0, x1, t1_ms, t0_ms]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    im0 = axes[0, 0].imshow(vp_c.T, aspect="auto", cmap="viridis", origin="upper", extent=extent_depth)
    axes[0, 0].set_title(r"$V_p$ depth-domain model")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(vs_c.T, aspect="auto", cmap="viridis", origin="upper", extent=extent_depth)
    axes[0, 1].set_title(r"$V_s$ depth-domain model")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(rho_c.T, aspect="auto", cmap="viridis", origin="upper", extent=extent_depth)
    axes[0, 2].set_title(r"$\rho$ depth-domain model")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    for ia, ang in enumerate(ANGLES_DEG):
        im = axes[1, ia].imshow(stack[:, ia, :].T, aspect="auto", cmap="gray", origin="upper", extent=extent_time)
        axes[1, ia].set_title(rf"Synthetic stack {ang:g}$^\circ$")
        plt.colorbar(im, ax=axes[1, ia], fraction=0.046)

    for ax in axes.ravel():
        ax.set_xlabel("Distance (m)")

    for ax in axes[0, :]:
        ax.set_ylabel("Depth (m)")

    for ax in axes[1, :]:
        ax.set_ylabel("Time (ms)")

    plt.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    print(f"[SAVE] {out_png}")


def make_time_model_preview(
    vp_t,
    vs_t,
    rho_t,
    t_axis,
    out_png: Path,
    dt=0.001,
    win=29,
    dx=10.0,
    x0=11300.0,
    fontsize=12,
    interpolation="bilinear",
    dpi=300,
):
    """Save a true time-domain model preview aligned with the predicted section layout."""
    import os
    import matplotlib.pyplot as plt

    vp_t = np.asarray(vp_t, dtype=np.float32)
    vs_t = np.asarray(vs_t, dtype=np.float32)
    rho_t = np.asarray(rho_t, dtype=np.float32)

    # Basic shape checks prevent silently producing misleading figures.
    if vp_t.ndim != 2 or vs_t.ndim != 2 or rho_t.ndim != 2:
        print("[TRUE-MODEL][WARN] input model shape should be [L, Nt].")
        return False

    if vp_t.shape != vs_t.shape or vp_t.shape != rho_t.shape:
        print(
            f"[TRUE-MODEL][WARN] shape mismatch: "
            f"vp={vp_t.shape}, vs={vs_t.shape}, rho={rho_t.shape}"
        )
        return False

    L, Nt = vp_t.shape
    half_win = int(win) // 2

    # Match the valid prediction window used by the inversion code, where a
    # temporal window of length win removes half_win samples at both boundaries.
    if Nt > 2 * half_win:
        tau_start = half_win
        tau_end = Nt - half_win
    else:
        tau_start = 0
        tau_end = Nt

    tau_idx = np.arange(tau_start, tau_end, dtype=np.int64)

    vp_plot = vp_t[:, tau_idx]
    vs_plot = vs_t[:, tau_idx]
    rho_plot = rho_t[:, tau_idx]

    l_idx = np.arange(L, dtype=np.float32)
    x_vals = float(x0) + l_idx * float(dx)

    x_left = float(x_vals[0])
    x_right = float(x_vals[-1])

    t_ms = tau_idx.astype(np.float32) * float(dt) * 1000.0

    y_top = float(t_ms[0])
    y_bottom = float(t_ms[-1])

    extent = [x_left, x_right, y_bottom, y_top]

    print("========== TRUE MODEL FIG9 ALIGN DEBUG ==========")
    print(f"L={L}, Nt(raw)={Nt}, Nt(plot)={len(tau_idx)}")
    print(f"tau range = {tau_idx[0]} -> {tau_idx[-1]}")
    print(f"x Distance range = {x_left:.1f} -> {x_right:.1f}")
    print(f"y Time range = {y_top:.1f} -> {y_bottom:.1f} ms")
    print("================================================")

    def robust_limits(img2d, qlo=2.0, qhi=98.0):
        """Use percentile limits to reduce the effect of extreme color values."""
        x = img2d[np.isfinite(img2d)]
        if x.size < 10:
            if np.isfinite(img2d).any():
                return float(np.nanmin(img2d)), float(np.nanmax(img2d))
            return 0.0, 1.0

        vmin = float(np.percentile(x, qlo))
        vmax = float(np.percentile(x, qhi))

        if vmax <= vmin:
            vmin, vmax = float(np.nanmin(x)), float(np.nanmax(x))

        return vmin, vmax

    plt.rcParams.update({
        "font.size": fontsize,
        "axes.titlesize": fontsize + 2,
        "axes.labelsize": fontsize,
        "xtick.labelsize": max(fontsize - 1, 1),
        "ytick.labelsize": max(fontsize - 1, 1),
    })

    fig, axes = plt.subplots(3, 1, figsize=(10.0, 12.5), constrained_layout=True)

    data_list = [vp_plot, vs_plot, rho_plot]
    title_list = [r"$V_p$", r"$V_s$", r"$\rho$"]
    unit_list = ["m/s", "m/s", "g/cc"]

    for i, ax in enumerate(axes):
        img = data_list[i].T
        vmin, vmax = robust_limits(img, 2.0, 98.0)

        im = ax.imshow(
            img,
            aspect="auto",
            cmap="viridis",
            origin="upper",
            extent=extent,
            interpolation=interpolation,
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(title_list[i], pad=8)
        ax.set_ylabel("Time (ms)")
        ax.set_xlim(x_left, x_right)
        ax.set_ylim(y_bottom, y_top)

        if i == 2:
            ax.set_xlabel("Distance (m)")
        else:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

        cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.015)
        cb.set_label(unit_list[i], rotation=90, labelpad=6)
        ax.grid(False)

    out_png = os.path.abspath(str(out_png))
    out_dir = os.path.dirname(out_png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if os.path.exists(out_png):
        print(f"[SAVE] {out_png} size={os.path.getsize(out_png) / 1024:.1f} KB")
        return True

    print(f"[SAVE-FAIL] {out_png}")
    return False


def main():
    """Run the full Marmousi2 data-preparation workflow."""
    # Locate or unzip the original Marmousi2 SEG-Y files.
    segy_paths = {k: get_local_segy_path(k, fname) for k, fname in FILES.items()}

    vp = read_marmousi2_segy(segy_paths["vp"])
    vs = read_marmousi2_segy(segy_paths["vs"])
    rho = read_marmousi2_segy(segy_paths["rho"])

    print("[RAW SHAPE]", "VP", vp.shape, "VS", vs.shape, "RHO", rho.shape)
    vp, vs, rho = safe_units(vp, vs, rho)

    print("[RAW RANGE] VP", float(vp.min()), float(vp.max()))
    print("[RAW RANGE] VS", float(vs.min()), float(vs.max()))
    print("[RAW RANGE] RHO", float(rho.min()), float(rho.max()))

    # Crop the same physical window from all three elastic-parameter models.
    vp_c = crop_and_downsample(vp)
    vs_c = crop_and_downsample(vs)
    rho_c = crop_and_downsample(rho)

    print("[CROP SHAPE]", vp_c.shape)
    print("[CROP RANGE] VP", float(vp_c.min()), float(vp_c.max()))
    print("[CROP RANGE] VS", float(vs_c.min()), float(vs_c.max()))
    print("[CROP RANGE] RHO", float(rho_c.min()), float(rho_c.max()))

    dx_eff = DX0 * X_DECIM
    dz_eff = DZ0 * Z_DECIM

    # Convert cropped depth-domain models to a common uniform time axis.
    vp_t, vs_t, rho_t, t_axis = depth_to_time(vp_c, vs_c, rho_c, dz=dz_eff, dt=DT)
    print("[TIME SHAPE]", vp_t.shape, "dt=", DT, "tmax=", float(t_axis[-1]) if len(t_axis) else None)

    # Generate an angle-dependent reflectivity series and convolve it with the
    # wavelet to obtain synthetic prestack seismic data.
    wav = ricker(FDOM, DT, WAVELET_LEN_S)
    stack_angles = []
    rc_angles = []

    for ang in ANGLES_DEG:
        rc = aki_richards_reflectivity(vp_t, vs_t, rho_t, ang)
        seis = fftconvolve(rc, wav[None, :], mode="same").astype(np.float32)
        rc_angles.append(rc)
        stack_angles.append(seis)

    stack = np.stack(stack_angles, axis=1).astype(np.float32)
    rc_stack = np.stack(rc_angles, axis=1).astype(np.float32)

    raw_amp = float(np.percentile(np.abs(stack), STACK_PCT))
    print(f"[CHECK] raw Stack P{STACK_PCT} amplitude = {raw_amp:.6g}")

    if raw_amp < 1e-7:
        raise RuntimeError(
            "Synthetic stack amplitude is almost zero. "
            "Check that VP/VS/RHOB are not transposed and that dz/dx are correct."
        )

    # Normalize seismic amplitudes for stable neural-network training. The
    # normalization factor is saved in the auxiliary file.
    if NORMALIZE_STACK:
        stack = stack / raw_amp
        print(f"[NORM] Stack divided by P{STACK_PCT} amplitude = {raw_amp:.6g}")

    # Add the singleton section dimension to match the downstream data format.
    mod = np.stack([vp_t, vs_t, rho_t], axis=1).astype(np.float32)
    stack4d = stack[:, None, :, :]
    mod4d = mod[:, None, :, :]

    print("[OUTPUT] Stack", stack4d.shape)
    print("[OUTPUT] Mod  ", mod4d.shape)

    # Save MATLAB-compatible files used by the inversion scripts.
    sio.savemat(OUTDIR / "marmousi2_crop_stack4d_35hz.mat", {"Stack": stack4d})
    sio.savemat(OUTDIR / "marmousi2_crop_mod4d.mat", {"Mod": mod4d})
    sio.savemat(OUTDIR / "marmousi2_crop_aux.mat", {
        "vp_depth_crop": vp_c,
        "vs_depth_crop": vs_c,
        "rho_depth_crop": rho_c,
        "vp_time": vp_t,
        "vs_time": vs_t,
        "rho_time": rho_t,
        "t_axis": t_axis,
        "wav": wav,
        "angles_deg": np.array(ANGLES_DEG, dtype=np.float32),
        "rc_stack": rc_stack,
        "x_range_m": np.array([X_MIN_M, X_MAX_M], dtype=np.float32),
        "z_range_m": np.array([Z_MIN_M, Z_MAX_M], dtype=np.float32),
        "dx_eff": np.array([dx_eff], dtype=np.float32),
        "dz_eff": np.array([dz_eff], dtype=np.float32),
        "dt": np.array([DT], dtype=np.float32),
        "fdom": np.array([FDOM], dtype=np.float32),
        "stack_norm_amp": np.array([raw_amp], dtype=np.float32),
    })

    # Save quick-look figures for checking the cropped model and time-domain
    # alignment before running inversion experiments.
    make_preview(vp_c, vs_c, rho_c, stack, OUTDIR / "marmousi2_crop_preview.png")

    make_time_model_preview(
        vp_t,
        vs_t,
        rho_t,
        t_axis,
        out_png=OUTDIR / "true_time_domain_model_fig9_layout.png",
        dt=0.001,
        win=29,
        dx=10.0,
        x0=11300.0,
    )

    print("[SAVE]", OUTDIR / "marmousi2_crop_stack4d_35hz.mat")
    print("[SAVE]", OUTDIR / "marmousi2_crop_mod4d.mat")
    print("[SAVE]", OUTDIR / "marmousi2_crop_aux.mat")
    print("[DONE]")


if __name__ == "__main__":
    main()
