import gzip
import shutil
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import fftconvolve


# ============================================================
# Marmousi2 VP/VS/RHOB SEGY -> cropped synthetic angle-stack
# Output format for your BNN/VAIM code:
#   Stack: [L, T, A, N]
#   Mod:   [L, T, 3, N]
# where:
#   L = cropped horizontal samples
#   T = 1 for 2D Marmousi section
#   A = number of angles
#   N = time samples
# ============================================================

# 你的本地数据路径
# Windows 路径建议用 r"..."，避免反斜杠转义问题。
WORKDIR = Path(r"D:\pycharm\project\marmousi")
OUTDIR = WORKDIR / "marmousi2_cropped"
WORKDIR.mkdir(parents=True, exist_ok=True)
OUTDIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# 1. Local file names
# -------------------------
# 脚本会优先读取已解压的 .segy；如果不存在，会自动从 .segy.gz 解压。
FILES = {
    "vp": "vp_marmousi-ii.segy.gz",
    "vs": "vs_marmousi-ii.segy.gz",
    "rho": "density_marmousi-ii.segy.gz",
}

# -------------------------
# 2. Crop area
# -------------------------
# Marmousi2 full model is roughly x=0~17 km, z=0~3.5 km, dx=dz=1.25 m.
# This crop corresponds to a right-side structurally complex area, similar to your red box.
# 裁剪你截图里右上角红色方框区域
# 约对应完整 Marmousi2 模型中的 x=11.3~16.9 km, z=0.5~2.05 km
X_MIN_M = 11_300.0
X_MAX_M = 16_900.0
Z_MIN_M = 500.0      # remove water layer / very shallow part
Z_MAX_M = 2_050.0

# Downsample to keep the test area lightweight.
# dx_eff = DX0 * X_DECIM, dz_eff = DZ0 * Z_DECIM
X_DECIM = 8
Z_DECIM = 2

# segyio.open(...).trace[:] 读出的 shape 是 [n_traces, n_samples]
# 对 Marmousi2 model SEGY，一般对应 [x, z] = [13601, 2801]
# 因此这里保持原方向，不要转置。
DX0 = 1.25
DZ0 = 1.25

# -------------------------
# 3. Synthetic data parameters
# -------------------------
DT = 0.001
FDOM = 35.0

# 推荐 Marmousi2 三角度组合：近角 / 中角 / 远角，但避免 45° 过大。
# 记得你的训练代码里的 physics angle_degrees 也要保持一致。
ANGLES_DEG = [5.0, 15.0, 25.0]

WAVELET_LEN_S = 0.128
NORMALIZE_STACK = True
STACK_PCT = 99.5


# ============================================================
# Local file helpers
# ============================================================
def get_local_segy_path(name: str, filename: str) -> Path:
    """
    从 WORKDIR 读取本地 .segy.gz 或已解压的 .segy，不联网下载。
    """
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


# ============================================================
# Robust Marmousi2 SEGY reader
# ============================================================
def ibm2float(u: np.ndarray) -> np.ndarray:
    """
    Convert IBM 32-bit floating point words to IEEE float32.
    Marmousi2 AGL model SEGY usually uses sample format code 1, IBM float.
    """
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
    读取 AGL Marmousi2 model SEGY。

    关键点：
      - 你的文件读取后正确 shape 应为 [x, z] = [13601, 2801]
      - 不要转置
      - 支持 IBM float(format=1) 和 IEEE float(format=5)
    """
    path = Path(path)
    print(f"[READ] {path}")

    with open(path, "rb") as f:
        f.seek(3200)
        bin_hdr = f.read(400)

    ns = _read_u2_be(bin_hdr, 20)       # samples per trace
    fmt = _read_u2_be(bin_hdr, 24)      # sample format code

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

    # Marmousi2 正确方向应该是 [x,z]=[13601,2801]。
    # 如果读出来刚好反了，才自动转置；正常情况下不会转置。
    if arr.shape[0] < arr.shape[1]:
        print(f"[WARN] read shape={arr.shape}, auto transpose to [x,z].")
        arr = arr.T

    print(f"[READ DONE] shape={arr.shape}, min={float(arr.min()):.6g}, max={float(arr.max()):.6g}")
    return arr.astype(np.float32, copy=False)


# ============================================================
# Processing functions
# ============================================================
def crop_and_downsample(arr: np.ndarray) -> np.ndarray:
    ix0 = int(round(X_MIN_M / DX0))
    ix1 = int(round(X_MAX_M / DX0))
    iz0 = int(round(Z_MIN_M / DZ0))
    iz1 = int(round(Z_MAX_M / DZ0))

    ix0 = max(0, min(ix0, arr.shape[0] - 1))
    ix1 = max(ix0 + 1, min(ix1, arr.shape[0]))
    iz0 = max(0, min(iz0, arr.shape[1] - 1))
    iz1 = max(iz0 + 1, min(iz1, arr.shape[1]))

    out = arr[ix0:ix1:X_DECIM, iz0:iz1:Z_DECIM].astype(np.float32)
    print(
        f"[CROP] x=[{X_MIN_M:g},{X_MAX_M:g}]m -> ix=[{ix0},{ix1}), "
        f"z=[{Z_MIN_M:g},{Z_MAX_M:g}]m -> iz=[{iz0},{iz1}), shape={out.shape}"
    )
    return out


def ricker(fdom: float, dt: float, length_s: float = 0.128) -> np.ndarray:
    n = int(round(length_s / dt))
    if n % 2 == 0:
        n += 1
    t = (np.arange(n, dtype=np.float32) - n // 2) * dt
    x = np.pi * float(fdom) * t
    w = (1.0 - 2.0 * x**2) * np.exp(-x**2)
    w = w / (np.sqrt(np.sum(w * w)) + 1e-12)
    return w.astype(np.float32)


def depth_to_time(vp_z, vs_z, rho_z, dz: float, dt: float):
    """
    Convert depth-domain models [nx,nz] to time-domain [nx,nt].
    Uses two-way time from Vp.
    """
    nx, _ = vp_z.shape
    vp_safe = np.clip(vp_z.astype(np.float32), 1000.0, None)

    twt = 2.0 * np.cumsum(np.ones_like(vp_safe, dtype=np.float32) * dz / vp_safe, axis=1)
    tmax = float(np.min(twt[:, -1]))
    t_axis = np.arange(0.0, tmax, dt, dtype=np.float32)
    nt = len(t_axis)

    if nt < 20:
        raise RuntimeError(f"Too few time samples after depth_to_time: nt={nt}. Check crop/depth units.")

    vp_t = np.empty((nx, nt), dtype=np.float32)
    vs_t = np.empty((nx, nt), dtype=np.float32)
    rho_t = np.empty((nx, nt), dtype=np.float32)

    for ix in range(nx):
        vp_t[ix] = np.interp(t_axis, twt[ix], vp_z[ix]).astype(np.float32)
        vs_t[ix] = np.interp(t_axis, twt[ix], vs_z[ix]).astype(np.float32)
        rho_t[ix] = np.interp(t_axis, twt[ix], rho_z[ix]).astype(np.float32)

    return vp_t, vs_t, rho_t, t_axis


def aki_richards_reflectivity(vp, vs, rho, angle_deg: float) -> np.ndarray:
    """
    Aki-Richards PP reflectivity approximation.
    Input:  [nx, nt]
    Output: [nx, nt]
    """
    theta = np.deg2rad(float(angle_deg))

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

    a = 0.5 * (d_vp + d_rho)
    b = 0.5 * d_vp - 2.0 * (vsm / np.clip(vpm, 1e-6, None)) ** 2 * (2.0 * d_vs + d_rho)
    c = 0.5 * d_vp

    rc = a + b * sin2 + c * (tan2 - sin2)
    rc = np.pad(rc, ((0, 0), (1, 0)), mode="constant")
    return np.nan_to_num(rc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def safe_units(vp, vs, rho):
    """Convert VP/VS from km/s to m/s if needed; keep density as g/cc."""
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
    import matplotlib.pyplot as plt
    import numpy as np

    # =====================================================
    # 物理坐标
    # =====================================================
    dx_eff = DX0 * X_DECIM      # 10.0 m
    dz_eff = DZ0 * Z_DECIM      # 2.5 m

    nx = vp_c.shape[0]
    nz = vp_c.shape[1]
    nt = stack.shape[2]

    # 横向距离：用裁剪区真实距离
    x0 = X_MIN_M
    x1 = X_MIN_M + (nx - 1) * dx_eff

    # 深度坐标：上排属性模型
    z0 = Z_MIN_M
    z1 = Z_MIN_M + (nz - 1) * dz_eff

    # 时间坐标：下排合成地震
    t0_ms = 0.0
    t1_ms = (nt - 1) * DT * 1000.0

    extent_depth = [x0, x1, z1, z0]
    extent_time = [x0, x1, t1_ms, t0_ms]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    # =====================================================
    # 上排：深度域属性模型
    # =====================================================
    im0 = axes[0, 0].imshow(
        vp_c.T,
        aspect="auto",
        cmap="viridis",
        origin="upper",
        extent=extent_depth
    )
    axes[0, 0].set_title(r"$V_p$ depth-domain model")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(
        vs_c.T,
        aspect="auto",
        cmap="viridis",
        origin="upper",
        extent=extent_depth
    )
    axes[0, 1].set_title(r"$V_s$ depth-domain model")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    im2 = axes[0, 2].imshow(
        rho_c.T,
        aspect="auto",
        cmap="viridis",
        origin="upper",
        extent=extent_depth
    )
    axes[0, 2].set_title(r"$\rho$ depth-domain model")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # =====================================================
    # 下排：时间域合成地震
    # =====================================================
    for ia, ang in enumerate(ANGLES_DEG):
        im = axes[1, ia].imshow(
            stack[:, ia, :].T,
            aspect="auto",
            cmap="gray",
            origin="upper",
            extent=extent_time
        )
        axes[1, ia].set_title(rf"Synthetic stack {ang:g}$^\circ$")
        plt.colorbar(im, ax=axes[1, ia], fraction=0.046)

    # =====================================================
    # 坐标轴标签
    # =====================================================
    for ax in axes.ravel():
        ax.set_xlabel("Distance (m)")

    # 上排属性模型：深度域
    axes[0, 0].set_ylabel("Depth (m)")
    axes[0, 1].set_ylabel("Depth (m)")
    axes[0, 2].set_ylabel("Depth (m)")

    # 下排合成地震：时间域
    axes[1, 0].set_ylabel("Time (ms)")
    axes[1, 1].set_ylabel("Time (ms)")
    axes[1, 2].set_ylabel("Time (ms)")

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
    """
    True time-domain model preview aligned with Fig9 predicted section.

    不依赖 sec，适合在 Marmousi 裁剪转换代码中直接使用。

    对齐规则：
      1. 使用完整横向范围 L；
      2. 时间方向裁掉 win//2，上下边界与预测 Fig9 一致；
      3. x axis: Distance (m)；
      4. y axis: Time (ms)；
      5. 3 行 1 列竖排；
      6. 只在最下面显示 Distance (m)。
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    vp_t = np.asarray(vp_t, dtype=np.float32)      # [L, Nt]
    vs_t = np.asarray(vs_t, dtype=np.float32)
    rho_t = np.asarray(rho_t, dtype=np.float32)

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

    # ==========================================================
    # 1. 与预测 Fig9 对齐：裁掉上下 half_win
    #    win=29 时 half_win=14
    #    预测图通常对应 tau = 14 ~ Nt-15
    # ==========================================================
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

    # ==========================================================
    # 2. x 轴：Distance (m)，和预测图一致
    # ==========================================================
    l_idx = np.arange(L, dtype=np.float32)
    x_vals = float(x0) + l_idx * float(dx)

    x_left = float(x_vals[0])
    x_right = float(x_vals[-1])

    # ==========================================================
    # 3. y 轴：Time (ms)，直接用 tau_idx * dt
    #    不再依赖 t_axis，避免 t_axis 和预测 tau_list 不一致
    # ==========================================================
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

    fig, axes = plt.subplots(
        3, 1,
        figsize=(10.0, 12.5),
        constrained_layout=True
    )

    data_list = [vp_plot, vs_plot, rho_plot]
    title_list = [
        r"$V_p$",
        r"$V_s$",
        r"$\rho$",
    ]
    unit_list = ["m/s", "m/s", "g/cc"]

    for i, ax in enumerate(axes):
        img = data_list[i].T   # [Nt, L]
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

        # 强制和预测图一致
        ax.set_xlim(x_left, x_right)
        ax.set_ylim(y_bottom, y_top)

        # 只在最后一幅图显示横坐标
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
    else:
        print(f"[SAVE-FAIL] {out_png}")
        return False

# ============================================================
# Main
# ============================================================
def main():
    segy_paths = {k: get_local_segy_path(k, fname) for k, fname in FILES.items()}

    vp = read_marmousi2_segy(segy_paths["vp"])
    vs = read_marmousi2_segy(segy_paths["vs"])
    rho = read_marmousi2_segy(segy_paths["rho"])

    print("[RAW SHAPE]", "VP", vp.shape, "VS", vs.shape, "RHO", rho.shape)
    vp, vs, rho = safe_units(vp, vs, rho)

    print("[RAW RANGE] VP", float(vp.min()), float(vp.max()))
    print("[RAW RANGE] VS", float(vs.min()), float(vs.max()))
    print("[RAW RANGE] RHO", float(rho.min()), float(rho.max()))

    # 注意：这里不要转置。
    # 你上传的 Marmousi2 model 正确方向是 [x, z] = [13601, 2801]。

    vp_c = crop_and_downsample(vp)
    vs_c = crop_and_downsample(vs)
    rho_c = crop_and_downsample(rho)

    print("[CROP SHAPE]", vp_c.shape)
    print("[CROP RANGE] VP", float(vp_c.min()), float(vp_c.max()))
    print("[CROP RANGE] VS", float(vs_c.min()), float(vs_c.max()))
    print("[CROP RANGE] RHO", float(rho_c.min()), float(rho_c.max()))

    dx_eff = DX0 * X_DECIM
    dz_eff = DZ0 * Z_DECIM

    vp_t, vs_t, rho_t, t_axis = depth_to_time(vp_c, vs_c, rho_c, dz=dz_eff, dt=DT)
    print("[TIME SHAPE]", vp_t.shape, "dt=", DT, "tmax=", float(t_axis[-1]) if len(t_axis) else None)

    wav = ricker(FDOM, DT, WAVELET_LEN_S)
    stack_angles = []
    rc_angles = []

    for ang in ANGLES_DEG:
        rc = aki_richards_reflectivity(vp_t, vs_t, rho_t, ang)
        seis = fftconvolve(rc, wav[None, :], mode="same").astype(np.float32)
        rc_angles.append(rc)
        stack_angles.append(seis)

    # [L, A, N]
    stack = np.stack(stack_angles, axis=1).astype(np.float32)
    rc_stack = np.stack(rc_angles, axis=1).astype(np.float32)

    raw_amp = float(np.percentile(np.abs(stack), STACK_PCT))
    print(f"[CHECK] raw Stack P{STACK_PCT} amplitude = {raw_amp:.6g}")

    if raw_amp < 1e-7:
        raise RuntimeError(
            "Synthetic stack amplitude is almost zero. "
            "Check that VP/VS/RHOB are not transposed and that dz/dx are correct."
        )

    if NORMALIZE_STACK:
        stack = stack / raw_amp
        print(f"[NORM] Stack divided by P{STACK_PCT} amplitude = {raw_amp:.6g}")

    # [L, 3, N]
    mod = np.stack([vp_t, vs_t, rho_t], axis=1).astype(np.float32)

    # Your code format: [L, T, A, N], [L, T, 3, N]
    stack4d = stack[:, None, :, :]
    mod4d = mod[:, None, :, :]

    print("[OUTPUT] Stack", stack4d.shape)
    print("[OUTPUT] Mod  ", mod4d.shape)

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

    make_preview(vp_c, vs_c, rho_c, stack, OUTDIR / "marmousi2_crop_preview.png")
    from pathlib import Path

    from pathlib import Path

    out_dir = Path(r"D:\pycharm\project\marmousi\marmousi2_cropped")
    out_dir.mkdir(parents=True, exist_ok=True)

    make_time_model_preview(
        vp_t,
        vs_t,
        rho_t,
        t_axis,
        out_png=out_dir / "true_time_domain_model_fig9_layout.png",
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
